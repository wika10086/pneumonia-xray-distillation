from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR
from dataset import build_transforms
from models import create_student_model, create_teacher_model
from utils import get_device


DEFAULT_CHECKPOINTS = [
    Path("checkpoints/teacher_best.pt"),
    Path("checkpoints/student_best.pt"),
    Path("checkpoints/distilled_student_best.pt"),
    Path("checkpoints/distilled_student_alpha03_t2_best.pt"),
]


def parse_args():
    parser = ArgumentParser(
        description="Evaluate trained models on the test split.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Dataset root")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loading workers")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--output-name", type=str, default="test_metrics.json", help="Output JSON file name")
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        default=DEFAULT_CHECKPOINTS,
        help="Checkpoint paths to evaluate",
    )
    return parser.parse_args()


def create_test_loader(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[DataLoader, list[str]]:
    test_dir = data_dir / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")

    _, test_transform = build_transforms(image_size=image_size, augment=False)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    return test_loader, test_dataset.classes


def create_model_from_checkpoint(checkpoint: dict[str, Any], num_classes: int) -> nn.Module:
    model_name = str(checkpoint.get("model_name", ""))
    if "resnet18" in model_name:
        return create_teacher_model(num_classes=num_classes, pretrained=False)
    if "mobilenet_v3_small" in model_name:
        return create_student_model(num_classes=num_classes, pretrained=False)
    raise ValueError(f"Unsupported model_name in checkpoint: {model_name}")


def update_confusion_matrix(
    matrix: list[list[int]],
    labels: torch.Tensor,
    predictions: torch.Tensor,
) -> None:
    for label, prediction in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
        matrix[int(label)][int(prediction)] += 1


def metrics_from_confusion_matrix(matrix: list[list[int]], class_names: list[str]) -> dict[str, Any]:
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[index][index] for index in range(len(class_names)))
    per_class = {}
    weighted_f1_sum = 0.0
    macro_f1_sum = 0.0

    for index, class_name in enumerate(class_names):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(len(class_names)) if row != index)
        fn = sum(matrix[index][col] for col in range(len(class_names)) if col != index)
        support = sum(matrix[index])

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        macro_f1_sum += f1
        weighted_f1_sum += f1 * support

    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": macro_f1_sum / len(class_names) if class_names else 0.0,
        "weighted_f1": weighted_f1_sum / total if total else 0.0,
        "total": total,
        "confusion_matrix": matrix,
        "confusion_matrix_rows": "true labels",
        "confusion_matrix_columns": "predicted labels",
        "class_names": class_names,
        "per_class": per_class,
    }


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    test_loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_classes = checkpoint.get("class_names")
    if saved_classes and list(saved_classes) != class_names:
        raise ValueError(
            f"Class mismatch for {checkpoint_path}: checkpoint={saved_classes}, dataset={class_names}"
        )

    model = create_model_from_checkpoint(checkpoint, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    matrix = [[0 for _ in class_names] for _ in class_names]
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        predictions = logits.argmax(dim=1)
        update_confusion_matrix(matrix, labels, predictions)

    metrics = metrics_from_confusion_matrix(matrix, class_names)
    metrics.update(
        {
            "checkpoint": str(checkpoint_path),
            "model_name": checkpoint.get("model_name"),
            "saved_epoch": checkpoint.get("epoch"),
            "best_val_acc": checkpoint.get("best_acc"),
            "alpha": checkpoint.get("alpha"),
            "temperature": checkpoint.get("temperature"),
        }
    )
    return metrics


def print_summary(results: list[dict[str, Any]]) -> None:
    print("Test set results")
    print("-" * 96)
    print(f"{'checkpoint':42} {'test_acc':>9} {'macro_f1':>9} {'normal_rec':>10} {'pneumonia_rec':>13}")
    print("-" * 96)
    for result in results:
        per_class = result["per_class"]
        print(
            f"{Path(result['checkpoint']).name:42} "
            f"{result['accuracy']:9.4f} "
            f"{result['macro_f1']:9.4f} "
            f"{per_class.get('normal', {}).get('recall', 0.0):10.4f} "
            f"{per_class.get('pneumonia', {}).get('recall', 0.0):13.4f}"
        )


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    test_loader, class_names = create_test_loader(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    print(f"Device: {device}")
    print(f"Classes: {class_names}")

    results = []
    for checkpoint_path in args.checkpoints:
        if not checkpoint_path.exists():
            print(f"Skip missing checkpoint: {checkpoint_path}")
            continue
        results.append(evaluate_checkpoint(checkpoint_path, test_loader, class_names, device))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(results)
    print(f"Saved metrics: {output_path}")


if __name__ == "__main__":
    main()
