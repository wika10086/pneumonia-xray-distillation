from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR
from dataset import build_transforms
from evaluate_models import create_model_from_checkpoint, metrics_from_confusion_matrix
from utils import get_device


def parse_args():
    parser = ArgumentParser(
        description="Tune a binary classification threshold on val and evaluate on test.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/distilled_student_alpha03_t2_best.pt"),
        help="Model checkpoint to evaluate",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Dataset root")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loading workers")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device")
    parser.add_argument("--target-class", type=str, default="pneumonia", help="Positive class name")
    parser.add_argument(
        "--min-sensitivity",
        type=float,
        default=0.99,
        help="Minimum target-class recall required when selecting the threshold",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--output-name", type=str, default="threshold_metrics.json", help="Output JSON file")
    return parser.parse_args()


def create_split_loader(
    data_dir: Path,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[DataLoader, list[str]]:
    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    _, transform = build_transforms(image_size=image_size, augment=False)
    dataset = datasets.ImageFolder(root=split_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    return loader, dataset.classes


def load_model(checkpoint_path: Path, class_names: list[str], device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_classes = checkpoint.get("class_names")
    if saved_classes and list(saved_classes) != class_names:
        raise ValueError(f"Class mismatch: checkpoint={saved_classes}, data={class_names}")

    model = create_model_from_checkpoint(checkpoint, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def collect_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
    positive_index: int,
    device: torch.device,
) -> tuple[list[float], list[int]]:
    probabilities: list[float] = []
    labels: list[int] = []

    for images, batch_labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, positive_index]
        probabilities.extend(float(value) for value in probs.cpu().tolist())
        labels.extend(int(value) for value in batch_labels.tolist())

    return probabilities, labels


def confusion_matrix_at_threshold(
    probabilities: list[float],
    labels: list[int],
    threshold: float,
    positive_index: int,
    negative_index: int,
) -> list[list[int]]:
    matrix = [[0, 0], [0, 0]]
    for probability, label in zip(probabilities, labels):
        prediction = positive_index if probability >= threshold else negative_index
        matrix[label][prediction] += 1
    return matrix


def binary_roc_auc(probabilities: list[float], labels: list[int], positive_index: int) -> float:
    positives = [(score, label) for score, label in zip(probabilities, labels) if label == positive_index]
    negatives = [(score, label) for score, label in zip(probabilities, labels) if label != positive_index]
    if not positives or not negatives:
        return 0.0

    ranked = sorted(zip(probabilities, labels), key=lambda item: item[0])
    rank_sum_positive = 0.0
    current_rank = 1
    index = 0
    while index < len(ranked):
        tie_end = index + 1
        while tie_end < len(ranked) and ranked[tie_end][0] == ranked[index][0]:
            tie_end += 1

        average_rank = (current_rank + current_rank + (tie_end - index) - 1) / 2
        for _, label in ranked[index:tie_end]:
            if label == positive_index:
                rank_sum_positive += average_rank

        current_rank += tie_end - index
        index = tie_end

    n_pos = len(positives)
    n_neg = len(negatives)
    return (rank_sum_positive - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def metric_bundle(
    probabilities: list[float],
    labels: list[int],
    threshold: float,
    class_names: list[str],
    positive_index: int,
    negative_index: int,
) -> dict[str, Any]:
    matrix = confusion_matrix_at_threshold(probabilities, labels, threshold, positive_index, negative_index)
    metrics = metrics_from_confusion_matrix(matrix, class_names)
    positive_name = class_names[positive_index]
    negative_name = class_names[negative_index]
    metrics.update(
        {
            "threshold": threshold,
            "roc_auc": binary_roc_auc(probabilities, labels, positive_index),
            "sensitivity": metrics["per_class"][positive_name]["recall"],
            "specificity": metrics["per_class"][negative_name]["recall"],
            "positive_class": positive_name,
            "negative_class": negative_name,
        }
    )
    return metrics


def threshold_candidates(probabilities: list[float]) -> list[float]:
    unique_scores = sorted(set(float(score) for score in probabilities))
    candidates = {0.0, 0.5, 1.0}
    candidates.update(unique_scores)
    for left, right in zip(unique_scores, unique_scores[1:]):
        candidates.add((left + right) / 2)
    return sorted(candidates)


def choose_threshold(
    probabilities: list[float],
    labels: list[int],
    class_names: list[str],
    positive_index: int,
    negative_index: int,
    min_sensitivity: float,
) -> dict[str, Any]:
    best = None
    fallback = None

    for threshold in threshold_candidates(probabilities):
        metrics = metric_bundle(probabilities, labels, threshold, class_names, positive_index, negative_index)

        fallback_key = (metrics["macro_f1"], metrics["accuracy"], metrics["sensitivity"], metrics["specificity"])
        if fallback is None or fallback_key > fallback[0]:
            fallback = (fallback_key, metrics)

        if metrics["sensitivity"] >= min_sensitivity:
            # Under the safety constraint, reduce false positives first.
            best_key = (metrics["specificity"], metrics["macro_f1"], metrics["accuracy"], threshold)
            if best is None or best_key > best[0]:
                best = (best_key, metrics)

    if best is not None:
        selected = best[1]
        selected["selection_rule"] = f"max specificity with sensitivity >= {min_sensitivity}"
        return selected

    selected = fallback[1]
    selected["selection_rule"] = "fallback max macro_f1 because sensitivity constraint was not reachable"
    return selected


def print_result(title: str, metrics: dict[str, Any]) -> None:
    print(title)
    print("-" * len(title))
    print(f"threshold:   {metrics['threshold']:.6f}")
    print(f"accuracy:    {metrics['accuracy']:.4f}")
    print(f"macro_f1:    {metrics['macro_f1']:.4f}")
    print(f"roc_auc:     {metrics['roc_auc']:.4f}")
    print(f"sensitivity: {metrics['sensitivity']:.4f}")
    print(f"specificity: {metrics['specificity']:.4f}")
    print(f"matrix:      {metrics['confusion_matrix']}")


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    val_loader, class_names = create_split_loader(
        args.data_dir, "val", args.image_size, args.batch_size, args.num_workers, device
    )
    test_loader, test_class_names = create_split_loader(
        args.data_dir, "test", args.image_size, args.batch_size, args.num_workers, device
    )
    if test_class_names != class_names:
        raise ValueError(f"Class mismatch: val={class_names}, test={test_class_names}")
    if len(class_names) != 2:
        raise ValueError(f"This script expects binary classification, got classes: {class_names}")
    if args.target_class not in class_names:
        raise ValueError(f"Target class {args.target_class!r} not found in {class_names}")

    positive_index = class_names.index(args.target_class)
    negative_index = 1 - positive_index
    model, checkpoint = load_model(args.checkpoint, class_names, device)

    val_probs, val_labels = collect_probabilities(model, val_loader, positive_index, device)
    test_probs, test_labels = collect_probabilities(model, test_loader, positive_index, device)

    val_default = metric_bundle(val_probs, val_labels, 0.5, class_names, positive_index, negative_index)
    val_selected = choose_threshold(
        val_probs, val_labels, class_names, positive_index, negative_index, args.min_sensitivity
    )
    test_default = metric_bundle(test_probs, test_labels, 0.5, class_names, positive_index, negative_index)
    test_selected = metric_bundle(
        test_probs,
        test_labels,
        val_selected["threshold"],
        class_names,
        positive_index,
        negative_index,
    )

    output = {
        "checkpoint": str(args.checkpoint),
        "model_name": checkpoint.get("model_name"),
        "saved_epoch": checkpoint.get("epoch"),
        "best_val_acc": checkpoint.get("best_acc"),
        "alpha": checkpoint.get("alpha"),
        "temperature": checkpoint.get("temperature"),
        "class_names": class_names,
        "target_class": args.target_class,
        "min_sensitivity": args.min_sensitivity,
        "val_default_threshold": val_default,
        "val_selected_threshold": val_selected,
        "test_default_threshold": test_default,
        "test_selected_threshold": test_selected,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Selection rule: {val_selected['selection_rule']}")
    print_result("Validation default threshold", val_default)
    print_result("Validation selected threshold", val_selected)
    print_result("Test default threshold", test_default)
    print_result("Test selected threshold", test_selected)
    print(f"Saved metrics: {output_path}")


if __name__ == "__main__":
    main()
