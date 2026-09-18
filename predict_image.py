from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import json
from pathlib import Path
from typing import Any

from PIL import Image
import torch

from dataset import IMAGE_EXTENSIONS, build_transforms
from evaluate_models import create_model_from_checkpoint
from explain_prediction import (
    build_probability_explanation,
    describe_focus_regions,
    generate_gradcam,
    safe_heatmap_name,
    save_gradcam_overlay,
)
from medical_basis import build_medical_basis
from utils import get_device


DEFAULT_CHECKPOINT = Path("checkpoints/distilled_student_augmented_best.pt")
DEFAULT_THRESHOLD = 0.520043


def parse_args():
    parser = ArgumentParser(
        description="Predict normal/pneumonia for one image or a folder of images.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=Path, required=True, help="Image file or folder")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Model checkpoint")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Positive-class threshold")
    parser.add_argument("--target-class", type=str, default="pneumonia", help="Positive class name")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device")
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.json"), help="Output JSON path")
    parser.add_argument("--save-heatmaps", action="store_true", help="Save Grad-CAM heatmaps for model explanation")
    parser.add_argument("--heatmap-dir", type=Path, default=Path("outputs/heatmaps"), help="Grad-CAM output folder")
    return parser.parse_args()


def collect_image_paths(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path}")
        return [path]

    if path.is_dir():
        image_paths = [
            file_path
            for file_path in path.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not image_paths:
            raise FileNotFoundError(f"No supported images found in: {path}")
        return sorted(image_paths)

    raise FileNotFoundError(f"Image path does not exist: {path}")


def load_prediction_model(
    checkpoint_path: Path,
    target_class: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], list[str], int, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = list(checkpoint.get("class_names") or [])
    if len(class_names) != 2:
        raise ValueError(f"This prediction script expects 2 classes, got: {class_names}")
    if target_class not in class_names:
        raise ValueError(f"Target class {target_class!r} is not in checkpoint classes: {class_names}")

    positive_index = class_names.index(target_class)
    negative_index = 1 - positive_index

    model = create_model_from_checkpoint(checkpoint, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint, class_names, positive_index, negative_index


@torch.no_grad()
def predict_one_image(
    image_path: Path,
    model: torch.nn.Module,
    transform,
    class_names: list[str],
    positive_index: int,
    negative_index: int,
    threshold: float,
    device: torch.device,
    save_heatmap: bool = False,
    heatmap_dir: Path | None = None,
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()

    positive_probability = float(probabilities[positive_index])
    prediction_index = positive_index if positive_probability >= threshold else negative_index
    heatmap_path: Path | None = None
    focus_regions: list[str] = []
    heatmap_error = ""

    if save_heatmap:
        try:
            heatmap = generate_gradcam(model, image_tensor, prediction_index)
            focus_regions = describe_focus_regions(heatmap)
            output_dir = heatmap_dir or Path("outputs/heatmaps")
            heatmap_path = save_gradcam_overlay(image, heatmap, output_dir / safe_heatmap_name(image_path))
        except Exception as exc:
            heatmap_error = f"Grad-CAM 生成失败：{exc}"

    explanation = build_probability_explanation(
        class_names=class_names,
        probabilities=probabilities,
        positive_index=positive_index,
        prediction_index=prediction_index,
        threshold=threshold,
        focus_regions=focus_regions,
        heatmap_path=heatmap_path,
    )
    if heatmap_error:
        explanation["reasons"].append(heatmap_error)
    medical_basis = build_medical_basis(
        prediction=class_names[prediction_index],
        positive_probability=positive_probability,
        threshold=threshold,
        focus_regions=focus_regions,
    )

    return {
        "image": str(image_path),
        "prediction": class_names[prediction_index],
        "target_class": class_names[positive_index],
        "threshold": threshold,
        "target_probability": positive_probability,
        "reason_summary": explanation["reason_summary"],
        "explanation": explanation,
        "medical_basis_summary": medical_basis["summary"],
        "medical_basis": medical_basis,
        "probabilities": {
            class_name: float(probabilities[index])
            for index, class_name in enumerate(class_names)
        },
    }


def print_predictions(results: list[dict[str, Any]]) -> None:
    print("Predictions")
    print("-" * 110)
    print(f"{'prediction':12} {'pneumonia_prob':>14} {'threshold':>10} image")
    print("-" * 110)
    for result in results:
        print(
            f"{result['prediction']:12} "
            f"{result['target_probability']:14.6f} "
            f"{result['threshold']:10.6f} "
            f"{result['image']}"
        )
        print(f"  原因：{result['reason_summary']}")
        print(f"  医学参考：{result['medical_basis_summary']}")
        focus_regions = result.get("explanation", {}).get("focus_regions", [])
        if focus_regions:
            print(f"  关注区域：{'、'.join(focus_regions)}")


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    image_paths = collect_image_paths(args.image)
    _, transform = build_transforms(image_size=args.image_size, augment=False)
    model, checkpoint, class_names, positive_index, negative_index = load_prediction_model(
        args.checkpoint, args.target_class, device
    )

    results = [
        predict_one_image(
            image_path=image_path,
            model=model,
            transform=transform,
            class_names=class_names,
            positive_index=positive_index,
            negative_index=negative_index,
            threshold=args.threshold,
            device=device,
            save_heatmap=args.save_heatmaps,
            heatmap_dir=args.heatmap_dir,
        )
        for image_path in image_paths
    ]

    output = {
        "checkpoint": str(args.checkpoint),
        "model_name": checkpoint.get("model_name"),
        "class_names": class_names,
        "target_class": args.target_class,
        "threshold": args.threshold,
        "device": str(device),
        "count": len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print_predictions(results)
    print(f"Saved predictions: {args.output}")


if __name__ == "__main__":
    main()
