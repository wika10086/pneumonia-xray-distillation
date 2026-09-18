from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR
from dataset import IMAGE_EXTENSIONS, build_transforms


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args():
    parser = ArgumentParser(
        description="Create a preview image for the current training augmentation.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Dataset root")
    parser.add_argument("--class-name", type=str, default="pneumonia", help="Class folder to sample from")
    parser.add_argument("--image-size", type=int, default=224, help="Preview image size")
    parser.add_argument("--samples", type=int, default=8, help="Number of augmented samples")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--output-name", type=str, default="augmentation_preview.png", help="Output image name")
    return parser.parse_args()


def find_first_image(data_dir: Path, class_name: str) -> Path:
    class_dir = data_dir / "train" / class_name
    if not class_dir.exists():
        raise FileNotFoundError(f"Class folder not found: {class_dir}")

    for path in sorted(class_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path
    raise FileNotFoundError(f"No images found in: {class_dir}")


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor.cpu() * STD + MEAN).clamp(0, 1)


def main() -> None:
    args = parse_args()
    image_path = find_first_image(args.data_dir, args.class_name)
    train_transform, val_transform = build_transforms(image_size=args.image_size, augment=True)

    image = Image.open(image_path).convert("RGB")
    original_tensor = denormalize(val_transform(image))
    augmented_tensors = [denormalize(train_transform(image)) for _ in range(args.samples)]

    grid = make_grid([original_tensor, *augmented_tensors], nrow=3, padding=8, pad_value=1.0)
    preview = to_pil_image(grid)

    draw = ImageDraw.Draw(preview)
    draw.rectangle((0, 0, preview.width, 24), fill="white")
    draw.text((8, 6), f"Original + {args.samples} augmented samples from {image_path}", fill="black")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    preview.save(output_path)
    print(f"Saved augmentation preview: {output_path}")


if __name__ == "__main__":
    main()
