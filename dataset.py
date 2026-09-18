from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _check_split_dir(split_dir: Path, split_name: str) -> None:
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Cannot find {split_name} directory: {split_dir}\n"
            f"Expected structure example: {split_dir}\\normal\\001.jpg"
        )

    class_dirs = [path for path in split_dir.iterdir() if path.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(
            f"No class folders found in {split_name}: {split_dir}\n"
            f"Example: {split_dir}\\normal\\001.jpg and {split_dir}\\pneumonia\\001.jpg"
        )

    has_image = False
    for class_dir in class_dirs:
        for file_path in class_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                has_image = True
                break
        if has_image:
            break

    if not has_image:
        raise FileNotFoundError(
            f"No image files found in {split_name}: {split_dir}\n"
            f"Supported formats: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )


def build_train_augmentation() -> list[transforms.Transform]:
    # Keep augmentation gentle for chest X-rays. Avoid horizontal flips because left/right anatomy matters.
    return [
        transforms.RandomApply(
            [
                transforms.RandomAffine(
                    degrees=7,
                    translate=(0.03, 0.03),
                    scale=(0.95, 1.05),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                )
            ],
            p=0.75,
        ),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.08, contrast=0.08)],
            p=0.50,
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))],
            p=0.10,
        ),
    ]


def build_transforms(image_size: int, augment: bool) -> tuple[transforms.Compose, transforms.Compose]:
    train_steps: list[transforms.Transform] = [
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
    ]
    if augment:
        train_steps.extend(build_train_augmentation())

    common_steps = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]

    train_transform = transforms.Compose(train_steps + common_steps)
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            *common_steps,
        ]
    )
    return train_transform, val_transform


def create_dataloaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    augment: bool,
) -> tuple[DataLoader, DataLoader, list[str]]:
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    _check_split_dir(train_dir, "train")
    _check_split_dir(val_dir, "val")

    train_transform, val_transform = build_transforms(image_size=image_size, augment=augment)
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            "Class folders in train and val are inconsistent.\n"
            f"train classes: {train_dataset.classes}\n"
            f"val classes: {val_dataset.classes}"
        )

    pin_memory = False
    try:
        import torch

        pin_memory = torch.cuda.is_available()
    except Exception:
        pin_memory = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, train_dataset.classes
