from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from PIL import Image


DATA_ROOT = Path(r"D:\picdata")
DATASET_URLS = {
    28: "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1",
    64: "https://zenodo.org/records/10519652/files/pneumoniamnist_64.npz?download=1",
    128: "https://zenodo.org/records/10519652/files/pneumoniamnist_128.npz?download=1",
    224: "https://zenodo.org/records/10519652/files/pneumoniamnist_224.npz?download=1",
}
CLASS_NAMES = {
    0: "normal",
    1: "pneumonia",
}


def parse_args():
    parser = ArgumentParser(description="Download and convert PneumoniaMNIST.")
    parser.add_argument("--size", type=int, default=224, choices=sorted(DATASET_URLS), help="Image size")
    return parser.parse_args()


def raw_path_for_size(size: int) -> Path:
    filename = "pneumoniamnist.npz" if size == 28 else f"pneumoniamnist_{size}.npz"
    return DATA_ROOT / "raw" / filename


def download_if_needed(size: int) -> Path:
    raw_path = raw_path_for_size(size)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists() and raw_path.stat().st_size > 0:
        print(f"Raw file already exists: {raw_path}")
        return raw_path

    dataset_url = DATASET_URLS[size]
    print(f"Downloading PneumoniaMNIST {size}x{size}: {dataset_url}")
    urlretrieve(dataset_url, raw_path)
    print(f"Downloaded: {raw_path}")
    return raw_path


def load_npz(size: int) -> np.lib.npyio.NpzFile:
    raw_path = download_if_needed(size)
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        raise FileNotFoundError(f"Raw file is missing or empty: {raw_path}")

    data = np.load(raw_path)
    image = data["train_images"][0].squeeze()
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError(f"Unexpected image shape: {image.shape}")
    actual_size = int(image.shape[0])
    if actual_size != size:
        raise ValueError(f"Expected image size {size}, got {actual_size}")

    print(f"Confirmed image size: {actual_size}x{actual_size}")
    return data


def reset_split_dirs() -> None:
    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES.values():
            class_dir = DATA_ROOT / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for image_path in class_dir.glob("*.png"):
                image_path.unlink()


def save_images(images: np.ndarray, labels: np.ndarray, split: str) -> dict[str, int]:
    counts = {class_name: 0 for class_name in CLASS_NAMES.values()}

    for index, (image_array, label_array) in enumerate(zip(images, labels)):
        label = int(np.asarray(label_array).reshape(-1)[0])
        class_name = CLASS_NAMES[label]
        class_dir = DATA_ROOT / split / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        image = Image.fromarray(image_array.squeeze().astype(np.uint8))
        image.save(class_dir / f"{split}_{index:05d}.png")
        counts[class_name] += 1

    return counts


def write_dataset_docs(summary: dict[str, dict[str, int]], size: int) -> None:
    metadata = {
        "name": "PneumoniaMNIST",
        "source": "MedMNIST+",
        "url": DATASET_URLS[size],
        "image_size": size,
        "task": "binary chest X-ray classification",
        "classes": CLASS_NAMES,
        "splits": summary,
    }
    metadata_path = DATA_ROOT / "pneumoniamnist_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = f"""# PneumoniaMNIST Dataset

Source: MedMNIST+ PneumoniaMNIST

Image size: {size}x{size}

Task: binary chest X-ray classification.

Classes:

- `normal`
- `pneumonia`

Directory structure:

```text
D:\\picdata
  train
    normal
    pneumonia
  val
    normal
    pneumonia
  test
    normal
    pneumonia
  raw
    {raw_path_for_size(size).name}
```

Image counts:

| Split | normal | pneumonia | total |
| --- | ---: | ---: | ---: |
| train | {summary["train"]["normal"]} | {summary["train"]["pneumonia"]} | {sum(summary["train"].values())} |
| val | {summary["val"]["normal"]} | {summary["val"]["pneumonia"]} | {sum(summary["val"].values())} |
| test | {summary["test"]["normal"]} | {summary["test"]["pneumonia"]} | {sum(summary["test"].values())} |
"""
    (DATA_ROOT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = load_npz(args.size)

    reset_split_dirs()
    summary = {
        "train": save_images(data["train_images"], data["train_labels"], "train"),
        "val": save_images(data["val_images"], data["val_labels"], "val"),
        "test": save_images(data["test_images"], data["test_labels"], "test"),
    }
    write_dataset_docs(summary, args.size)

    print(f"PneumoniaMNIST {args.size}x{args.size} converted.")
    for split, counts in summary.items():
        total = sum(counts.values())
        print(f"{split}: {counts} total={total}")


if __name__ == "__main__":
    main()
