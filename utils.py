import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    predictions = logits.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.numel()
    return correct, total


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_acc: float,
    class_names: list[str],
    model_name: str,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "best_acc": best_acc,
        "class_names": class_names,
        "num_classes": len(class_names),
        "model_name": model_name,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"找不到模型权重文件：{path}")
    return torch.load(path, map_location=device)

