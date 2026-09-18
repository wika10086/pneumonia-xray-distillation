from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
import torch
from torch import nn
import torch.nn.functional as F


def confidence_from_margin(margin: float) -> tuple[str, str]:
    if margin < 0.05:
        return "很低", "结果非常接近阈值，建议重点复核。"
    if margin < 0.15:
        return "较低", "结果离阈值较近，建议谨慎参考。"
    if margin < 0.30:
        return "中等", "结果和阈值有一定距离，但仍建议结合原图复核。"
    return "较高", "结果明显远离阈值，模型判断相对稳定。"


def build_probability_explanation(
    *,
    class_names: list[str],
    probabilities: list[float],
    positive_index: int,
    prediction_index: int,
    threshold: float,
    focus_regions: list[str] | None = None,
    heatmap_path: Path | None = None,
) -> dict[str, Any]:
    positive_name = class_names[positive_index]
    positive_probability = float(probabilities[positive_index])
    predicted_name = class_names[prediction_index]
    threshold_margin = abs(positive_probability - threshold)
    confidence_level, confidence_note = confidence_from_margin(threshold_margin)

    sorted_probs = sorted(
        ((class_names[index], float(probability)) for index, probability in enumerate(probabilities)),
        key=lambda item: item[1],
        reverse=True,
    )
    probability_gap = sorted_probs[0][1] - sorted_probs[1][1] if len(sorted_probs) >= 2 else 0.0

    if prediction_index == positive_index:
        threshold_reason = (
            f"{positive_name} 概率为 {positive_probability:.4f}，高于阈值 {threshold:.4f} "
            f"约 {threshold_margin:.4f}，因此模型输出 {predicted_name}。"
        )
    else:
        threshold_reason = (
            f"{positive_name} 概率为 {positive_probability:.4f}，低于阈值 {threshold:.4f} "
            f"约 {threshold_margin:.4f}，因此模型输出 {predicted_name}。"
        )

    reasons = [
        threshold_reason,
        f"两个类别的概率差约为 {probability_gap:.4f}；差值越大，模型越偏向当前类别。",
        f"置信度等级：{confidence_level}。{confidence_note}",
    ]

    if focus_regions:
        reasons.append(
            "Grad-CAM 显示模型主要关注图像的"
            + "、".join(focus_regions)
            + "；这只是算法关注区域，不等同于病灶定位。"
        )
    if heatmap_path is not None:
        reasons.append(f"热力图已保存到：{heatmap_path}")

    return {
        "predicted_class": predicted_name,
        "positive_class": positive_name,
        "positive_probability": positive_probability,
        "threshold": threshold,
        "threshold_margin": threshold_margin,
        "probability_gap": probability_gap,
        "confidence_level": confidence_level,
        "focus_regions": focus_regions or [],
        "heatmap_path": str(heatmap_path) if heatmap_path else "",
        "reason_summary": reasons[0],
        "reasons": reasons,
        "caution": "以上原因只解释模型输出，不能替代医生诊断。",
    }


def find_last_conv2d(model: nn.Module) -> nn.Conv2d:
    last_conv: nn.Conv2d | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("模型中没有找到 Conv2d 层，无法生成 Grad-CAM。")
    return last_conv


def generate_gradcam(model: nn.Module, image_tensor: torch.Tensor, class_index: int) -> torch.Tensor:
    target_layer = find_last_conv2d(model)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            grad_input = image_tensor.detach().clone().requires_grad_(True)
            logits = model(grad_input)
            score = logits[:, class_index].sum()
            score.backward()

        if not activations or not gradients:
            raise RuntimeError("没有捕获到模型激活或梯度。")

        activation = activations[-1]
        gradient = gradients[-1]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().float().cpu()

        cam_min = float(cam.min())
        cam_max = float(cam.max())
        if cam_max <= cam_min:
            return torch.zeros_like(cam)
        return (cam - cam_min) / (cam_max - cam_min)
    finally:
        forward_handle.remove()
        backward_handle.remove()


def describe_focus_regions(heatmap: torch.Tensor, max_regions: int = 2) -> list[str]:
    if heatmap.numel() == 0 or float(heatmap.max()) <= 0:
        return []

    row_names = ["上部", "中部", "下部"]
    col_names = ["左侧", "中央", "右侧"]
    height, width = heatmap.shape
    scores: list[tuple[float, str]] = []

    for row in range(3):
        top = row * height // 3
        bottom = (row + 1) * height // 3
        for col in range(3):
            left = col * width // 3
            right = (col + 1) * width // 3
            region_score = float(heatmap[top:bottom, left:right].mean())
            scores.append((region_score, f"{row_names[row]}{col_names[col]}"))

    scores.sort(reverse=True, key=lambda item: item[0])
    best_score = scores[0][0]
    if best_score <= 0:
        return []

    selected = [name for score, name in scores if score >= best_score * 0.85]
    return selected[:max_regions]


def safe_heatmap_name(image_path: Path) -> str:
    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:8]
    return f"{image_path.stem}_gradcam_{digest}.jpg"


def save_gradcam_overlay(original_image: Image.Image, heatmap: torch.Tensor, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    original = original_image.convert("RGB")
    width, height = original.size
    values = [int(max(0.0, min(1.0, float(value))) * 255) for value in heatmap.flatten().tolist()]
    heatmap_image = Image.new("L", tuple(reversed(heatmap.shape)))
    heatmap_image.putdata(values)

    resampling = getattr(Image, "Resampling", Image).BILINEAR
    heatmap_image = heatmap_image.resize((width, height), resampling)
    colored = ImageOps.colorize(heatmap_image, black="#00103a", mid="#ffe15a", white="#ff2b2b")
    overlay = Image.blend(original, colored, alpha=0.42)
    overlay.save(output_path, quality=92)
    return output_path
