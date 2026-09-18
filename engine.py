from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from utils import accuracy_from_logits


@dataclass
class Metrics:
    loss: float
    accuracy: float


@dataclass
class DistillMetrics:
    loss: float
    hard_loss: float
    soft_loss: float
    accuracy: float


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Metrics:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        correct, count = accuracy_from_logits(logits, labels)
        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += count

    return Metrics(loss=total_loss / total_samples, accuracy=total_correct / total_samples)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Metrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        correct, count = accuracy_from_logits(logits, labels)
        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += count

    return Metrics(loss=total_loss / total_samples, accuracy=total_correct / total_samples)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hard_loss = F.cross_entropy(student_logits, labels)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature**2)
    total_loss = (1 - alpha) * hard_loss + alpha * soft_loss
    return total_loss, hard_loss, soft_loss


def distill_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float,
    temperature: float,
) -> DistillMetrics:
    student.train()
    teacher.eval()

    total_loss = 0.0
    total_hard_loss = 0.0
    total_soft_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            teacher_logits = teacher(images)

        optimizer.zero_grad(set_to_none=True)
        student_logits = student(images)
        loss, hard_loss, soft_loss = distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            alpha=alpha,
            temperature=temperature,
        )
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        correct, count = accuracy_from_logits(student_logits, labels)
        total_loss += loss.item() * batch_size
        total_hard_loss += hard_loss.item() * batch_size
        total_soft_loss += soft_loss.item() * batch_size
        total_correct += correct
        total_samples += count

    return DistillMetrics(
        loss=total_loss / total_samples,
        hard_loss=total_hard_loss / total_samples,
        soft_loss=total_soft_loss / total_samples,
        accuracy=total_correct / total_samples,
    )

