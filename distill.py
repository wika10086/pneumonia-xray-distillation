from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn

from config import add_common_train_args
from dataset import create_dataloaders
from engine import distill_one_epoch, evaluate
from models import create_student_model, create_teacher_model
from utils import get_device, load_checkpoint, save_checkpoint, save_json, set_seed


def parse_args():
    parser = ArgumentParser(description="使用教师模型蒸馏训练学生模型")
    add_common_train_args(parser)
    parser.add_argument("--teacher-checkpoint", type=Path, default=Path("checkpoints/teacher_best.pt"), help="教师模型权重路径")
    parser.add_argument("--student-pretrained", action="store_true", help="学生模型使用 ImageNet 预训练权重")
    parser.add_argument("--alpha", type=float, default=0.7, help="蒸馏损失权重，越大越依赖教师模型")
    parser.add_argument("--temperature", type=float, default=4.0, help="蒸馏温度，常用 2 到 8")
    parser.add_argument("--checkpoint-name", type=str, default="distilled_student_best.pt", help="最佳蒸馏学生模型文件名")
    parser.add_argument("--history-name", type=str, default="distill_history.json", help="训练日志文件名")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"使用设备：{device}")

    train_loader, val_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
    )
    print(f"类别：{class_names}")

    teacher_checkpoint = load_checkpoint(args.teacher_checkpoint, device)
    saved_classes = teacher_checkpoint.get("class_names")
    if saved_classes and list(saved_classes) != class_names:
        raise ValueError(
            "教师模型保存时的类别顺序和当前数据集不一致。\n"
            f"教师模型类别：{saved_classes}\n"
            f"当前数据集类别：{class_names}"
        )

    teacher = create_teacher_model(num_classes=len(class_names), pretrained=False).to(device)
    teacher.load_state_dict(teacher_checkpoint["model_state"])
    teacher.eval()

    student = create_student_model(num_classes=len(class_names), pretrained=args.student_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_acc = 0.0
    history = []
    checkpoint_path = args.checkpoint_dir / args.checkpoint_name

    for epoch in range(1, args.epochs + 1):
        train_metrics = distill_one_epoch(
            student=student,
            teacher=teacher,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            alpha=args.alpha,
            temperature=args.temperature,
        )
        val_metrics = evaluate(student, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_hard_loss": train_metrics.hard_loss,
            "train_soft_loss": train_metrics.soft_loss,
            "train_acc": train_metrics.accuracy,
            "val_loss": val_metrics.loss,
            "val_acc": val_metrics.accuracy,
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"distill loss {train_metrics.loss:.4f}, acc {train_metrics.accuracy:.4f} | "
            f"val loss {val_metrics.loss:.4f}, acc {val_metrics.accuracy:.4f}"
        )

        if val_metrics.accuracy > best_acc:
            best_acc = val_metrics.accuracy
            save_checkpoint(
                path=checkpoint_path,
                model=student,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_acc,
                class_names=class_names,
                model_name="mobilenet_v3_small_distilled_student",
                extra={
                    "teacher_checkpoint": str(args.teacher_checkpoint),
                    "student_pretrained": args.student_pretrained,
                    "alpha": args.alpha,
                    "temperature": args.temperature,
                },
            )
            print(f"已保存最佳蒸馏学生模型：{checkpoint_path}")

    save_json(args.output_dir / args.history_name, history)
    print(f"蒸馏完成，最佳验证准确率：{best_acc:.4f}")


if __name__ == "__main__":
    main()
