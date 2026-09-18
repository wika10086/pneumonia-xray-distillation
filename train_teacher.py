from argparse import ArgumentParser

import torch
from torch import nn

from config import add_common_train_args
from dataset import create_dataloaders
from engine import evaluate, train_one_epoch
from models import create_teacher_model
from utils import get_device, save_checkpoint, save_json, set_seed


def parse_args():
    parser = ArgumentParser(description="训练教师模型 ResNet18")
    add_common_train_args(parser)
    parser.add_argument("--pretrained", action="store_true", help="使用 ImageNet 预训练权重")
    parser.add_argument("--checkpoint-name", type=str, default="teacher_best.pt", help="最佳模型文件名")
    parser.add_argument("--history-name", type=str, default="teacher_history.json", help="训练日志文件名")
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

    model = create_teacher_model(num_classes=len(class_names), pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_acc = 0.0
    history = []
    checkpoint_path = args.checkpoint_dir / args.checkpoint_name

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_acc": train_metrics.accuracy,
            "val_loss": val_metrics.loss,
            "val_acc": val_metrics.accuracy,
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {train_metrics.loss:.4f}, acc {train_metrics.accuracy:.4f} | "
            f"val loss {val_metrics.loss:.4f}, acc {val_metrics.accuracy:.4f}"
        )

        if val_metrics.accuracy > best_acc:
            best_acc = val_metrics.accuracy
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_acc,
                class_names=class_names,
                model_name="resnet18_teacher",
                extra={"pretrained": args.pretrained},
            )
            print(f"已保存最佳教师模型：{checkpoint_path}")

    save_json(args.output_dir / args.history_name, history)
    print(f"训练完成，最佳验证准确率：{best_acc:.4f}")


if __name__ == "__main__":
    main()
