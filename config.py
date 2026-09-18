from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path


DEFAULT_DATA_DIR = Path(r"D:\picdata")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_OUTPUT_DIR = Path("outputs")


def add_common_train_args(parser: ArgumentParser) -> None:
    parser.formatter_class = ArgumentDefaultsHelpFormatter
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="ImageFolder 数据集根目录")
    parser.add_argument("--image-size", type=int, default=224, help="输入图片会被缩放到这个尺寸")
    parser.add_argument("--batch-size", type=int, default=16, help="每次送入模型的图片数量")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减，用于轻微防止过拟合")
    parser.add_argument("--num-workers", type=int, default=0, help="读取图片的后台进程数量，Windows 初学阶段建议用 0")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，便于复现实验")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="训练设备")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR, help="模型权重保存目录")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="训练日志保存目录")
    parser.add_argument("--augment", action="store_true", help="开启简单数据增强")

