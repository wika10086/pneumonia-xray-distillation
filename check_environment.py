import sys

import torch
import torchaudio
import torchvision


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"torchaudio: {torchaudio.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    if cuda_available:
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    cpu_tensor = torch.randn(4, 4)
    cpu_result = cpu_tensor @ cpu_tensor.T
    print(f"CPU tensor test: {tuple(cpu_result.shape)} on {cpu_result.device}")

    device = torch.device("cuda" if cuda_available else "cpu")
    device_tensor = torch.randn(512, 512, device=device)
    device_result = device_tensor @ device_tensor
    if cuda_available:
        torch.cuda.synchronize()
    print(f"Device tensor test: {tuple(device_result.shape)} on {device_result.device}")


if __name__ == "__main__":
    main()
