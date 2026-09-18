from torch import nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights


def create_teacher_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_student_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model

