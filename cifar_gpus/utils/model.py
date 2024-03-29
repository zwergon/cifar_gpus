import torchvision
import torch.nn as nn


def create_model():
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = nn.Identity()
    return model
