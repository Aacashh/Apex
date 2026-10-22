import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


def _replace_bn_with_gn(module: nn.Module, num_groups: int = 8) -> None:
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)):
            ng = min(num_groups, child.num_features)
            while ng > 1 and child.num_features % ng != 0:
                ng -= 1
            gn = nn.GroupNorm(num_groups=ng, num_channels=child.num_features, affine=child.affine)
            if child.affine:
                gn.weight.data.copy_(child.weight.data)
                gn.bias.data.copy_(child.bias.data)
            setattr(module, name, gn)
        else:
            _replace_bn_with_gn(child, num_groups)


def _match_channels(x: torch.Tensor, expected: int) -> torch.Tensor:
    c = x.shape[1]
    if c == expected:
        return x
    if expected == 1 and c >= 3:
        r, g, b = (x[:, 0:1], x[:, 1:2], x[:, 2:3])
        gray = 0.2989 * r + 0.587 * g + 0.114 * b
        return gray
    if expected == 3 and c == 1:
        return x.repeat(1, 3, 1, 1)
    if c > expected:
        return x[:, :expected]
    rep = (expected + c - 1) // c
    x_rep = x.repeat(1, rep, 1, 1)
    return x_rep[:, :expected]


class CNNMnist(nn.Module):

    def __init__(self, num_classes: int = 10, in_channels: int = 1, image_size: int = 28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            h = F.max_pool2d(self.conv1(dummy), 2)
            h = F.max_pool2d(self.conv2(h), 2)
            flat_dim = int(h.numel() // h.shape[0])
        self.fc1 = nn.Linear(flat_dim, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _match_channels(x, self.conv1.in_channels)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_model(
    name, dataset, num_classes, device="cpu", pretrained=False, normalization="batchnorm"
):
    if pretrained:
        raise ValueError("Apex uses models trained from random initialization")
    channels, size = (3, 32) if dataset.startswith("CIFAR") else (1, 28)
    if name == "ResNet18" and channels == 3:
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "CNN-MNIST" and channels == 1:
        model = CNNMnist(num_classes, channels, size)
    else:
        raise ValueError("Use ResNet18 for CIFAR or CNN-MNIST for the grayscale datasets")
    if normalization == "groupnorm":
        _replace_bn_with_gn(model, 8)
    elif normalization != "batchnorm":
        raise ValueError("Unknown normalization")
    return model.to(device)
