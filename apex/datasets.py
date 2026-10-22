from pathlib import Path
from torchvision import datasets, transforms

DATA_ROOT = Path("data")
_MNIST_MEAN, _MNIST_STD = ((0.1307,), (0.3081,))
_FMNIST_MEAN, _FMNIST_STD = ((0.286,), (0.353,))
_CIFAR10_MEAN, _CIFAR10_STD = ((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616))
_CIFAR100_MEAN, _CIFAR100_STD = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))


def get_transforms(name, train=True):
    name = name.lower()
    registry = {
        "cifar10": (_CIFAR10_MEAN, _CIFAR10_STD),
        "cifar100": (_CIFAR100_MEAN, _CIFAR100_STD),
        "mnist": (_MNIST_MEAN, _MNIST_STD),
        "fashion-mnist": (_FMNIST_MEAN, _FMNIST_STD),
    }
    mean, std = registry[name]
    augmentation = (
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        if train and name.startswith("cifar")
        else []
    )
    return transforms.Compose(
        augmentation + [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )


def get_dataset(name, train=True, root=None, download=False):
    registry = {
        "CIFAR10": datasets.CIFAR10,
        "CIFAR100": datasets.CIFAR100,
        "Fashion-MNIST": datasets.FashionMNIST,
        "MNIST": datasets.MNIST,
    }
    if name not in registry:
        raise ValueError(f"Unknown dataset: {name}")
    return registry[name](
        Path(root) if root else DATA_ROOT,
        train=train,
        download=download,
        transform=get_transforms(name, train),
    )
