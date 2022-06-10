
__all__ = [
    'get_image_dataset',
    'train_val_split_sizes',
    'NumpyLoader',
    'METADATA',
]

from .image import get_image_dataset, train_val_split_sizes
from .utils import NumpyLoader

METADATA = {
    'image_shape': {
        'MNIST': (28, 28, 1),
        'FashionMNIST': (28, 28, 1),
        'KMNIST': (28, 28, 1),
        'SVHN': (32, 32, 3),
        'CIFAR10': (32, 32, 3),
        'CIFAR100': (32, 32, 3),
        'Imagenet': (224, 224, 3),
    },
    'num_train': {
        'MNIST': 60_000,
        'FashionMNIST': 60_000,
        'KMNIST': 60_000,
        'SVHN': 60_000,
        'CIFAR10': 60_000,
        'CIFAR100': 60_000,
    },
    'num_test': {
        'MNIST': 10_000,
        'FashionMNIST': 10_000,
        'KMNIST': 10_000,
        'SVHN': 10_000,
        'CIFAR10': 10_000,
        'CIFAR100': 10_000,
    },
    'mean': {
        'MNIST': (0.1307,),
        'FashionMNIST': (0.2860,),
        'SVHN': (0.4377, 0.4438, 0.4728),
        'CIFAR10': (0.4914, 0.4822, 0.4465),
        'CIFAR100': (0.5071, 0.4866, 0.4409),
        'Imagenet': (0.485, 0.456, 0.406),
    },
    'std': {
        'MNIST': (0.3081,),
        'FashionMNIST': (0.3530,),
        'SVHN': (0.1980, 0.2010, 0.1970),
        'CIFAR10': (0.2470, 0.2435, 0.2616),
        'CIFAR100': (0.2673, 0.2564, 0.2762),
        'Imagenet': (0.229, 0.224, 0.225),
        }
}