"""Image dataset loading functionality."""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils import data
from torchvision import datasets, transforms

METADATA = {
    "image_shape": {
        "MNIST": (28, 28, 1),
        "FashionMNIST": (28, 28, 1),
        "KMNIST": (28, 28, 1),
        "SVHN": (32, 32, 3),
        "CIFAR10": (32, 32, 3),
        "CIFAR100": (32, 32, 3),
        "Imagenet": (224, 224, 3),
    },
    "num_train": {
        "MNIST": 60_000,
        "FashionMNIST": 60_000,
        "KMNIST": 60_000,
        "SVHN": 60_000,
        "CIFAR10": 60_000,
        "CIFAR100": 60_000,
        "Imagenet": 1_281_167,
    },
    "num_test": {
        "MNIST": 10_000,
        "FashionMNIST": 10_000,
        "KMNIST": 10_000,
        "SVHN": 10_000,
        "CIFAR10": 10_000,
        "CIFAR100": 10_000,
        "Imagenet": 100_000,
    },
    "num_classes": {
        "MNIST": 10,
        "FashionMNIST": 10,
        "KMNIST": 10,
        "SVHN": 10,
        "CIFAR10": 10,
        "CIFAR100": 100,
        "Imagenet": 1000,
    },
    "mean": {
        "MNIST": (0.1307,),
        "FashionMNIST": (0.2860,),
        "SVHN": (0.4377, 0.4438, 0.4728),
        "CIFAR10": (0.4914, 0.4822, 0.4465),
        "CIFAR100": (0.5071, 0.4866, 0.4409),
        "Imagenet": (0.485, 0.456, 0.406),
    },
    "std": {
        "MNIST": (0.3081,),
        "FashionMNIST": (0.3530,),
        "SVHN": (0.1980, 0.2010, 0.1970),
        "CIFAR10": (0.2470, 0.2435, 0.2616),
        "CIFAR100": (0.2673, 0.2564, 0.2762),
        "Imagenet": (0.229, 0.224, 0.225),
    },
}


TRAIN_TRANSFORMATIONS = {
    "MNIST": [transforms.RandomCrop(28, padding=2)],
    "FashionMNIST": [transforms.RandomCrop(28, padding=2)],
    "SVHN": [transforms.RandomCrop(32, padding=4)],
    "CIFAR10": [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ],
    "CIFAR100": [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ],
    "Imagenet": [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()],
}

TEST_TRANSFORMATIONS = {
    "MNIST": [],
    "FashionMNIST": [],
    "SVHN": [],
    "CIFAR10": [],
    "CIFAR100": [],
    "Imagenet": [transforms.Resize(256), transforms.CenterCrop(224)],
}


class Flatten:
    """Transform to flatten an image for use with MLPs."""

    def __call__(self, array: np.ndarray) -> np.ndarray:
        return np.ravel(array)


class MoveChannelDim:
    """Transform to change from PyTorch image ordering to Jax/TF ordering."""

    def __call__(self, array: np.ndarray) -> np.ndarray:
        return np.moveaxis(array, 0, -1)


class ToNumpy:
    """Transform to convert from a PyTorch Tensor to a Numpy ndarray."""

    def __call__(self, tensor: torch.Tensor) -> np.ndarray:
        return np.array(tensor, dtype=np.float32)


def get_image_dataset(
    dataset_name: str,
    data_dir: str = "../raw_data",
    flatten_img: bool = False,
    val_percent: float = 0.1,
    random_seed: int = 42,
    perform_augmentations: bool = True,
) -> Tuple[data.Dataset, data.Dataset, Optional[data.Dataset]]:
    """Provides PyTorch `Dataset`s for the specified image dataset_name.
    Args:
        dataset_name: the `str` name of the dataset. E.g. `'MNIST'`.
        data_dir: the directory where the datasets should be downloaded to and
          loaded from. (Default: `'../raw_data'`)
        flatten_img: a `bool` indicating whether images should be flattened.
            (Default: `False`)
        val_percent: the `float` percentage of training data to use for
            validation. (Default: `0.1`)
        random_seed: the `int` random seed for splitting the val data and
            applying random affine transformations. (Default: 42)
        perform_augmentations: a `bool` indicating whether to apply random
            transformations to the training data. (Default: `True`)
    Returns:
        `(train_dataset, test_dataset)` if `val_percent` is 0 otherwise
            `(train_dataset, test_dataset, val_dataset)`
    """
    dataset_choices = [
        "MNIST",
        "FashionMNIST",
        "SVHN",
        "CIFAR10",
        "CIFAR100",
        "Imagenet",
    ]
    if dataset_name not in dataset_choices:
        msg = (
            f"Dataset should be one of {dataset_choices} but was "
            + f"{dataset_name} instead."
        )
        raise RuntimeError(msg)

    if dataset_name in ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100", "Imagenet"]:
        train_kwargs = {"train": True}
        test_kwargs = {"train": False}

    elif dataset_name == "SVHN":
        train_kwargs = {"split": "train"}
        test_kwargs = {"split": "test"}

    data_dir = Path(data_dir).resolve()

    common_transforms = [
        ToNumpy(),
        MoveChannelDim(),
    ]

    if flatten_img:
        common_transforms += [Flatten()]

    # We need to disable train augmentations when fitting mode of linear model
    # and for sample-then-optimise posterior sampling.
    if perform_augmentations:
        train_augmentations = TRAIN_TRANSFORMATIONS[dataset_name]
    else:
        train_augmentations = TEST_TRANSFORMATIONS[dataset_name]

    transform_train = transforms.Compose(
        train_augmentations
        + [
            transforms.ToTensor(),
            transforms.Normalize(
                METADATA["mean"][dataset_name], METADATA["std"][dataset_name]
            ),
        ]
        + common_transforms
    )

    transform_test = transforms.Compose(
        TEST_TRANSFORMATIONS[dataset_name]
        + [
            transforms.ToTensor(),
            transforms.Normalize(
                METADATA["mean"][dataset_name], METADATA["std"][dataset_name]
            ),
        ]
        + common_transforms
    )

    if dataset_name == "Imagenet":
        train_dir = data_dir / "imagenet/train"
        val_dir = data_dir / "imagenet/validation"

        train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)

        test_dataset = datasets.ImageFolder(val_dir, transform=transform_test)

    else:
        dataset = getattr(datasets, dataset_name)
        train_dataset = dataset(
            **train_kwargs,
            transform=transform_train,
            download=True,
            root=data_dir,
        )
        test_dataset = dataset(
            **test_kwargs,
            transform=transform_test,
            download=True,
            root=data_dir,
        )

    if val_percent != 0.0:
        num_train, num_val = train_val_split_sizes(len(train_dataset), val_percent)

        train_dataset, val_dataset = data.random_split(
            train_dataset,
            [num_train, num_val],
            torch.Generator().manual_seed(random_seed)
            if random_seed is not None
            else None,
        )

        return train_dataset, test_dataset, val_dataset
    else:
        return train_dataset, test_dataset, None


def train_val_split_sizes(num_train: int, val_percent: float) -> Tuple[int, int]:
    num_val = int(val_percent * num_train)
    num_train = num_train - num_val

    return num_train, num_val
