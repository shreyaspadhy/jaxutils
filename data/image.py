"""Image dataset functionality, borrowed from https://github.com/JamesAllingham/learning-invariances/blob/main/src/data/image.py."""
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from torch.utils import data
from torchvision import datasets, transforms

from . import METADATA


TRAIN_TRANSFORMATIONS = {
    'MNIST': [transforms.RandomCrop(28, padding=2)],
    'FashionMNIST': [transforms.RandomCrop(28, padding=2)],
    'SVHN': [transforms.RandomCrop(32, padding=4)],
    'CIFAR10': [transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip()],
    'CIFAR100': [transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(), ],
    'Imagenet': [transforms.RandomResizedCrop(224),
                 transforms.RandomHorizontalFlip()],
}

TEST_TRANSFORMATIONS = {
    'MNIST': [],
    'FashionMNIST': [],
    'SVHN': [],
    'CIFAR10': [],
    'CIFAR100': [],
    'Imagenet': [transforms.Resize(256), transforms.CenterCrop(224)],
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
) -> Union[
    Tuple[data.Dataset, data.Dataset], Tuple[data.Dataset,
                                             data.Dataset, data.Dataset]
]:
    """Provides PyTorch `Dataset`s for the specified image dataset_name.
    Args:
        dataset_name: the `str` name of the dataset. E.g. `'MNIST'`.
        data_dir: the `str` directory where the datasets should be downloaded to
            and loaded from. (Default: `'../raw_data'`)
        flatten_img: a `bool` indicating whether images should be flattened.
            (Default: `False`)
        val_percent: the `float` percentage of training data to use for
            validation. (Default: `0.1`)
        random_seed: the `int` random seed for splitting the val data and
            applying random affine transformations. (Default: 42)
        perform_augmentations: a `bool` indicating whether to apply random
            affine transformations to the training data. (Default: `True`)
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
        msg = f"Dataset should be one of {dataset_choices} but was {dataset_name} instead."
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

    if perform_augmentations:
        train_augmentations = TRAIN_TRANSFORMATIONS[dataset_name]
    else:
        train_augmentations = TEST_TRANSFORMATIONS[dataset_name]
    transform_train = transforms.Compose(
        train_augmentations
        + [
            transforms.ToTensor(),
            transforms.Normalize(
                METADATA['mean'][dataset_name], METADATA['std'][dataset_name]),
        ]
        + common_transforms
    )

    transform_test = transforms.Compose(
        TEST_TRANSFORMATIONS[dataset_name]
        + [
            transforms.ToTensor(),
            transforms.Normalize(
                METADATA['mean'][dataset_name], METADATA['std'][dataset_name]),
        ]
        + common_transforms
    )

    if dataset_name == "Imagenet":
        train_dir = data_dir / "imagenet/train"
        val_dir = data_dir / "imagenet/val"

        train_dataset = datasets.ImageFolder(
            train_dir, transform=transform_train)

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
        num_train, num_val = train_val_split_sizes(
            len(train_dataset), val_percent)

        train_dataset, val_dataset = data.random_split(
            train_dataset,
            [num_train, num_val],
            torch.Generator().manual_seed(random_seed)
            if random_seed is not None
            else None,
        )

        return train_dataset, test_dataset, val_dataset
    else:
        return train_dataset, test_dataset


def train_val_split_sizes(num_train: int, val_percent: float) -> Tuple[int, int]:
    num_val = int(val_percent * num_train)
    num_train = num_train - num_val

    return num_train, num_val