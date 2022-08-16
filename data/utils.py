"""Utility functions for agnostic datasets across Pytorch and TF."""
from typing import Iterator, Optional

import numpy as np
from flax.training.common_utils import shard


def get_agnostic_batch(
    batch: np.ndarray, dataset_type: str, tfds_keys: Optional[list] = None
) -> np.ndarray:
    if dataset_type == "pytorch":
        batch = shard(batch)
        # TODO: Find a nicer way to be agnostic to TF vs PyTorch
    if dataset_type == "tf":
        if tfds_keys is None:
            tfds_keys = ["image", "label"]
        if "label" in tfds_keys and "label_check" in tfds_keys:
            if batch["label"] != batch["label_check"]:
                raise ValueError(
                    "Label mismatch, check your dataset shuffling when "
                    "calculating y_samples."
                )
        batch = tuple([batch[k] for k in tfds_keys])
        # batch = (batch['image'], batch['label'])

    return batch


def get_agnostic_iterator(iterator: Iterator, dataset_type: str) -> Iterator:
    if dataset_type == "tf":
        iterator = iterator
    elif dataset_type == "pytorch":
        iterator = iter(iterator)

    return iterator
