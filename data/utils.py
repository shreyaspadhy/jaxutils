"""Utility functions for agnostic datasets across Pytorch and TF."""
import pickle
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from flax.training import checkpoints
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


def _get_agnostic_path(path: str, use_gcs=False):
    if not use_gcs:
        path = Path(path).resolve()
        path.mkdir(parents=True, exist_ok=True)
    return path

def restore_checkpoint(checkpoint_dir: str):
    """Restore a checkpoint from a given path, including gcs bucket."""
    checkpoint_dir = _get_agnostic_path(checkpoint_dir, use_gcs=checkpoint_dir.startswith('gs://'))

    checkpoint_path = checkpoints.latest_checkpoint(checkpoint_dir)
    restored_state = checkpoints.restore_checkpoint(checkpoint_path, target=None)

    return restored_state


def save_pickle(obj, path: str, storage_client=None):
    path = _get_agnostic_path(path, use_gcs=storage_client is not None)
    if storage_client is not None:
        bucket = storage_client.bucket('sampled_laplace_data')
        blob = bucket.blob(str(path))

        pickle_out = pickle.dumps(obj)
        blob.upload_from_string(pickle_out)
    
    else:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)


def load_pickle(path: str, storage_client=None):
    path = _get_agnostic_path(path, use_gcs=storage_client is not None)
    if storage_client is not None:
        bucket = storage_client.bucket('sampled_laplace_data')
        # strip "gs://sampled_laplace_data/" from path
        
        path = str(path[26:])
        print('path is ', path)
        blob = bucket.blob(path)

        pickle_in = blob.download_as_string()
        obj = pickle.loads(pickle_in)
    else:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
    
    return obj