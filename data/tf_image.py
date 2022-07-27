# Large amounts of this code has been taken from Uncertainty Baselines.
# Specifically, https://github.com/google/uncertainty-baselines/blob/b538c7fbecdb599cb5eb9299cfae0c5b1f06fb9c/baselines/jft/input_utils.py
"""tf Data input pipelines for optimised dataloading on TPU."""
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import flax
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging
from clu import preprocess_spec
from jaxutils.data import tf_preprocess

Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]
Features = Dict[str, Tensor]

# Random augmentations should happen before normalisation.
TRAIN_TRANSFORMATIONS = {
    "MNIST": "random_crop_with_pad(28, 2)|value_range(0, 1)|normalize((0.1307,), (0.3081,))"
}

TEST_TRANSFORMATIONS = {"MNIST": "value_range(0, 1)|normalize((0.1307,), (0.3081,))"}

COMMON_TRANSFORMATIONS = {"MNIST": "decode(1)"}

TF_TO_PYTORCH_NAMES = {"mnist": "MNIST"}

PYTORCH_TO_TF_NAMES = {"MNIST": "mnist"}


def get_dataset_builder(
    dataset_name: str,
    try_gcs: bool = False,
    data_dir: Optional[Union[str, Path]] = None,
) -> tfds.core.DatasetBuilder:
    """Get a tfds builder for a dataset, trying either GCS or a data_dir."""
    # TODO: If GCS fails, try the data_dir instead, defaulting to GCS.
    dataset_builder = tfds.builder(dataset_name, try_gcs=try_gcs, data_dir=data_dir)

    return dataset_builder


def get_process_split(
    split: str, process_index: int, process_count: int, drop_remainder: bool
) -> tfds.typing.SplitArg:
    """Returns the split for the given process given a multi-process setup.

    This function takes in a dataset split, and splits the dataset further across
    multiple processes, return `process_count` splits. Given a `process_index`, it
    then returns the corresponding split on the specific process.

    Notes:
      Unchanged from uncertainty_baselines."""
    splits = tfds.even_splits(split, n=process_count, drop_remainder=drop_remainder)
    process_split = splits[process_index]
    return process_split


def get_process_num_examples(
    builder: tfds.core.DatasetBuilder,
    split: str,
    process_batch_size: int,
    process_index: int,
    process_count: int,
    drop_remainder: bool,
) -> int:
    """Returns the number of examples in a given process's split.

    Notes:
      Unchanged from uncertainty_baselines.
    """
    process_split = get_process_split(
        split,
        process_index=process_index,
        process_count=process_count,
        drop_remainder=drop_remainder,
    )
    num_examples = builder.info.splits[process_split].num_examples

    if drop_remainder:
        device_batch_size = process_batch_size // jax.local_device_count()
        num_examples = math.floor(num_examples / device_batch_size) * device_batch_size

    return num_examples


def get_num_examples(
    dataset_builder: tfds.core.DatasetBuilder,
    split: str,
    process_batch_size: int,
    drop_remainder: bool = True,
    process_count: Optional[int] = None,
) -> int:
    """Returns the total number of examples in a (sharded) dataset split.

    Args:
      dataset: Either a dataset name or a dataset builder object.
      split: Specifies which split of the data to load.
      process_batch_size: Per process batch size.
      drop_remainder: Whether to drop remainders when sharding across processes
        and batching.
      process_count: Number of global processes (over all "hosts") across
        which the dataset will be sharded. If None, then the number of global
        processes will be obtained from `jax.process_count()`.
    Returns:
      The number of examples in the dataset split that will be read when sharded
      across available processes.
    """
    if process_count is None:
        process_count = jax.process_count()

    num_examples = 0
    for i in range(process_count):
        num_examples += get_process_num_examples(
            dataset_builder,
            split=split,
            process_batch_size=process_batch_size,
            process_index=i,
            process_count=process_count,
            drop_remainder=drop_remainder,
        )

    remainder = dataset_builder.info.splits[split].num_examples - num_examples
    if remainder:
        warning = (
            f"Dropping {remainder} examples from the {split} split of the "
            f"{dataset_builder.info.name} dataset."
        )
        logging.warning(warning)

    return num_examples


def preprocess_with_per_example_rng(
    ds: tf.data.Dataset,
    preprocess_fn: Callable[[Features], Features],
    *,
    rng: jnp.ndarray,
) -> tf.data.Dataset:
    """Maps `ds` using the preprocess_fn and a deterministic RNG per example.

    Args:
      ds: Dataset containing Python dictionary with the features. The 'rng'
        feature should not exist.
      preprocess_fn: Preprocessing function that takes a Python dictionary of
        tensors and returns a Python dictionary of tensors. The function should be
        convertible into a TF graph.
      rng: Base RNG to use. Per example RNGs will be derived from this by folding
        in the example index.
    Returns:
      The dataset mapped by the `preprocess_fn`.
    Notes:
      Unchanged from uncertainty_baselines.
    """

    def _fn(example_index: int, features: Features) -> Features:
        example_index = tf.cast(example_index, tf.int32)
        features["rng"] = tf.random.experimental.stateless_fold_in(
            tf.cast(rng, tf.int64), example_index
        )
        processed = preprocess_fn(features)
        if isinstance(processed, dict) and "rng" in processed:
            del processed["rng"]
        return processed

    return ds.enumerate().map(_fn, num_parallel_calls=tf.data.AUTOTUNE)


def get_image_dataset(
    dataset_name: str,
    process_batch_size: int,
    eval_process_batch_size: int,
    shuffle_train_split: bool = True,
    shuffle_eval_split: bool = True,
    drop_remainder: bool = True,
    process_index: Optional[int] = None,
    process_count: Optional[int] = None,
    data_dir: Optional[Union[str, Path]] = None,
    try_gcs: bool = False,
    val_percent: float = 0.1,
    perform_augmentations: bool = True,
    datasplit_rng: Optional[jnp.ndarray] = None,
):
    """Provides Tensorflow `Dataset`s for the specified image dataset_name.

     Args:
        dataset_name: the `str` name of the dataset. E.g. `'MNIST'`.
        process_batch_size: Per process batch size.
        eval_process_batch_size: Per process batch size for eval splits.
        shuffle_train_split: Whether to shuffle the train dataset (both on file
          & example level).
        shuffle_eval_split: Whether to shuffle the eval datasets.
        drop_remainder: Whether to drop remainders when batching and splitting
          across processes.
        process_index: Integer id in the range [0, process_count) of the current
          process in a multi-process setup. If None, then the index will be
          obtained from `jax.process_index()`.
        process_count: Number of global processes (over all "hosts") across
          which the dataset will be sharded. If None, then the number of global
          processes will be obtained from `jax.process_count()`.
        data_dir: the `str` directory where the datasets should be downloaded to
          and loaded from. (Default: `'../raw_data'`)
        try_gcs: a `bool` indicating whether to try to download the dataset from
          Google Cloud Storage. (Default: `False`). Only one of `try_gcs` or
          `data_dir` should be specified.
        val_percent: the `float` percentage of training data to use for
          validation. If the dataset contains a `validation` split on TFDS,
          then if `val_percent` is greater than 0, that split is returned,
          regardless of the percentage amount. Otherwise, x% of the train data
          is split and returned as val. (Default: `0.1`)
        perform_augmentations: a `bool` indicating whether to apply random
          transformations to the training data. (Default: `True`)
        datasplit_rng: A jax.random.PRNG key to use for seeding shuffle
          operations and preprocessing ops. Must be set if getting val_split.
    Returns:
        `datasets`: a Tuple of (train_dataset, Optional[val_dataset], test_dataset)
        `num_examples`: a Tuple of (train_num_examples, Optional[val_num_examples],
          test_num_examples).
    """
    dataset_choices = [
        "mnist",
    ]

    if dataset_name not in dataset_choices:
        msg = f"Dataset should be one of {dataset_choices} but was {dataset_name} instead."
        raise RuntimeError(msg)

    if process_index is None:
        process_index = jax.process_index()

    if process_count is None:
        process_count = jax.process_count()

    rng_available = datasplit_rng is not None
    if not rng_available and (shuffle_train_split or shuffle_eval_split):
        raise ValueError("Please set 'rng' when shuffling.")

    if rng_available:
        # We need 1 rng for the shuffle_seed, per process.
        rng = jax.random.fold_in(datasplit_rng, process_index)
    else:
        rng = [None, None]

    dataset_builder = get_dataset_builder(
        dataset_name, try_gcs=try_gcs, data_dir=data_dir
    )

    # Experimental threading and optimization for faster data loading.
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.threading.private_threadpool_size = 48
    dataset_options.threading.max_intra_op_parallelism = 1

    # Provide the explicit seed for shuffling the dataset here. This can control
    # the split between train and val (if there is one).
    # TODO: I'm not sure this changes the train/val split shuffling.
    read_config = tfds.ReadConfig(shuffle_seed=rng[0], options=dataset_options)

    # Create train, test, and optional val split strings. If val_percent is not
    # 0 and a validation split does not exist for dataset, we return the split
    # as `train[:x%]`, `train[x:%]` and `test`, otherwise if the val split is
    # present, we return the split as `train`, `val` and `test`.
    ds_splits = get_dataset_splits(dataset_builder, val_percent=val_percent)

    datasets = []
    split_num_examples = []

    # We follow the convention that split_num == 0 is train, rest are eval
    for split_num, split in enumerate(ds_splits):
        # pick either train or eval batch_size, and shuffling behaviour
        split_process_batch_size = (
            process_batch_size if split_num == 0 else eval_process_batch_size
        )
        shuffle = shuffle_train_split if split_num == 0 else shuffle_eval_split

        # Get the number of examples in the split, before sharding on processes.
        num_examples = get_num_examples(
            dataset_builder,
            split,
            split_process_batch_size,
            drop_remainder=drop_remainder,
        )
        split_num_examples.append(num_examples)

        # Further divide the split per process.
        process_split = get_process_split(
            split=split,
            process_index=process_index,
            process_count=process_count,
            drop_remainder=drop_remainder,
        )

        # The split per process is shuffled.
        ds = dataset_builder.as_dataset(
            split=process_split,
            shuffle_files=shuffle,
            read_config=read_config,
            decoders={"image": tfds.decode.SkipDecoding()},
        )

        datasets.append(ds)

    if val_percent == 0:
        # Insert None into the list in the middle if no validation splits.
        datasets.insert(1, None)
        split_num_examples.insert(1, None)

    return datasets, split_num_examples


def get_image_dataloader(
    dataset: tf.data.Dataset,
    dataset_name: str,
    process_batch_size: int,
    num_epochs: int,
    shuffle: bool = False,
    shuffle_buffer_size: int = 10_000,
    rng: Optional[jax.random.PRNGKey] = None,
    cache: Union[str, bool] = False,
    repeat_after_batching: bool = False,
    drop_remainder: bool = True,
    prefetch_size: int = 4,
    prefetch_on_device: Optional[int] = None,
    perform_augmentations: bool = False,
):
    """Return data iterator for a given TensorFlow dataset.

    Args:
        dataset: tf Dataset object.
        dataset_name: the `str` name of the dataset.
        process_batch_size: Per process batch size.
        num_epochs: Number of epochs for which to repeat the dataset. None
          to repeat forever.
        shuffle: Whether to shuffle the dataset (both on file & example level).
        shuffle_buffer_size: Number of examples in the shuffle buffer.
        rng: RNG to use for shuffling and deterministic augmentation.
        cache: Whether to cache the unprocessed dataset in memory before
          preprocessing and batching ("loaded"), after preprocessing & batching
          ("batched"), or not at all (False).
        repeat_after_batching: Whether to `repeat` the dataset before or after
          batching. Repeating after batching maintains the same batches, but
          shuffled across epochs.
        drop_remainder: Whether to drop remainders when batching and splitting
          across processes.
        prefetch_size: The number of elements in the final dataset to prefetch
          in the background. This should be a small (say <10) positive integer
          or tf.data.AUTOTUNE.
        prefetch_on_device: If not None, the number of elements to prefetch with
          Jax/Flax on device. For CPUs/TPUs, it is advisable to use None.

    Returns:
       dataloader: Python iterator of length `num_epochs * steps_per_epoch`.
    """
    if (shuffle or perform_augmentations) and rng is None:
        raise ValueError("Please set 'rng' when shuffling or performing augmentations.")

    if rng is not None:
        shuffle_rng, augment_rng = jax.random.split(rng, 2)

    if cache == "loaded":
        dataset = dataset.cache()

    if shuffle:
        # NOTE: The original uncertainty_baselines code did not set the
        # reshuffle_each_iteration value to True, but this results in the same
        # batches being produced across epochs, and no further reshuffling b/w
        # epochs. Empirically, this does not impact performance too much on the
        # MNIST dataset.
        dataset = dataset.shuffle(
            shuffle_buffer_size, seed=shuffle_rng[0], reshuffle_each_iteration=True
        )

    if not repeat_after_batching:
        dataset = dataset.repeat(num_epochs)

    # Perform augmnetations after repeat(), to have different augs each epoch.
    pp_split = COMMON_TRANSFORMATIONS[dataset_name]
    if perform_augmentations:
        add_pp = TRAIN_TRANSFORMATIONS[dataset_name]
        pp_split = pp_split + "|" + add_pp
    else:
        add_pp = TEST_TRANSFORMATIONS[dataset_name]
        if add_pp is not None:
            pp_split = pp_split + "|" + add_pp

    # Define Preprocessing Functions for the Datasets using clu.
    preprocess_fn = preprocess_spec.parse(
        spec=pp_split, available_ops=tf_preprocess.all_ops()
    )

    mask_fn = lambda ex: dict(mask=1.0, **ex)
    if preprocess_fn is not None:
        preprocess_and_mask_fn = lambda ex: mask_fn(preprocess_fn(ex))
    else:
        preprocess_and_mask_fn = mask_fn

    dataset = preprocess_with_per_example_rng(
        dataset, preprocess_and_mask_fn, rng=augment_rng
    )

    num_devices = jax.local_device_count()
    batch_size_per_device = process_batch_size // num_devices

    if not drop_remainder:
        # If we're not dropping the remainder, then we append additional zero
        # valued examples with zero-valued masks to the dataset such that
        # batching with drop_remainder=True will yield a dataset whose final
        # batch is padded as needed.
        # NOTE: We're batching the dataset over two dimensions,
        # `batch_size_per_device` and `num_devices`. Therefore, adding
        # `batch_size_per_device*num_devices - 1` padding examples covers the
        # worst case of 1 example left over after the first batching application
        # with batch size `batch_size_per_device` (since we'd need
        # `batch_size_per_device*num_devices - 1` additional examples).
        padding_example = tf.nest.map_structure(
            lambda spec: tf.zeros(spec.shape, spec.dtype)[None], dataset.element_spec
        )
        padding_example["mask"] = [0.0]
        padding_dataset = tf.data.Dataset.from_tensor_slices(padding_example)
        dataset = dataset.concatenate(
            padding_dataset.repeat(batch_size_per_device * num_devices - 1)
        )

    batch_dims = [num_devices, batch_size_per_device]
    for batch_size in reversed(batch_dims):
        dataset = dataset.batch(batch_size, drop_remainder=True)

    if cache == "batched":
        dataset = dataset.cache()

    if repeat_after_batching:
        dataset = dataset.repeat(num_epochs)

    dataset = dataset.prefetch(prefetch_size)

    # Create a Python iterator object, then memoryview the tf Tensors to numpy,
    # and optionally prefetch some elements on the device.
    iterator = iter(dataset)

    def _prepare(x):
        # Transforms x into read-only numpy array without copy if possible, see:
        # https://github.com/tensorflow/tensorflow/issues/33254#issuecomment-542379165
        return np.asarray(memoryview(x))

    iterator = (jax.tree_map(_prepare, xs) for xs in iterator)

    if prefetch_on_device:
        iterator = flax.jax_utils.prefetch_to_device(
            iterator, prefetch_on_device, devices=None
        )

    return iterator


def get_dataset_splits(
    dataset_builder: tfds.core.DatasetBuilder, val_percent: float = 0.0
) -> List[str]:
    """Return split strings for train, test, and optionally val."""
    if val_percent > 0.0:
        if "validation" in dataset_builder.info.splits.keys():
            return ["train", "validation", "test"]
        else:
            train_percent = int(100.0 * (1.0 - val_percent))
            return [f"train[:{train_percent}%]", f"train[{train_percent}%:]", "test"]
    else:
        return ["train", "test"]
