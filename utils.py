"""Common utility functions."""

import collections
import os

from typing import Callable, Optional
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import tensorflow as tf
import torch
from flax.traverse_util import flatten_dict, ModelParamTraversal, unflatten_dict

import wandb


def setup_training(wandb_run):
    """Helper function that sets up training configs and logs to wandb."""
    # TF can hog GPU memory, so we hide the GPU device from it.
    tf.config.experimental.set_visible_devices([], "GPU")

    # Without this, JAX is automatically using 90% GPU for pre-allocation.
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
    # os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
    # Disable logging of compiles.
    jax.config.update("jax_log_compiles", False)

    # Log various JAX configs to wandb, and locally.
    wandb_run.summary.update({
        "jax_process_index": jax.process_index(),
        "jax.process_count": jax.process_count(), })


# Taken from https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
def flatten_nested_dict(nested_dict, parent_key='', sep='.'):
    items = []
    for name, cfg in nested_dict.items():
        new_key = parent_key + sep + name if parent_key else name
        if isinstance(cfg, collections.MutableMapping):
            items.extend(flatten_nested_dict(
                cfg, new_key, sep=sep).items())
        else:
            items.append((new_key, cfg))

    return dict(items)


def generate_keys(seed: int):
    """Generate keys for random number generators.
    Args:
        seed: Random seed.
    Returns:
        model_rng (jax.random.PRNGKey): JAX PRNG key.
        datasplit_rng (int): int for Pytorch random split.
    """
    rng = random.PRNGKey(seed)
    model_rng, datasplit_rng = jax.random.split(rng, 2)
    datasplit_rng = np.array(
        random.randint(datasplit_rng, [1, ], 0, 999999), dtype='int64').item(0)

    return model_rng, datasplit_rng


def log_model_params(
        params, wandb_run, as_summary: bool = False, summary_id: str = '',
        param_prefix: Optional[str] = None):
    """Log model parameter tensors as histograms to wandb.
    Args:
        params: Model parameters to log as histograms.
        wandb_run: Weights and Biases run object.
        as_summary: Whether to log as a wandb summary.
        summary_id: ID of the summary.
        param_prefix: Prefix to add to the parameter names.
    """
    # TODO: Check that params are not already flat
    params = flatten_dict(params)
    for param_name, param in params.items():
        log_param_name = '/'.join(param_name)
        if param_prefix is not None:
            log_param_name = param_prefix + '/' + log_param_name
        if as_summary:
            wandb_run.summary.update(
                {f"{log_param_name}_{summary_id}":
                    wandb.Histogram(np.asarray(param))})
        else:
            wandb_run.log(
                {log_param_name: wandb.Histogram(np.asarray(param))})

    params = unflatten_dict(params)


def get_agg_fn(agg: str) -> Callable:
    raise_if_not_in_list(agg, ['mean', 'sum'], 'aggregation')

    if agg == 'mean':
        return jnp.mean
    else:
        return jnp.sum


def tree_concatenate(list_of_trees):
    """Convert a list of trees of identical structure into a single tree of lists."""
    return jax.tree_map(lambda *xs: jnp.array(list(xs)), *list_of_trees)


def raise_if_not_in_list(val, valid_options, varname):
    if val not in valid_options:
        msg = f'`{varname}` should be one of `{valid_options}` but was `{val}` instead.'
        raise RuntimeError(msg)


def flatten_params(params):
    vals = []
    for k, v in flatten_dict(params).items():
        keys = '_'.join(k)
        if "BatchNorm" not in keys:
            vals.append(v.ravel())

    vals = jnp.concatenate(vals)

    return vals


def print_param_shapes(params):
    for k, v in flatten_dict(params).items():
        print(k, v.shape)


def print_params(params):
    for k, v in flatten_dict(params).items():
        print(k, v)
