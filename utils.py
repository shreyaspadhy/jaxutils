"""Common utility functions."""

import collections
import os

import jax
import jax.numpy as jnp
import jax.random as jrnd
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
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.50"

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
        seed (int): Random seed.
    Returns:
        model_rng (jax.random.PRNGKey): JAX PRNG key.
        datasplit_rng (int): int for Pytorch random split.
    """
    rng = jrnd.PRNGKey(seed)
    model_rng, datasplit_rng = jax.random.split(rng, 2)
    datasplit_rng = np.array(
        jrnd.randint(datasplit_rng, [1, ], 0, 999999), dtype='int64').item(0)

    return model_rng, datasplit_rng


def log_model_params(
        params, wandb_run, as_summary: bool = False, summary_id: str = ''):
    """Log model parameter tensors as histograms to wandb.
    Args:
        params (Pytree): Model parameters to log as histograms.
        wandb_run: Weights and Biases run object.
        as_summary (bool): Whether to log as a wandb summary.
    """
    # TODO: Check that params are not already flat
    params = flatten_dict(params)
    for param_name, param in params.items():
        if as_summary:
            wandb_run.summary.update(
                {f"{param_name}_{summary_id}":
                    wandb.Histogram(np.asarray(param))})
        else:
            wandb_run.log({param_name: wandb.Histogram(np.asarray(param))})

    params = unflatten_dict(params)
