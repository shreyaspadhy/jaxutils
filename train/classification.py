"""Implementing Loss fns, train and eval steps for classification tasks."""

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn


def create_crossentropy_loss(
        model: nn.Module,
        batch_inputs: jnp.ndarray,
        batch_labels: jnp.ndarray,
        num_classes: int,
        accumulate: str = 'mean',
        **kwargs: Any) -> Callable:
    """Creates crossentropy loss used for classification tasks.

    Note: L2 regularisation (Gaussian priors on weights) is handled by the
    optimiser using `adamw`.

    Args:
        model: a ``nn.Module`` object defining the flax model
        batch_inputs: size (B, *) with batched inputs
        batch_labels: size (B, 1) with batched class labels
        num_classes: number of classes in the dataset
        accumulate: whether to accumulate using 'mean' or 'sum'
    Returns:
        Scalar containing mean (or sum) of loss over a minibatch.
    """

    def batched_loss_fn(params):
        # model.apply is already vectorised over the batch dimension.
        batch_logits = model.apply(params, batch_inputs)

        # optax.softmax_cross_entropy takes in one-hot labels
        labels_onehot = jax.nn.one_hot(labels, num_classes=num_classes)

        loss = optax.softmax_cross_entropy(batch_logits, labels_onehot)

        if accumulate == 'mean':
            return jnp.mean(loss, axis=0)
        elif accumulate == 'sum':
            return jnp.sum(loss, axis=0)

    return jax.jit(batched_loss_fn)


def create_train_step(model, optimizer, num_classes):
    @jax.jit
    def train_step(
            params, opt_state, batch_inputs, batch_labels):
        loss_fn = create_crossentropy_loss(
            model, batch_inputs, batch_labels, num_classes, accumulate='mean')
        loss_grad_fn = jax.value_and_grad(loss_fn)

        loss_val, grads = loss_grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss_val

    return train_step


def create_eval_step(model, num_classes):
    @jax.jit
    def eval_step(params, batch_inputs, batch_labels):
        loss_fn = create_crossentropy_loss(
            model, batch_inputs, batch_labels, num_classes, accumulate='sum')

        loss_val = loss_fn(params)
        return loss_val

    return pmf_eval_step
