"""Implementing Loss fns, train and eval steps for classification tasks."""

from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from jax.tree_util import tree_map

from jaxutils.utils import get_agg_fn


def create_crossentropy_loss(
    model: nn.Module,
    batch_inputs: jnp.ndarray,
    batch_labels: jnp.ndarray,
    num_classes: int,
    train: bool = False,
    aggregate: str = "mean",
    **kwargs: Any,
) -> Callable:
    """Creates crossentropy loss used for classification tasks.

    Note: L2 regularisation (Gaussian priors on weights) is handled by the
    optimiser using `adamw`.

    Args:
        model: a ``nn.Module`` object defining the flax model
        batch_inputs: size (B, *) with batched inputs
        batch_labels: size (B, 1) with batched class labels
        num_classes: number of classes in the dataset
        train: whether to mutate batchnorm statistics or not.
        aggregate: whether to aggregate using 'mean' or 'sum'
    Returns:
        Scalar containing mean (or sum) of loss over a minibatch.
    """

    def batched_loss_fn(params, model_state):
        # Get either mean or sum aggregations
        agg = get_agg_fn(aggregate)
        # model.apply is already vectorised over the  batch dimension.
        batch_logits, new_state = model.apply(
            {"params": params, **model_state},
            batch_inputs,
            train=train,
            mutable=list(model_state.keys()) if train else {},
        )
        batch_metrics = _create_loss_and_metrics(
            batch_logits, batch_labels, num_classes
        )

        batch_metrics = tree_map(lambda x: agg(x, axis=0), batch_metrics)

        loss = batch_metrics["nll"]

        return loss, (batch_metrics, new_state)

    return jax.jit(batched_loss_fn)


def create_train_step(model, optimizer, num_classes):
    @jax.jit
    def train_step(state, batch_inputs, batch_labels):
        loss_fn = create_crossentropy_loss(
            model, batch_inputs, batch_labels, num_classes, aggregate="sum", train=True
        )
        loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        (_, (train_metrics, model_state)), grads = loss_grad_fn(
            state.params, state.model_state
        )

        grads = jax.lax.pmean(grads, "device")
        train_metrics = jax.lax.pmean(train_metrics, "device")

        return (
            state.apply_gradients(grads=grads, model_state=model_state),
            train_metrics,
        )

    return train_step


def create_eval_step(model, num_classes):
    @jax.jit
    def eval_step(state, batch_inputs, batch_labels):
        loss_fn = create_crossentropy_loss(
            model, batch_inputs, batch_labels, num_classes, aggregate="sum", train=False
        )

        _, (eval_metrics, _) = loss_fn(state.params, state.model_state)

        eval_metrics = jax.lax.pmean(eval_metrics, "device")

        return eval_metrics

    return eval_step


def _create_loss_and_metrics(batch_logits, batch_labels, num_classes):
    # optax.softmax_cross_entropy takes in one-hot labels
    labels_onehot = jax.nn.one_hot(batch_labels, num_classes=num_classes)

    loss = optax.softmax_cross_entropy(batch_logits, labels_onehot)

    accuracy = jnp.argmax(batch_logits, -1) == batch_labels

    return {"nll": loss, "ll": -loss, "accuracy": accuracy, "error": 1 - accuracy}
