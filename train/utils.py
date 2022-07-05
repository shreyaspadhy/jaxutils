from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from jax.tree_util import tree_map
from jaxutils.utils import tree_concatenate
import optax
import jax.numpy as jnp

import ml_collections
from typing import Optional


class TrainState(train_state.TrainState):
    """A Flax train state that also manages batch norm statistics."""
    model_state: FrozenDict

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )


def aggregated_metrics_dict(list_of_metrics_dicts, len_dataset, prefix):
    """Sum over list of batch_metrics dicts, and divide by dataset size."""
    metrics_dict = tree_map(
        lambda x: jnp.sum(x) / len_dataset,
        tree_concatenate(list_of_metrics_dicts))
    return {f'{prefix}/{k}': v for k, v in metrics_dict.items()}


def batchwise_metrics_dict(metrics_dict, batch_size, prefix):
    """Divide metrics_dict by batch_size."""
    new_metrics_dict = {
        f'{prefix}/{k}': v / batch_size for k, v in metrics_dict.items()}

    return new_metrics_dict


def get_lr_and_schedule(
        optim_name: str,
        optim_config: ml_collections.ConfigDict,
        lr_schedule_name: Optional[str],
        lr_schedule_config: Optional[ml_collections.ConfigDict]):
    """Returns an optimizer with (optional lr_schedule)."""
    if lr_schedule_name is not None:
        schedule = getattr(optax, lr_schedule_name)
        lr = schedule(
            init_value=optim_config.lr,
            decay_rate=lr_schedule_config.decay_rate,
            transition_steps=lr_schedule_config.transition_steps,)
    else:
        lr = optim_config.lr

    optimizer = getattr(optax, optim_name)
    optimizer = optax.inject_hyperparams(optimizer)
    optimizer = optimizer(learning_rate=lr)

    return optimizer
