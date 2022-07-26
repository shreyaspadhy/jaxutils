from typing import Callable, Iterable, Optional

import jax.numpy as jnp
import ml_collections
import optax
from flax.core.frozen_dict import FrozenDict
from flax import jax_utils
from flax.training import train_state
from flax.jax_utils import unreplicate
from flax.training.common_utils import shard
from jax.tree_util import tree_map
from jaxutils.data.utils import NumpyLoader, get_agnostic_batch
from jaxutils.utils import tree_concatenate
from functools import partial
import wandb
from itertools import islice
from tqdm import tqdm
import matplotlib.pyplot as plt


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
    
    def replicate(self):
        return jax_utils.replicate(self)

    def unreplicate(self):
        return jax_utils.unreplicate(self)


def aggregated_metrics_dict(
    list_of_metrics_dicts, len_dataset, prefix, n_devices=1):
    """Sum over list of batch_metrics dicts, and divide by dataset size."""
    metrics_dict = tree_map(
        lambda x: jnp.sum(x) * n_devices / len_dataset,
        tree_concatenate(list_of_metrics_dicts))
    return {f'{prefix}/{k}': v for k, v in metrics_dict.items()}


def batchwise_metrics_dict(metrics_dict, batch_size, prefix):
    """Divide metrics_dict by batch_size."""
    new_metrics_dict = {
        f'{prefix}/{k}': v / batch_size for k, v in metrics_dict.items()}

    return new_metrics_dict


def train_epoch(
    train_step_fn: Callable,
    data_iterator: Iterable,
    steps_per_epoch: int,
    num_points: int,
    state: TrainState, 
    wandb_run: wandb.run,
    log_prefix: str,
    dataset_type: str = "tf",
    log_global_metrics: bool = False,
    epoch=0,
    ):
    assert dataset_type in ["tf", "pytorch"]
    batch_metrics = []
    for i in range(steps_per_epoch):
        batch = next(data_iterator)
        batch = get_agnostic_batch(batch, dataset_type)
        n_devices, B = batch[0].shape[:2]

        if log_global_metrics:
            state, metrics, global_metrics = train_step_fn(state, *batch)
            wandb_run.log(unreplicate(global_metrics))
        else:
            state, metrics = train_step_fn(state, *batch)
        # These metrics are summed over each sharded batch, and averaged
        # over the num_devices. We need to multiply by num_devices to sum
        # over the entire dataset correctly, and then aggregate.
        metrics = unreplicate(metrics)
        batch_metrics.append(metrics)
        # Further divide by sharded batch size, to get average metrics
        wandb_run.log(
            batchwise_metrics_dict(metrics, B, f'{log_prefix}/batchwise'))

    epoch_metrics = aggregated_metrics_dict(
        batch_metrics, num_points, log_prefix, 
        n_devices=n_devices)
    wandb_run.log(epoch_metrics)
        
    if log_global_metrics:
        return state, epoch_metrics, global_metrics
    else:
        return state, epoch_metrics
    

def eval_epoch(
    eval_step_fn: Callable,
    data_iterator: Iterable,
    steps_per_epoch: int,
    num_points: int,
    state: TrainState, 
    wandb_run: wandb.run,
    log_prefix: str,
    dataset_type: str = "tf",
    log_global_metrics: bool = False,
    ):
    assert dataset_type in ["tf", "pytorch"]
    batch_metrics = []
    for i in range(steps_per_epoch):
        batch = next(data_iterator)
        batch = get_agnostic_batch(batch, dataset_type)
        n_devices, B = batch[0].shape[:2]

        if log_global_metrics:
            metrics, global_metrics = eval_step_fn(state, *batch)
            wandb_run.log(unreplicate(global_metrics))
        else:
            metrics = eval_step_fn(state, *batch)
        # These metrics are summed over each sharded batch, and averaged
        # over the num_devices. We need to multiply by num_devices to sum
        # over the entire dataset correctly, and then aggregate.
        metrics = unreplicate(metrics)
        batch_metrics.append(metrics)
        # Further divide by sharded batch size, to get average metrics
        wandb_run.log(
            batchwise_metrics_dict(metrics, B, f'{log_prefix}/batchwise'))

    epoch_metrics = aggregated_metrics_dict(
        batch_metrics, num_points, log_prefix, 
        n_devices=n_devices)
    wandb_run.log(epoch_metrics)
    
    if log_global_metrics:
        return epoch_metrics, global_metrics
    else:
        return epoch_metrics
    

def get_lr_and_schedule(
        optim_name: str,
        optim_config: ml_collections.ConfigDict,
        lr_schedule_name: Optional[str],
        lr_schedule_config: Optional[ml_collections.ConfigDict],
        steps_per_epoch: Optional[int] = None,):
    """Returns an optimizer with (optional lr_schedule)."""
    if lr_schedule_name is not None and lr_schedule_config is not None:
        schedule = getattr(optax, lr_schedule_name)
        if lr_schedule_name == "piecewise_constant_schedule":
            # Check required configs are present
            required_configs = ["scales_per_epoch"]
            if not all(name in lr_schedule_config for name in required_configs):
                print(lr_schedule_config)
                raise ValueError(
                    f"{lr_schedule_name} requires {required_configs}")
            
            # Convert scales_per_epoch from str to int, scale by train_loader
            lr_boundaries_and_scales = {}
            scales_per_epoch = lr_schedule_config.get('scales_per_epoch', None)
            for k, v in scales_per_epoch.items():
                boundary = int(k) * steps_per_epoch
                lr_boundaries_and_scales[boundary] = v
            
            lr_schedule_config.boundaries_and_scales = {
                str(k): v for k, v in lr_boundaries_and_scales.items()}
            
            # Define LR Schedule
            lr = schedule(
                init_value=optim_config.lr,
                boundaries_and_scales=lr_boundaries_and_scales,)
        elif lr_schedule_name == "exponential_decay":
            # Check required configs are present
            required_configs = ["decay_rate", "transition_steps"]
            if not all(name in lr_schedule_config for name in required_configs):
                raise ValueError(
                    f"{lr_schedule_name} requires {required_configs}")

            # Define LR Schedule
            lr = schedule(
                init_value=optim_config.lr,
                decay_rate=lr_schedule_config.decay_rate,
                transition_steps=lr_schedule_config.transition_steps,)
    else:
        lr = optim_config.lr

    optimizer = getattr(optax, optim_name)
    optimizer = optax.inject_hyperparams(optimizer)

    if optim_config.get("weight_decay", None) is not None:
        optimizer = optimizer(learning_rate=lr,
                              weight_decay=optim_config.weight_decay)
    else:
        optimizer = optimizer(learning_rate=lr)

    return optimizer
