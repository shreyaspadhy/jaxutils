from typing import Any, Callable, Iterable, Optional, Union

import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax import jax_utils
from flax.core.frozen_dict import FrozenDict
from flax.jax_utils import unreplicate
from flax.training import train_state
from flax.traverse_util import ModelParamTraversal
from jax.tree_util import tree_map
from jaxutils.data.utils import get_agnostic_batch
from jaxutils.utils import tree_concatenate

import wandb

PyTree = Any


class TrainState(train_state.TrainState):
    """A Flax train state that also manages batch norm statistics."""

    model_state: FrozenDict

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
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


def aggregated_metrics_dict(list_of_metrics_dicts, len_dataset, prefix, n_devices=1):
    """Sum over list of batch_metrics dicts, and divide by dataset size."""
    metrics_dict = tree_map(
        lambda x: jnp.sum(x) * n_devices / len_dataset,
        tree_concatenate(list_of_metrics_dicts),
    )
    return {f"{prefix}/{k}": v for k, v in metrics_dict.items()}


def batchwise_metrics_dict(metrics_dict, batch_size, prefix):
    """Divide metrics_dict by batch_size."""
    new_metrics_dict = {
        f"{prefix}/{k}": v / batch_size for k, v in metrics_dict.items()
    }

    return new_metrics_dict

def add_prefix_to_dict_keys(d, prefix):
    """Add prefix to all keys in a dictionary."""
    return {f"{prefix}/{k}": v for k, v in d.items()}

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
    epoch: int = 0,
    num_epochs: int = 1,
    em_step: Optional[int] = None,
):
    assert dataset_type in ["tf", "pytorch"]
    
    log_prefix = log_prefix + f"/em_{em_step}" if em_step is not None else log_prefix
    step_log_prefix = log_prefix.split("/")[0]

    em_epoch = em_step * num_epochs + epoch if em_step is not None else epoch
    batch_metrics = []
    for i in range(steps_per_epoch):
        batch = get_agnostic_batch(next(data_iterator), dataset_type)
        n_devices, B = batch[0].shape[:2]

        
        if log_global_metrics:
            state, metrics, global_metrics = train_step_fn(state, *batch)
        else:
            state, metrics = train_step_fn(state, *batch)
        
        ######################## EVERYTHING BELOW IS FOR W&B LOGGING ##############
        train_step = epoch * steps_per_epoch + i
        em_train_step = em_epoch * steps_per_epoch + i

        if log_global_metrics:
            global_metrics = unreplicate(global_metrics)
            if em_step is not None:
                global_metrics = add_prefix_to_dict_keys(global_metrics, f"em_{em_step}")
            wandb_run.log({**global_metrics,
                           **{f'{step_log_prefix}/train_step': train_step, 
                              f'{step_log_prefix}/em_train_step': em_train_step}})
        else:
            state, metrics = train_step_fn(state, *batch)
        # These metrics are summed over each sharded batch, and averaged
        # over the num_devices. We need to multiply by num_devices to sum
        # over the entire dataset correctly, and then aggregate.
        metrics = unreplicate(metrics)
        batch_metrics.append(metrics)
        
        # Further divide by sharded batch size, to get average metrics
        metrics = {**batchwise_metrics_dict(metrics, B, f"{log_prefix}/batchwise"),
                   **{f'{step_log_prefix}/train_step': train_step, 
                      f'{step_log_prefix}/em_train_step': em_train_step}}
        wandb_run.log(metrics)

    epoch_metrics = aggregated_metrics_dict(
        batch_metrics, num_points, log_prefix, n_devices=n_devices
    )
    wandb_run.log({**epoch_metrics,
                   **{f'{step_log_prefix}/train_epoch': epoch, 
                      f'{step_log_prefix}/em_train_epoch': em_epoch}})

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
        batch = get_agnostic_batch(next(data_iterator), dataset_type)
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
        wandb_run.log(batchwise_metrics_dict(metrics, B, f"{log_prefix}/batchwise"))

    epoch_metrics = aggregated_metrics_dict(
        batch_metrics, num_points, log_prefix, n_devices=n_devices
    )
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
    steps_per_epoch: Optional[int] = None,
    model_mask: Optional[PyTree] = None,
):
    """Returns an optimizer with (optional lr_schedule)."""
    if lr_schedule_name is not None and lr_schedule_config is not None:
        schedule = getattr(optax, lr_schedule_name)
        if lr_schedule_name == "piecewise_constant_schedule":
            # Check required configs are present
            required_configs = ["scales_per_epoch"]
            if not all(name in lr_schedule_config for name in required_configs):
                print(lr_schedule_config)
                raise ValueError(f"{lr_schedule_name} requires {required_configs}")

            # Convert scales_per_epoch from str to int, scale by train_loader
            lr_boundaries_and_scales = {}
            scales_per_epoch = lr_schedule_config.get("scales_per_epoch", None)
            for k, v in scales_per_epoch.items():
                boundary = int(k) * steps_per_epoch
                lr_boundaries_and_scales[boundary] = v

            lr_schedule_config.boundaries_and_scales = {
                str(k): v for k, v in lr_boundaries_and_scales.items()
            }

            # Define LR Schedule
            lr = schedule(
                init_value=optim_config.lr,
                boundaries_and_scales=lr_boundaries_and_scales,
            )
        elif lr_schedule_name == "exponential_decay":
            # Check required configs are present
            required_configs = ["decay_rate", "transition_steps"]
            if not all(name in lr_schedule_config for name in required_configs):
                raise ValueError(f"{lr_schedule_name} requires {required_configs}")

            # Define LR Schedule
            lr = schedule(
                init_value=optim_config.lr,
                decay_rate=lr_schedule_config.decay_rate,
                transition_steps=lr_schedule_config.transition_steps,
            )

        elif lr_schedule_name == "warmup_exponential_decay_schedule":
            # Check required configs are present
            required_configs = ["init_value", "warmup_steps", "transition_steps",
                                "decay_rate", "transition_begin",]
            if not all(name in lr_schedule_config for name in required_configs):
                raise ValueError(f"{lr_schedule_name} requires {required_configs}")
            
            # Define RL Schedule
            lr = schedule(
                init_value=lr_schedule_config.init_value,
                peak_value=optim_config.lr,
                warmup_steps=lr_schedule_config.warmup_steps,
                transition_steps=lr_schedule_config.transition_steps,
                decay_rate=lr_schedule_config.decay_rate,
                transition_begin=lr_schedule_config.transition_begin,)

        elif lr_schedule_name == "linear_schedule":
            # Check required configs are present
            required_configs = ["end_value", "transition_steps"]
            if not all(name in lr_schedule_config for name in required_configs):
                raise ValueError(f"{lr_schedule_name} requires {required_configs}")

            # Define LR Schedule
            lr = schedule(
                init_value=optim_config.lr,
                end_value=lr_schedule_config.end_value,
                transition_steps=lr_schedule_config.transition_steps,
            )
        else:
            raise ValueError("Scheduler not supported")
    
    else:
        lr = optim_config.lr

    optimizer = getattr(optax, optim_name)
    optimizer = optax.inject_hyperparams(optimizer)

    use_nesterov = optim_config.get("nesterov", False)
    weight_decay = optim_config.get("weight_decay", None)

    absolute_clipping = optim_config.get("absolute_clipping", None)

    if optim_name == "sgd":
        optimizer = optimizer(
            learning_rate=lr, momentum=optim_config.momentum, nesterov=use_nesterov
        )
        if weight_decay is not None:
            optimizer = optax.chain(
                optimizer, optax.additive_weight_decay(weight_decay, model_mask)
            )

    if optim_name == "adamw":
        # If adamw, weight_decay is a passable parameter.
        if weight_decay is None:
            raise ValueError("weight_decay must be specified for adamw")
        optimizer = optimizer(learning_rate=lr, weight_decay=weight_decay)

    if optim_name == "adam":
        optimizer = optimizer(learning_rate=lr)

    if absolute_clipping is not None:
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(absolute_clipping),
            optimizer
        )


    # if optim_config.get("weight_decay", None) is not None:
    #     if optim_name == "sgd":
    #         optimizer = optax.chain(
    #             optimizer(learning_rate=lr)
    #             optax.additive_weight_decay(optim_config.weight_decay, model_mask))
    #     elif optim_name == "adamw":

    #     return optimizer

    # if optim_config.get("nesterov", False)  is True:
    #     optimizer = optimizer(learning_rate=lr, momentum=optim_config.momentum,
    #                           nesterov=True)
    # else:
    #     optimizer = optimizer(learning_rate=lr)

    return optimizer


def get_lr_from_opt_state(opt_state):
    """Returns the learning rate from the opt_state."""
    if isinstance(opt_state, optax.InjectHyperparamsState):
        lr = opt_state.hyperparams["learning_rate"]
    elif isinstance(opt_state, tuple):
        for o_state in opt_state:
            if isinstance(o_state, optax.InjectHyperparamsState):
                lr = o_state.hyperparams["learning_rate"]
            if isinstance(o_state, tuple):
                for o_state2 in o_state:
                    if isinstance(o_state2, optax.InjectHyperparamsState):
                        lr = o_state2.hyperparams["learning_rate"]
    
    return lr

def get_model_masks(params, param_wd_dict: Union[dict, float]):
    """Create boolean masks on Pytrees for model parameters.
    Args:
        params (Pytree): Model parameters.
        param_wd_dict: Dictionary containing param names for which unique
            masks are required to be created.
    Returns:
        dict: Dictionary containing Pytree masks for each param name.
    """
    # If weight_decay params is a float, return all params to perform wd over.
    if isinstance(param_wd_dict, float):
        all_true = jax.tree_map(lambda _: True, params)
        print('resturn all')
        return all_true
    # Otherwise, iterate through dict, and return dict of model masks.
    all_false = jax.tree_map(lambda _: False, params)
    param_masks = {}
    for name in param_wd_dict.keys():
        subparam_update = ModelParamTraversal(lambda p, _: name in p)
        param_masks[name] = subparam_update.update(lambda _: True, all_false)

    return param_masks
