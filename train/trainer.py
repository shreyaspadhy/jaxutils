import logging
from pathlib import Path
from typing import Optional, Union

import jax
import jaxutils
import jax.numpy as jnp
import jax.random as random
import ml_collections.config_flags
import torch
from absl import app, flags
from flax import linen as nn
from flax.jax_utils import unreplicate
from flax.training import checkpoints
from jaxutils.data.pt_image import get_image_dataset as get_pt_image_dataset
from jaxutils.data.pt_preprocess import NumpyLoader
from jaxutils.data.tf_image import PYTORCH_TO_TF_NAMES, get_image_dataloader
from jaxutils.data.tf_image import get_image_dataset as get_tf_image_dataset
from jaxutils.data.utils import get_agnostic_iterator
import jaxutils.models as models
from jaxutils.train.classification import create_eval_step, create_train_step
import optax
from jaxutils.train.utils import (
    TrainState,
    eval_epoch,
    get_lr_and_schedule,
    get_model_masks,
    train_epoch,
)
from jaxutils.utils import (
    flatten_nested_dict,
    generate_keys,
    log_model_params,
    setup_training,
    update_config_dict,
)
from tqdm import trange

import wandb

ml_collections.config_flags.DEFINE_config_file(
    "config",
    "./experiments/mnist_lenet.py",
    "Training configuration.",
    lock_config=True,
)

FLAGS = flags.FLAGS
logger = logging.getLogger(__name__)


def main(config):
    wandb_kwargs = {
        "project": config.wandb.project,
        "entity": config.wandb.entity,
        "config": flatten_nested_dict(config.to_dict()),
        "mode": "online" if config.wandb.log else "disabled",
        "settings": wandb.Settings(code_dir=config.wandb.code_dir),
    }
    with wandb.init(**wandb_kwargs) as run:
        ####################### Refresh Config Dicts #########################
        # Update config file with run.config to update hparam_sweep values
        config.unlock()
        config.update_from_flattened_dict(run.config)
        # Add hparams that need to be computed from sweeped hparams to configs
        computed_configs = {}
        update_config_dict(config, run, computed_configs)
        config.lock()

        # Setup training flags and log to Wandb
        setup_training(run)

        ######################## Set up random seeds #########################
        if config.use_split_global_seed:
            seed = config.get("global_seed", 0)
            model_rng, datasplit_rng = generate_keys(seed)
            torch.manual_seed(seed)
        else:
            seed = config.get("global_seed", 0)
            torch.manual_seed(config.get("global_seed", 0))
            global_rng = random.PRNGKey(seed)
            model_rng = random.PRNGKey(config.model_seed)
            datasplit_rng = config.datasplit_seed

        ################### Load dataset/dataloders ##########################
        # NOTE: On a single TPU v3 pod, we have 1 process, 8 devices
        batch_size = config.dataset.process_batch_size * jax.process_count()
        eval_batch_size = config.dataset.eval_process_batch_size * jax.process_count()

        if (
            batch_size % jax.device_count() != 0
            or eval_batch_size % jax.device_count() != 0
        ):
            raise ValueError(
                f"Batch sizes ({batch_size} and {eval_batch_size}) must "
                f"be divisible by device number ({jax.device_count()})"
            )

        if config.dataset_type == "pytorch":
            train_dataset, test_dataset, val_dataset = get_pt_image_dataset(
                dataset_name=config.dataset.dataset_name,
                data_dir=config.dataset.data_dir,
                flatten_img=config.dataset.flatten_img,
                val_percent=config.dataset.val_percent,
                random_seed=datasplit_rng,
                perform_augmentations=config.dataset.perform_augmentations,
            )
            # Create Dataloaders
            train_loader = NumpyLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=config.dataset.shuffle_train_split,
                num_workers=config.dataset.num_workers,
                drop_last=config.dataset.drop_remainder,
            )
            test_loader = NumpyLoader(
                test_dataset,
                batch_size=eval_batch_size,
                shuffle=config.dataset.shuffle_eval_split,
                num_workers=config.dataset.num_workers,
                drop_last=config.dataset.drop_remainder,
            )
            if val_dataset is not None:
                val_loader = NumpyLoader(
                    val_dataset,
                    batch_size=eval_batch_size,
                    shuffle=config.dataset.shuffle_eval_split,
                    num_workers=config.dataset.num_workers,
                    drop_last=config.dataset.drop_remainder,
                )
            else:
                val_loader = None

            train_config = {
                "n_train": len(train_dataset),
                "n_val": len(val_dataset) if val_dataset is not None else None,
                "n_test": len(test_dataset),
                "dataset.batch_size": batch_size,
                "dataset.eval_batch_size": eval_batch_size,
                "train_steps_per_epoch": len(train_loader),
                "val_steps_per_epoch": len(val_loader)
                if val_loader is not None
                else None,
                "test_steps_per_epoch": len(test_loader),
            }

            # Add all training dataset hparams back to config_dicts
            update_config_dict(config, run, train_config)
        elif config.dataset_type == "tf":
            datasets, num_examples = get_tf_image_dataset(
                dataset_name=PYTORCH_TO_TF_NAMES[config.dataset.dataset_name],
                process_batch_size=config.dataset.process_batch_size,
                eval_process_batch_size=config.dataset.eval_process_batch_size,
                shuffle_train_split=config.dataset.shuffle_train_split,
                shuffle_eval_split=config.dataset.shuffle_eval_split,
                drop_remainder=config.dataset.drop_remainder,
                data_dir=config.dataset.data_dir,
                try_gcs=config.dataset.try_gcs,
                val_percent=config.dataset.val_percent,
                datasplit_rng=random.PRNGKey(datasplit_rng),
            )

            train_dataset, val_dataset, test_dataset = datasets
            shuffle_rng, global_rng = random.split(global_rng, 2)
            shuffle_rngs = random.split(shuffle_rng, 3)

            # Create Dataloaders
            train_loader = get_image_dataloader(
                train_dataset,
                dataset_name=config.dataset.dataset_name,
                process_batch_size=config.dataset.process_batch_size,
                num_epochs=config.n_epochs,
                shuffle=config.dataset.shuffle_train_split,
                shuffle_buffer_size=config.dataset.shuffle_buffer_size,
                rng=shuffle_rngs[0],
                cache=config.dataset.cache,
                repeat_after_batching=config.dataset.repeat_after_batching,
                drop_remainder=config.dataset.drop_remainder,
                prefetch_size=config.dataset.prefetch_size,
                prefetch_on_device=config.dataset.prefetch_on_device,
                perform_augmentations=config.dataset.perform_augmentations,
            )
            test_loader = get_image_dataloader(
                test_dataset,
                dataset_name=config.dataset.dataset_name,
                process_batch_size=config.dataset.eval_process_batch_size,
                num_epochs=config.n_epochs,
                shuffle=config.dataset.shuffle_eval_split,
                shuffle_buffer_size=config.dataset.shuffle_buffer_size,
                rng=shuffle_rngs[1],
                cache=config.dataset.cache,
                repeat_after_batching=config.dataset.repeat_after_batching,
                drop_remainder=config.dataset.drop_remainder,
                prefetch_size=config.dataset.prefetch_size,
                prefetch_on_device=config.dataset.prefetch_on_device,
                perform_augmentations=False,
            )
            if val_dataset is not None:
                val_loader = get_image_dataloader(
                    val_dataset,
                    dataset_name=config.dataset.dataset_name,
                    process_batch_size=config.dataset.eval_process_batch_size,
                    num_epochs=config.n_epochs,
                    shuffle=config.dataset.shuffle_eval_split,
                    shuffle_buffer_size=config.dataset.shuffle_buffer_size,
                    rng=shuffle_rngs[2],
                    cache=config.dataset.cache,
                    repeat_after_batching=config.dataset.repeat_after_batching,
                    drop_remainder=config.dataset.drop_remainder,
                    prefetch_size=config.dataset.prefetch_size,
                    prefetch_on_device=config.dataset.prefetch_on_device,
                    perform_augmentations=False,
                )
            else:
                val_loader = None

            train_config = {
                "n_train": num_examples[0],
                "n_val": num_examples[1],
                "n_test": num_examples[2],
                "dataset.batch_size": batch_size,
                "dataset.eval_batch_size": eval_batch_size,
                "train_steps_per_epoch": num_examples[0] // batch_size,
                "val_steps_per_epoch": num_examples[1] // eval_batch_size
                if num_examples[1] is not None
                else None,
                "test_steps_per_epoch": num_examples[2] // eval_batch_size,
            }

            # Add all training dataset hparams back to config_dicts
            update_config_dict(config, run, train_config)

        # Create and initialise model
        model_cls = getattr(models, config.model_name)
        model = model_cls(**config.model.to_dict())
        # model = LeNetSmall(**config.model.to_dict())
        # model = ResNet(block_cls=ResNetBlock, **config.model.to_dict())

        dummy_init = jnp.expand_dims(jnp.ones(config.dataset.image_shape), 0)
        variables = model.init(model_rng, dummy_init)

        model_state, params = variables.pop("params")
        print('length params : ', len(params.keys()), len(model_state.keys()))
        # jaxutils.utils.print_param_shapes(params)
        del variables


        # Load from checkpoint, and append specific model seed
        config.unlock()
        config.checkpoint_dir = config.checkpoint_dir + f"/{seed}"
        config.lock()

        checkpoint_dir = Path(config.checkpoint_dir).resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if config.load_from_checkpoint:
            print(f"checkpoint_dir: {checkpoint_dir}")
            checkpoint_path = checkpoints.latest_checkpoint(checkpoint_dir)
            print('test')
            restored_state = checkpoints.restore_checkpoint(checkpoint_path, 
                                                            target=None)
            params = restored_state["params"]
            model_state = restored_state["model_state"]
        
        # Get Optimizer and Learning Rate Schedule, if any
        if config.optim.get("weight_decay", None) is not None:
            model_mask = get_model_masks(params, 
                                         config.optim.weight_decay)
        else:
            model_mask = None
        config.unlock()

        optimizer = get_lr_and_schedule(
            config.optim_name,
            config.optim,
            config.get("lr_schedule_name", None),
            config.get("lr_schedule", None),
            steps_per_epoch=config.train_steps_per_epoch,
            model_mask=model_mask,
        )
        config.lock()
        
        state = TrainState.create(
            apply_fn=model.apply,
            model_state=model_state,
            params=params,
            tx=optimizer,
            )

        # Define model artifacts to save models using wandb.
        model_artifact = wandb.Artifact(
            "model", type="model", description="trained model"
        )
        best_model_artifact = wandb.Artifact(
            "best_model", type="best model", description="best trained model"
        )

        state = train_model(
            model,
            state,
            train_loader,
            val_loader,
            test_loader,
            checkpoint_dir,
            run,
            model_artifact,
            best_model_artifact,
            config,
        )


def train_model(
    model: nn.Module,
    state: TrainState,
    train_loader: NumpyLoader,
    val_loader: Optional[NumpyLoader],
    test_loader: NumpyLoader,
    checkpoint_dir: Union[str, Path],
    run: wandb.run,
    wandb_artifact: wandb.Artifact,
    best_wandb_artifact: wandb.Artifact,
    config: ml_collections.ConfigDict,
) -> TrainState:

    # Create training and evaluation functions
    train_step = create_train_step(
        model, state.tx, num_classes=config.dataset.num_classes
    )
    eval_step = create_eval_step(model, num_classes=config.dataset.num_classes)

    # Create parallel version of the train and eval step
    p_train_step = jax.pmap(train_step, "device", donate_argnums=(0,))
    p_eval_step = jax.pmap(eval_step, "device")

    # Replicate the train state on each device
    state = state.replicate()

    # Perform Training
    val_losses = []
    epochs = trange(config.n_epochs)
    for epoch in epochs:
        state, train_metrics = train_epoch(
            train_step_fn=p_train_step,
            data_iterator=get_agnostic_iterator(train_loader, config.dataset_type),
            steps_per_epoch=config.train_steps_per_epoch,
            num_points=config.n_train,
            state=state,
            wandb_run=run,
            log_prefix="train",
            dataset_type=config.dataset_type,
            epoch=epoch,
        )

        epochs.set_postfix(train_metrics)

        if epoch % config.wandb.params_log_interval == 0:
            log_model_params(unreplicate(state).params, run, param_prefix="params")
            log_model_params(
                unreplicate(state).model_state, run, param_prefix="model_state"
            )

        # Optionally evaluate on val dataset
        if val_loader is not None:
            val_metrics = eval_epoch(
                eval_step_fn=p_eval_step,
                data_iterator=get_agnostic_iterator(val_loader, config.dataset_type),
                steps_per_epoch=config.val_steps_per_epoch,
                num_points=config.n_val,
                state=state,
                wandb_run=run,
                log_prefix="val",
                dataset_type=config.dataset_type,
            )

            val_losses.append(val_metrics["val/nll"])

            # Save best validation loss/epoch over training
            if val_metrics["val/nll"] <= min(val_losses):
                run.summary["best_val_loss"] = val_metrics["val/nll"]
                run.summary["best_epoch"] = epoch

                checkpoints.save_checkpoint(
                    checkpoint_dir / "best",
                    unreplicate(state),
                    epoch + 1,
                    keep=1,
                    overwrite=True,
                )

        # Now eval on test dataset every few intervals if perform_eval=True
        if (epoch + 1) % config.eval_interval == 0 and config.perform_eval:
            test_metrics = eval_epoch(
                eval_step_fn=p_eval_step,
                data_iterator=get_agnostic_iterator(test_loader, config.dataset_type),
                steps_per_epoch=config.test_steps_per_epoch,
                num_points=config.n_test,
                state=state,
                wandb_run=run,
                log_prefix="test",
                dataset_type=config.dataset_type,
            )
            epochs.set_postfix(test_metrics)

        # Find the part of opt_state that contains InjectHyperparamsState
        opt_state = unreplicate(state).opt_state
        if isinstance(opt_state, optax.InjectHyperparamsState):
            lr_to_log = opt_state.hyperparams["learning_rate"]
        elif isinstance(opt_state, tuple):
            for o_state in opt_state:
                # print('test', type(o_state))
                if isinstance(o_state, optax.InjectHyperparamsState):
                    lr_to_log = o_state.hyperparams["learning_rate"]
                if isinstance(o_state, tuple):
                    for o_state2 in o_state:
                        if isinstance(o_state2, optax.InjectHyperparamsState):
                            lr_to_log = o_state2.hyperparams["learning_rate"]

        run.log({"lr": lr_to_log})

        # Save Model Checkpoints
        if (epoch + 1) % config.save_interval == 0:
            checkpoints.save_checkpoint(
                checkpoint_dir, unreplicate(state), epoch + 1, keep=5, overwrite=True
            )

    # Save all model checkpoints to wandb
    wandb_artifact.add_dir(checkpoint_dir)
    wandb.run.log_artifact(wandb_artifact)

    if val_loader is not None:
        best_wandb_artifact.add_dir(checkpoint_dir / "best")
        wandb.run.log_artifact(best_wandb_artifact, aliases="best")

    return state


if __name__ == "__main__":
    # Adds jax flags to the program.
    jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)
