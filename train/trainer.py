from pathlib import Path
from typing import Optional, Union
import logging

import jax
import jax.numpy as jnp
import jax.random as random
import ml_collections.config_flags
import optax
from absl import app, flags
from clu import parameter_overview
from flax import linen as nn
from flax.training import checkpoints
from flax.training.common_utils import shard
from flax.jax_utils import unreplicate
from tqdm import trange

import torch
import wandb
from jaxutils.data.image import get_image_dataset as get_pt_image_dataset
from jaxutils.data.tf_datasets import get_image_dataset as get_tf_image_dataset
from jaxutils.data.tf_datasets import get_image_dataloader, get_num_examples, PYTORCH_TO_TF_NAMES
from jaxutils.data.utils import NumpyLoader
from jaxutils.models.lenets import LeNetSmall
from jaxutils.models.resnets import ResNetBlock, ResNet
from jaxutils.train.classification import create_eval_step, create_train_step
from jaxutils.train.utils import (
    aggregated_metrics_dict,
    batchwise_metrics_dict,
    get_dataset_iterator,
    get_lr_and_schedule,
    TrainState,
    eval_epoch,
    train_epoch,
)
from jaxutils.utils import (
    flatten_nested_dict,
    generate_keys,
    log_model_params,
    setup_training,
    update_config_dict
)

ml_collections.config_flags.DEFINE_config_file(
    "config", "./experiments/mnist_lenet.py",
    "Training configuration.", lock_config=True
)

FLAGS = flags.FLAGS
logger = logging.getLogger(__name__)


def main(config):
    wandb_kwargs = {
        'project': config.wandb.project,
        'entity': config.wandb.entity,
        'config': flatten_nested_dict(config.to_dict()),
        'mode': "online" if config.wandb.log else "disabled",
        'settings': wandb.Settings(code_dir=config.wandb.code_dir),
    }
    with wandb.init(**wandb_kwargs) as run:
        ####################### Refresh Config Dicts #########################
        # Update config file with run.config to update hparam_sweep values
        config.unlock()
        config.update_from_flattened_dict(run.config)
        # Add hparams that need to be computed from sweeped hparams to configs
        # computed_configs = {
        #     'model.n_out': config.dataset.num_classes,
        # }
        # config.update_from_flattened_dict(computed_configs)
        # run.config.update(computed_configs, allow_val_change=True)
        config.lock()

        # Setup training flags and log to Wandb
        setup_training(run)
        
        ####################### Setup logging ###############################
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        # Setup logging, we only want one process per machine to log things.
        logger.setLevel(
            logging.INFO if jax.process_index() == 0 else logging.ERROR)

        ######################## Set up random seeds #########################
        if config.use_split_global_seed:
            seed = config.get("global_seed", 0)
            model_rng, datasplit_rng = generate_keys(seed)
            torch.manual_seed(seed)
        else:
            seed = config.get("global_seed", 0)
            rng = random.PRNGKey(seed)
            model_rng = random.PRNGKey(config.model_seed)
            datasplit_rng = config.datasplit_seed
            torch.manual_seed(config.get("global_seed", 0))

        ################### Load dataset/dataloders ##########################
        batch_size = config.dataset.process_batch_size * jax.process_count()
        eval_batch_size = config.dataset.eval_process_batch_size * jax.process_count()
            
        if config.dataset_type == "pytorch":
            train_dataset, test_dataset, val_dataset = get_pt_image_dataset(
                dataset_name=config.dataset.dataset_name,
                data_dir=config.dataset.data_dir,
                flatten_img=config.dataset.flatten_img,
                val_percent=config.dataset.val_percent,
                random_seed=datasplit_rng,
                perform_augmentations=config.dataset.perform_augmentations)
            # Create Dataloaders
            train_loader = NumpyLoader(
                train_dataset, batch_size=batch_size,
                shuffle=True, num_workers=config.dataset.num_workers,
                drop_last=config.use_tpu,)
            test_loader = NumpyLoader(
            test_dataset, batch_size=eval_batch_size,
            shuffle=False, num_workers=config.dataset.num_workers,
            drop_last=config.use_tpu,)
            if val_dataset is not None:
                val_loader = NumpyLoader(
                    val_dataset, batch_size=eval_batch_size,
                    shuffle=False, num_workers=config.dataset.num_workers,
                    drop_last=config.use_tpu,)
            else:
                val_loader = None
            
            train_config = {
                'n_train': len(train_dataset),
                'n_val': len(val_dataset) if val_dataset is not None else None,
                'n_test': len(test_dataset),
                'dataset.batch_size': batch_size,
                'dataset.eval_batch_size': eval_batch_size,
                'train_steps_per_epoch': len(train_loader),
                'val_steps_per_epoch': len(val_loader) if val_loader is not None else None,
                'test_steps_per_epoch': len(test_loader),}

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
                perform_augmentations=config.dataset.perform_augmentations,
                rng=random.PRNGKey(datasplit_rng))
            
            train_dataset, val_dataset, test_dataset = datasets
            # Create Dataloaders
            train_loader = get_image_dataloader(
                train_dataset, 
                process_batch_size=config.dataset.process_batch_size, 
                num_epochs=config.n_epochs,
                shuffle=config.dataset.shuffle_train_split,
                shuffle_buffer_size=config.dataset.shuffle_buffer_size,
                shuffle_rng=rng,
                cache=config.dataset.cache,
                repeat_after_batching=config.dataset.repeat_after_batching,
                drop_remainder=config.dataset.drop_remainder,
                prefetch_size=config.dataset.prefetch_size,
                prefetch_on_device=config.dataset.prefetch_on_device,)
            test_loader = get_image_dataloader(
                test_dataset, 
                process_batch_size=config.dataset.eval_process_batch_size, 
                num_epochs=config.n_epochs,
                shuffle=config.dataset.shuffle_eval_split,
                shuffle_buffer_size=config.dataset.shuffle_buffer_size,
                cache=config.dataset.cache,
                repeat_after_batching=config.dataset.repeat_after_batching,
                drop_remainder=config.dataset.drop_remainder,
                prefetch_size=config.dataset.prefetch_size,
                prefetch_on_device=config.dataset.prefetch_on_device,)
            if val_dataset is not None:
                val_loader = get_image_dataloader(
                    val_dataset, 
                    process_batch_size=config.dataset.eval_process_batch_size, 
                    num_epochs=config.n_epochs,
                    shuffle=config.dataset.shuffle_eval_split,
                    shuffle_buffer_size=config.dataset.shuffle_buffer_size,
                    cache=config.dataset.cache,
                    repeat_after_batching=config.dataset.repeat_after_batching,
                    drop_remainder=config.dataset.drop_remainder,
                    prefetch_size=config.dataset.prefetch_size,
                    prefetch_on_device=config.dataset.prefetch_on_device,)
            else:
                val_loader = None
                
            train_config = {
                'n_train': num_examples[0],
                'n_val': num_examples[1],
                'n_test': num_examples[2],
                'dataset.batch_size': batch_size,
                'dataset.eval_batch_size': eval_batch_size,
                'train_steps_per_epoch': num_examples[0] // batch_size,
                'val_steps_per_epoch': num_examples[1] // eval_batch_size if num_examples[1] is not None else None,
                'test_steps_per_epoch': num_examples[2] // eval_batch_size,}

            # Add all training dataset hparams back to config_dicts
            update_config_dict(config, run, train_config)
        
        # Create and initialise model
        model = LeNetSmall(**config.model.to_dict())
        # model = ResNet(block_cls=ResNetBlock, **config.model.to_dict())

        dummy_init = jnp.expand_dims(jnp.ones(config.dataset.image_shape), 0)
        variables = model.init(model_rng, dummy_init)

        model_state, params = variables.pop('params')
        del variables
        
        # Get Optimizer and Learning Rate Schedule, if any
        config.unlock()
        optimizer = get_lr_and_schedule(
            config.optim_name,
            config.optim,
            config.get('lr_schedule_name', None),
            config.get('lr_schedule', None),
            steps_per_epoch=config.train_steps_per_epoch,)
        config.lock()
        
        # Create train_state to save and load
        state = TrainState.create(apply_fn=model.apply,
                                  params=params,
                                  tx=optimizer,
                                  model_state=model_state)

        # Load from checkpoint, and append specific model seed
        config.unlock()
        config.checkpoint_dir = config.checkpoint_dir + f'/{seed}'
        config.lock()

        checkpoint_dir = Path(config.checkpoint_dir).resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if config.load_from_checkpoint:
            if config.wandb.load_model:
                model_at = wandb.run.use_artifact(
                    config.wandb.model_artifact_name + ":latest")
                checkpoint_path = model_at.download()
                restored_state = checkpoints.restore_checkpoint(checkpoint_path,
                                                                target=None)
            else:
                print(f'checkpoint_dir: {checkpoint_dir}')
                checkpoint_path = checkpoints.latest_checkpoint(checkpoint_dir)
                state = checkpoints.restore_checkpoint(checkpoint_path,
                                                    target=state)

        # Define model artifacts to save models using wandb.
        model_artifact = wandb.Artifact(
            "model", type='model', description='trained model')
        best_model_artifact = wandb.Artifact(
            "best_model", type='best model', description='best trained model')

        
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
            config,)


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
    config: ml_collections.ConfigDict,): 
    
    # Create training and evaluation functions
    train_step = create_train_step(model, state.tx,
                                   num_classes=config.dataset.num_classes)
    eval_step = create_eval_step(model,
                                 num_classes=config.dataset.num_classes)
    
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
            data_iterator=get_dataset_iterator(train_loader, config.dataset_type), 
            steps_per_epoch=config.train_steps_per_epoch, 
            num_points=config.n_train, 
            state=state, 
            wandb_run=run, 
            log_prefix='train',
            dataset_type=config.dataset_type)
        
        epochs.set_postfix(train_metrics)

        if epoch % config.wandb.params_log_interval == 0:
            log_model_params(unreplicate(state).params, run,
                                param_prefix='params')
            log_model_params(unreplicate(state).model_state, run,
                                param_prefix='model_state')

        # Optionally evaluate on val dataset
        if val_loader is not None:
            val_metrics = eval_epoch(
                eval_step_fn=p_eval_step, 
                data_iterator=get_dataset_iterator(val_loader, config.dataset_type),
                steps_per_epoch=config.val_steps_per_epoch, 
                num_points=config.n_val, 
                state=state, 
                wandb_run=run, 
                log_prefix='val',
                dataset_type=config.dataset_type)
            
            val_losses.append(val_metrics['val/nll'])

            # Save best validation loss/epoch over training
            if val_metrics['val/nll'] <= min(val_losses):
                run.summary['best_val_loss'] = val_metrics['val/nll']
                run.summary['best_epoch'] = epoch

                checkpoints.save_checkpoint(
                    checkpoint_dir / "best", unreplicate(state), epoch+1, keep=1,
                    overwrite=True)

        # Now eval on test dataset every few intervals if perform_eval=True
        if (epoch + 1) % config.eval_interval == 0 and config.perform_eval:
            test_metrics = eval_epoch(
                eval_step_fn=p_eval_step, 
                data_iterator=get_dataset_iterator(test_loader, config.dataset_type),
                steps_per_epoch=config.test_steps_per_epoch, 
                num_points=config.n_test, 
                state=state, 
                wandb_run=run, 
                log_prefix='test',
                dataset_type=config.dataset_type)

        # Save any auxilliary variables to log
        run.log({'lr': unreplicate(state).opt_state.hyperparams['learning_rate']})

        # Save Model Checkpoints
        if (epoch + 1) % config.save_interval == 0:
            checkpoints.save_checkpoint(
                checkpoint_dir, unreplicate(state), epoch+1, keep=5, 
                overwrite=True)

    # Save all model checkpoints to wandb
    wandb_artifact.add_dir(checkpoint_dir)
    wandb.run.log_artifact(wandb_artifact)

    if val_loader is not None:
        best_wandb_artifact.add_dir(checkpoint_dir / "best")
        wandb.run.log_artifact(best_wandb_artifact, aliases="best")


if __name__ == "__main__":
    # Adds jax flags to the program.
    jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)
