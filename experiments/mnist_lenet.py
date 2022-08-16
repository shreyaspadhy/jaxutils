"""Training a LeNetSmall model on MNIST."""

import ml_collections
from jaxutils.data.image import METADATA


def get_config():
    """Config for training LeNetSmall on MNIST."""
    config = ml_collections.ConfigDict()

    config.global_seed = 0
    config.model_seed = 0
    config.datasplit_seed = 0
    config.use_split_global_seed = False

    # Dataset Configs
    config.dataset = ml_collections.ConfigDict()
    config.dataset.dataset_name = "MNIST"
    config.dataset.data_dir = "/scratch3/gosset/sp2058/repos/raw_data"
    config.dataset.flatten_img = False
    config.dataset.val_percent = 0.1
    config.dataset.perform_augmentations = True
    config.dataset.num_workers = 4

    # Add METADATA information from jaxutils
    for key in METADATA:
        config.dataset[key] = METADATA[key][config.dataset.dataset_name]

    # Training Configs
    config.batch_size = 256
    config.eval_batch_size = 1024
    config.n_epochs = 90
    config.perform_eval = True
    config.eval_interval = 5

    # Model Configs
    config.model = ml_collections.ConfigDict()
    config.model.n_out = config.dataset.num_classes

    config.checkpoint_dir = "/scratch3/gosset/sp2058/flax_models/MNIST"
    config.load_from_checkpoint = False
    config.save_interval = 5

    # Optimizer Configs
    config.lr_schedule_name = "piecewise_constant_schedule"
    config.gamma = 0.1
    config.scales_per_epoch = {"40": config.gamma, "70": config.gamma}

    config.lr_schedule = ml_collections.ConfigDict()
    config.lr_schedule.boundaries_and_scales = {}

    config.optim_name = "adamw"
    config.optim = ml_collections.ConfigDict()
    config.optim.lr = 3e-3
    config.optim.weight_decay = 1e-4

    # Wandb Configs
    config.wandb = ml_collections.ConfigDict()
    config.wandb.log = False
    config.wandb.load_model = False
    config.wandb.project = "sampled-laplace"
    config.wandb.entity = "cbl-mlg"
    config.wandb.artifact_name = "lenetsmall_mnist"
    config.wandb.params_log_interval = 10

    config.linear = ml_collections.ConfigDict()
    config.linear.load_nn_checkpoint = True
    return config
