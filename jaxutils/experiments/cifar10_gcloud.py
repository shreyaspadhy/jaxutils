"""Training a LeNetSmall model on CIFAR10."""

import ml_collections
from jaxutils.data.image import METADATA


def get_config():
    """Config for training LeNetSmall on MNIST."""
    config = ml_collections.ConfigDict()

    config.use_tpu = True

    config.global_seed = 0
    config.model_seed = 0
    config.datasplit_seed = 0
    config.use_split_global_seed = False

    # Dataset Configs
    config.dataset = ml_collections.ConfigDict()
    config.dataset.dataset_name = "CIFAR10"
    config.dataset.data_dir = "/home/shreyaspadhy_gmail_com/raw_data"
    config.dataset.flatten_img = False
    config.dataset.val_percent = 0.1
    config.dataset.perform_augmentations = True
    config.dataset.num_workers = 16

    # Add METADATA information from jaxutils
    for key in METADATA:
        config.dataset[key] = METADATA[key][config.dataset.dataset_name]

    # Training Configs
    config.batch_size = 400  # 27000
    config.eval_batch_size = 2000  # 2000
    config.n_epochs = 90
    config.perform_eval = True
    config.eval_interval = 5

    # Model Configs
    config.model = ml_collections.ConfigDict()
    config.model.num_classes = config.dataset.num_classes
    config.model.stage_sizes = [2, 2, 2, 2]
    config.model.num_filters = 64

    config.checkpoint_dir = "/home/shreyaspadhy_gmail_com/flax_models/CIFAR10"
    config.load_from_checkpoint = False
    config.save_interval = 20

    # Optimizer Configs
    config.lr_schedule_name = "piecewise_constant_schedule"
    config.gamma = 0.1

    config.lr_schedule = ml_collections.ConfigDict()
    config.lr_schedule.scales_per_epoch = {"40": config.gamma, "70": config.gamma}
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
    config.wandb.code_dir = "/home/shreyaspadhy_gmail_com/linearised-NNs"

    return config
