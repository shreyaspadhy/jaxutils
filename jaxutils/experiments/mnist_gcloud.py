"""Training a LeNetSmall model on MNIST."""

import ml_collections
from jaxutils.data.pt_image import METADATA


def get_config():
    """Config for training LeNetSmall on MNIST."""
    config = ml_collections.ConfigDict()

    config.use_tpu = True

    config.global_seed = 0
    config.model_seed = 0
    config.datasplit_seed = 0
    config.use_split_global_seed = False

    # Dataset Configs
    config.dataset_type = "tf"

    config.dataset = ml_collections.ConfigDict()
    config.dataset.dataset_name = "MNIST"

    config.dataset.try_gcs = True
    if config.dataset_type == "tf" and config.dataset.try_gcs:
        config.dataset.data_dir = None
    else:
        config.dataset.data_dir = "/home/shreyaspadhy_gmail_com/raw_data"

    config.dataset.flatten_img = False
    config.dataset.val_percent = 0.0
    config.dataset.perform_augmentations = True
    config.dataset.num_workers = 16

    config.dataset.process_batch_size = 200
    config.dataset.batch_size = config.dataset.process_batch_size

    config.dataset.eval_process_batch_size = 2000  # 10000/125
    config.dataset.eval_batch_size = config.dataset.eval_process_batch_size

    config.dataset.cache = False
    config.dataset.repeat_after_batching = False
    config.dataset.shuffle_train_split = True
    config.dataset.shuffle_eval_split = False
    config.dataset.shuffle_buffer_size = 10_000
    config.dataset.prefetch_size = 4
    config.dataset.prefetch_on_device = None
    config.dataset.drop_remainder = True

    # Add METADATA information from jaxutils
    for key in METADATA:
        config.dataset[key] = METADATA[key][config.dataset.dataset_name]

    config.n_epochs = 90
    config.perform_eval = True
    config.eval_interval = 5

    # Model Configs
    config.model_name = "LeNetSmall"
    config.model = ml_collections.ConfigDict()
    config.model.n_out = config.dataset.num_classes

    config.checkpoint_dir = "/home/shreyaspadhy_gmail_com/flax_models/MNIST"
    config.load_from_checkpoint = False
    config.save_interval = 20

    # Optimizer Configs
    config.lr_schedule_name = "piecewise_constant_schedule"
    config.gamma = 0.1

    config.lr_schedule = ml_collections.ConfigDict()
    config.lr_schedule.scales_per_epoch = {"40": config.gamma, "70": config.gamma}
    config.lr_schedule.boundaries_and_scales = {}

    config.optim_name = "sgd"

    config.optim = ml_collections.ConfigDict()
    if config.optim_name == "sgd":
        config.optim.lr = 1e-2
        config.optim.momentum = 0.9
        config.optim.nesterov = False
    elif config.optim_name == "adamw":
        config.optim.lr = 3e-3

    config.optim.weight_decay = 1e-4

    # Wandb Configs
    config.wandb = ml_collections.ConfigDict()
    config.wandb.log = True
    config.wandb.load_model = False
    config.wandb.project = "pmf-training"
    config.wandb.entity = "shreyaspadhy"
    config.wandb.artifact_name = "lenetsmall_mnist"
    config.wandb.params_log_interval = 10
    config.wandb.code_dir = "/home/shreyaspadhy_gmail_com/collaborative-filtering-jax"

    return config
