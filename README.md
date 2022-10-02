# jaxutils

This repository contains common utilities in JAX/Flax to use across research projects at the Cambridge MLG Group. Currently, we implement the following functionality that might be useful - 

### Dataloading & Processing

We provide PyTorch and Tensorflow dataloading and processing functionality for many standard vision datasets. These dataloaders are included in the following files - 

- `pt_image.py` contains dataloaders with a PyTorch backend that load datasets as `jnp.ndarray` arrays for use with Jax/Flax models.
- `tf_image.py` contains dataloaders with a Tensorflow backend. There is additional functionality for parallel dataloading and multithreading that makes the Tensorflow backend faster on TPUs by 2x-5x (atleast on Google Cloud). A large amount of this functionality is heavily inspired by the excellent dataloading in [Uncertainty Baselines](https://github.com/google/uncertainty-baselines/blob/main/baselines/jft/input_utils.py).

### Standard Flax models

We provide some LeNet-CNN variants, and ResNet18 models. These models were initially designed in order to load in pre-trained PyTorch checkpoints, which is why there are some non-traditional definitions of `AvgPooling`. Additionally, we also provide - 

- `convert_utils.py` contains functionality to convert PyTorch models to Flax. Though these might be hardcoded somewhat to reflect the exact weight and layer names for our specific usecase, it should be easy to modify these functions in order to load in your own PyTorch models into Flax.

### Classification Pipeline

We also provide a helper script `trainer.py` that is meant to serve as a standalone script to train a model from scratch on an image classification dataset, with all the Jax bells and whistles (`vmap`, `pmap`, multi-node training etc.). This script contains some additional logging and syncing features with Weights and Biases, as well as config definitions using `config_dicts`.

### Miscellanous Utilities

The library also contains many miscellaneous utilities that we ended up writing to help with our research. We hope that some of these are useful to folks! It is also entirely possible that there are better ways to accomplish a lot of these functionalities, and please feel free to open a PR and improve on these!

- `train/utils.py` provides the following
    - `TrainState`: A flax train state that applies gradients, and manages batch norm statistics.
    - `aggregated_metrics_dict` and `batchwise_metrics_dict` accumulate metrics dictionaries aggregated across an entire epoch (or batchwise), and appends prefixes for easy logging to Wandb.
    - `train_epoch` and `eval_epoch` serve as `pmapped` examples of multi-device training with metrics accumulation, and added functionality to agnostically use a PyTorch or TF backend for dataloading with the `get_agnostic_batch` function.
    - `get_lr_and_schedule` is a helper function that parses optimiser configs from a `config_dict` and returns the requisite `optax` object with (optional) LR scheduling.
- `utils.py` provides the following
    - `setup_training` is useful to setup Jax training on GPUs, and log some variables to log which devices are being used etc.
    - `flatten_params` flattens a Jax PyTree into a `jnp.ndarray`
    - `flatten_jacobian` flattens a Jacobian (PyTree) into a `jnp.ndarray`
    - `print_params` is a heper utility that prints out the values of each leaf in a Jax PyTree.
    - `print_param_shapes` is a heper utility that prints out the shapes of each leaf in a Jax PyTree.