from absl import logging
import torch
import jax
from flax import linen as nn
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from typing import Any

PyTree = Any

def convert_model_param_keys(pytorch_name, resnet="18"):
    if resnet == "18":
      block_name = "ResNetBlock"
      layer_list = [2, 2, 2, 2]
    else:
      raise RuntimeError("Choose one of {'18'}.")

    split = pytorch_name.split('.')
    
    #################### First Layer Weights and BN Layers #####################
    # blt models have a different naming convention for the first layer weights
    # and BN layers, where conv1.0.weight -> conv1.weight and conv1.1 refers to
    # the BN layers
    # conv1.0.weight -> (params, conv_init, kernel)
    if split[0] == "conv1" and split[1] == "0":
      if split[2] in ['bias']:
        logging.warning("conv1.0.bias is not a valid key for BLT models")
        return None

      return ("params", "conv_init", "kernel")
  
    # bn1.{weight|bias|running_mean|running_var} -> ({params|batch_stats}, bn_init, {scale|bias|mean|var})
    if split[0] == "conv1" and split[1] == "1":
      if split[2] in ['num_batches_tracked']:
        logging.warning(f"Ignore num_batches_tracked for layer {pytorch_name}")
        return None

      return ("params" if split[2] in ["weight", "bias"] else "batch_stats",
              "bn_init",
              "scale" if split[2] == "weight" else split[2][8:] if split[2] in ["running_mean", "running_var"] else "bias")
    
    ################### Conv Layer Weights and BN Layers #######################
    
    # layer_list.{i}.conv{j}.weight -> (params, ResNetBlock_{i}, Conv_{j-1}, kernel)
    if len(split) == 4 and split[0] == 'layer_list' and split[1].isdigit() and split[2][:-1] == 'conv':
      if split[3] in ['bias']:
        logging.warning(f"{pytorch_name} is not a valid key for BLT models")
        return None

      return ("params",
              f"{block_name}_{split[1]}",
              f"Conv_{int(split[2][-1]) - 1}",
              "kernel")

    # layer_list.{i}.bn{j}.{weight|bias|running_mean|running_var} -> ({params|batch_stats}, ResNetBlock_{i}, BatchNorm_{j-1}, {scale|bias|mean|var})
    if len(split) == 4 and split[0] == 'layer_list' and split[1].isdigit() and split[2][:-1] == 'bn':
      if split[3] in ['num_batches_tracked']:
        logging.warning(f"Ignore num_batches_tracked for layer {pytorch_name}")
        return None

      return ("params" if split[3] in ["weight", "bias"] else "batch_stats",
              f"{block_name}_{split[1]}",
              f"BatchNorm_{int(split[2][-1]) - 1}",
              "scale" if split[3] == "weight" else split[3][8:] if split[3] in ["running_mean", "running_var"] else "bias")


    ############################ Downsampling Layers ###########################

    # layer_list.{i}.downsample.0.weight -> (params, ResNetBlock_{i}, conv_proj, kernel)
    if len(split) == 5 and split[0] == 'layer_list' and split[1].isdigit() and split[2] == 'downsample' and split[3] == '0':
      if split[4] in ['bias']:
        logging.warning(f"{pytorch_name} is not a valid key for BLT models")
        return None

      return ("params",
              f"{block_name}_{split[1]}",
              "conv_proj",
              "kernel")

    # layer_list.{i}.downsample.1.{weight|bias|running_mean|running_var} -> ({params|batch_stats}, ResNetBlock_{i}, norm_proj, {scale|bias|mean|var})
    if len(split) == 5 and split[0] == 'layer_list' and split[1].isdigit() and split[2] == 'downsample' and split[3] == '1':
      if split[4] in ['num_batches_tracked']:
        logging.warning(f"Ignore num_batches_tracked for layer {pytorch_name}")
        return None

      return ("params" if split[4] in ["weight", "bias"] else "batch_stats",
              f"{block_name}_{split[1]}",
              "norm_proj",
              "scale" if split[4] == "weight" else split[4][8:] if split[4] in ["running_mean", "running_var"] else "bias")
    
    ############################# Output Layers ################################
    # output_block.{weight|bias} -> (params, Dense_0, {kernel|bias})
    if len(split) == 2 and split[0] == 'output_block':
      return ("params", "Dense_0", "bias" if split[1] == "bias" else "kernel")
  


def convert_resnet_model(
    model_name: str,
    pytorch_model: torch.nn.Module,
    jax_params: PyTree) -> PyTree:
    """Utility function to transfer params from Pytorch models to Flax."""
    # jax_params = flatten_dict(unfreeze(jax_params))
    # print(jax_params)
    if model_name == 'resnet18':
        jax_to_pytorch_keys, converted_jax_params = {}, {}
        for key, param in pytorch_model.state_dict().items():
            if convert_model_param_keys(key, resnet='18') is not None:
                jax_to_pytorch_keys[convert_model_param_keys(key, resnet='18')] = key

            if len(param.shape) != 4:
                converted_jax_params[key] = param.numpy().T
            else:
                # Pytorch [N, C, H, W] -> Jax [H, W, C, N]
                converted_jax_params[key] = param.numpy().transpose((2, 3, 1, 0))
        
        print(jax_to_pytorch_keys)
        new_jax_params = {key: converted_jax_params[jax_to_pytorch_keys[key]] for key in jax_params.keys()}
        new_jax_params = freeze(unflatten_dict(new_jax_params))
        
        return new_jax_params

    else:
        raise NotImplementedError('Only resnet18 models converted so far.')