from typing import Any

import torch
from absl import logging
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

PyTree = Any

FC_MAP = {"weight": "kernel", "bias": "bias"}
CONV_MAP = {"weight": "kernel", "bias": "bias"}
BN_MAP = {
    "weight": "scale",
    "bias": "bias",
    "running_mean": "mean",
    "running_var": "var",
}


def convert_lenet_param_keys(pytorch_name, model_name="LeNetSmall"):
    """Convert LeNet models from bayesian-lottery-tickets into Flax models.

    LeNet Models have a naming convention where conv{i} refers to the conv
    blocks, and fc refers to the final output layer.
    """
    split = pytorch_name.split(".")

    # conv{i}.conv.{weight|bias} -> (params, conv{i}, Conv_0, {kernel|bias})
    if split[0][:-1] == "conv" and split[1] == "conv":
        return ("params", split[0], "Conv_0", CONV_MAP[split[2]])

    # conv{i}.bn.{weight|bias|running_mean|running_var} -> ({params|batch_stats}, conv{i}, BatchNorm_0, {scale|bias|mean|var})
    if split[0][:-1] == "conv" and split[1] == "bn":
        if split[2] in ["num_batches_tracked"]:
            logging.warning(f"Ignore num_batches_tracked for layer {pytorch_name}")
            return None

        return (
            "params" if split[2] in ["weight", "bias"] else "batch_stats",
            split[0],
            "BatchNorm_0",
            BN_MAP[split[2]],
        )

    # fc.{weight|bias} -> (params, Dense_0, {kernel|bias})
    if split[0] == "fc":
        return ("params", "dense", FC_MAP[split[1]])


def convert_resnet_param_keys(pytorch_name, model_name="resnet18"):
    """Convert ResNet models from bayesian-lottery-tickets into Flax models.

    blt models have a different naming convention for the first layer weights
    and BN layers, where conv1.0.weight -> conv1.weight and conv1.1 refers to
    the BN layers.
    """
    split = pytorch_name.split(".")

    if model_name == "resnet18":
        block_name = "ResNetBlock"
        layer_list = [2, 2, 2, 2]
    elif model_name == "resnet50":
        block_name = "BottleneckResNetBlock"
        layer_list = [3, 4, 6, 3]
    else:
        raise NotImplementedError("Only resnet18 and resnet50 implemented for now.")

    #################### First Layer Weights and BN Layers #################
    # TODO: Can probably combine these if-else statements.
    if model_name == "resnet50":  # 3x3 conv blocks are involved    
        conv1_layer_map = {"0": "0", "3": "1", "6": "2"}
        # conv1.{0, 3, 6}.weight -> (params, conv_init_{0, 1, 2}, kernel)
        if split[0] == "conv1" and split[1] in conv1_layer_map.keys():
            if split[2] in ["bias"]:
                logging.warning(
                    f"conv1.{conv1_layer_map[split[1]]}.bias is not a valid key for BLT models")
                return None

            return ("params", f"conv_init_{conv1_layer_map[split[1]]}", "kernel")
        
        conv1_bn_layer_map = {"1": "0", "4": "1", "7": "2"}
        # conv1.{1, 4, 7}.{weight|bias|running_mean|running_var} -> ({params|batch_stats}, bn_init, {scale|bias|mean|var})
        if split[0] == "conv1" and split[1] in conv1_bn_layer_map.keys():
            if split[2] in ["num_batches_tracked"]:
                logging.warning(f"Ignore num_batches_tracked for layer {pytorch_name}")
                return None

            return (
                "params" if split[2] in ["weight", "bias"] else "batch_stats",
                f"bn_init_{conv1_bn_layer_map[split[1]]}",
                BN_MAP[split[2]],
            )
    else:  # 1x3 or 1x7 blocks only
        # conv1.0.weight -> (params, conv_init, kernel)
        if split[0] == "conv1" and split[1] == "0":
            if split[2] in ["bias"]:
                logging.warning("conv1.0.bias is not a valid key for BLT models")
                return None

            return ("params", "conv_init", "kernel")

        # conv1.1.{weight|bias|running_mean|running_var} -> ({params|batch_stats}, bn_init, {scale|bias|mean|var})
        if split[0] == "conv1" and split[1] == "1":
            if split[2] in ["num_batches_tracked"]:
                logging.warning(f"Ignore num_batches_tracked for layer {pytorch_name}")
                return None

            return (
                "params" if split[2] in ["weight", "bias"] else "batch_stats",
                "bn_init",
                BN_MAP[split[2]],
            )

    ################### Conv Layer Weights and BN Layers ###################

    # layer_list.{i}.conv{j}.weight -> (params, ResNetBlock_{i}, Conv_{j-1}, kernel)
    if (
        len(split) == 4
        and split[0] == "layer_list"
        and split[1].isdigit()
        and split[2][:-1] == "conv"
    ):
        if split[3] in ["bias"]:
            logging.warning(f"{pytorch_name} is not a valid key for BLT models")
            return None

        return (
            "params",
            f"{block_name}_{split[1]}",
            f"Conv_{int(split[2][-1]) - 1}",
            "kernel",
        )

    # layer_list.{i}.bn{j}.{weight|bias|running_mean|running_var} -> ({params|batch_stats}, ResNetBlock_{i}, BatchNorm_{j-1}, {scale|bias|mean|var})
    if (
        len(split) == 4
        and split[0] == "layer_list"
        and split[1].isdigit()
        and split[2][:-1] == "bn"
    ):
        if split[3] in ["num_batches_tracked"]:
            logging.warning(f"Ignore num_batches_tracked for layer {pytorch_name}")
            return None

        return (
            "params" if split[3] in ["weight", "bias"] else "batch_stats",
            f"{block_name}_{split[1]}",
            f"BatchNorm_{int(split[2][-1]) - 1}",
            BN_MAP[split[3]],
        )

    ############################ Downsampling Layers #######################

    # layer_list.{i}.downsample.0.weight -> (params, ResNetBlock_{i}, conv_proj, kernel)
    if (
        len(split) == 5
        and split[0] == "layer_list"
        and split[1].isdigit()
        and split[2] == "downsample"
        and split[3] == "0"
    ):
        if split[4] in ["bias"]:
            logging.warning(f"{pytorch_name} is not a valid key for BLT models")
            return None

        return ("params", f"{block_name}_{split[1]}", "conv_proj", "kernel")

    # layer_list.{i}.downsample.1.{weight|bias|running_mean|running_var} -> ({params|batch_stats}, ResNetBlock_{i}, norm_proj, {scale|bias|mean|var})
    if (
        len(split) == 5
        and split[0] == "layer_list"
        and split[1].isdigit()
        and split[2] == "downsample"
        and split[3] == "1"
    ):
        if split[4] in ["num_batches_tracked"]:
            logging.warning(f"Ignore num_batches_tracked for layer {pytorch_name}")
            return None

        return (
            "params" if split[4] in ["weight", "bias"] else "batch_stats",
            f"{block_name}_{split[1]}",
            "norm_proj",
            BN_MAP[split[4]],
        )

    ############################# Output Layers ############################
    # output_block.{weight|bias} -> (params, Dense_0, {kernel|bias})
    if len(split) == 2 and split[0] == "output_block":
        return ("params", "Dense_0", FC_MAP[split[1]])


def convert_model(
    model_name: str, pytorch_model: torch.nn.Module, jax_params: PyTree,
    dataparallel_model: bool = False
) -> PyTree:
    """Utility function to transfer params from Pytorch models to Flax."""
    # TODO: Add a check that params is frozen and not flat, and not repeat
    jax_params = flatten_dict(unfreeze(jax_params))

    if model_name in ["resnet18", "resnet50"]:
        convert_keys = convert_resnet_param_keys
    elif model_name in ["LeNetSmall", "LeNet", "LeNetBig"]:
        convert_keys = convert_lenet_param_keys
    else:
        raise NotImplementedError(f"{model_name} conversion not implemented yet.")

    jax_to_pytorch_keys, converted_jax_params = {}, {}
    print(pytorch_model.state_dict().keys())
    for key, param in pytorch_model.state_dict().items():
        if dataparallel_model:
            key = key[7:]
        if convert_keys(key, model_name) is not None:
            jax_to_pytorch_keys[convert_keys(key, model_name)] = key

        if len(param.shape) != 4:
            converted_jax_params[key] = param.numpy().T
        else:
            # Pytorch # [outC, inC, kH, kW] -> Jax [kH, kW, inC, outC]
            converted_jax_params[key] = param.numpy().transpose((2, 3, 1, 0))

    print(jax_to_pytorch_keys)
    new_jax_params = {
        key: converted_jax_params[jax_to_pytorch_keys[key]] for key in jax_params.keys()
    }
    new_jax_params = freeze(unflatten_dict(new_jax_params))

    return new_jax_params
