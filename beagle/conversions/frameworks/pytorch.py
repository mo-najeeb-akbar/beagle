from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from beagle.conversions.dispatch import ConversionRegistry
from beagle.conversions.types import Layer, ParamDict, Tolerance
from beagle.conversions.verify import compare_arrays


def _to_numpy(arr: any) -> np.ndarray:
    return np.array(jnp.asarray(arr))


def _get_param_by_name(layer: Layer, name: str) -> any:
    for param_name, param in layer.named_parameters():
        if param_name == name:
            return param
    return None


def transfer_conv2d(layer: Layer, params: ParamDict) -> None:
    import torch
    
    kernel = _to_numpy(params["kernel"])
    
    if len(kernel.shape) == 4:
        kernel = np.transpose(kernel, (3, 2, 0, 1))
    elif len(kernel.shape) == 2:
        kernel = np.transpose(kernel, (1, 0))
        kernel = kernel.reshape(kernel.shape[0], kernel.shape[1], 1, 1)
    else:
        raise ValueError(f"Unexpected kernel shape: {kernel.shape}")

    weight_param = _get_param_by_name(layer, "weight")
    if weight_param is not None:
        weight_param.data = torch.from_numpy(kernel)

    if "bias" in params:
        bias = _to_numpy(params["bias"])
        bias_param = _get_param_by_name(layer, "bias")
        if bias_param is not None:
            bias_param.data = torch.from_numpy(bias)


def transfer_conv2d_transpose(layer: Layer, params: ParamDict) -> None:
    import torch
    
    kernel = _to_numpy(params["kernel"])
    kernel = np.transpose(kernel, (2, 3, 0, 1))

    weight_param = _get_param_by_name(layer, "weight")
    if weight_param is not None:
        weight_param.data = torch.from_numpy(kernel)

    if "bias" in params:
        bias = _to_numpy(params["bias"])
        bias_param = _get_param_by_name(layer, "bias")
        if bias_param is not None:
            bias_param.data = torch.from_numpy(bias)


def transfer_linear(layer: Layer, params: ParamDict) -> None:
    import torch
    
    kernel = _to_numpy(params["kernel"])
    
    if len(kernel.shape) == 2:
        kernel = np.transpose(kernel, (1, 0))
    elif len(kernel.shape) == 4 and kernel.shape[0] == 1 and kernel.shape[1] == 1:
        kernel = np.transpose(kernel, (3, 2, 0, 1))
    
    weight_param = _get_param_by_name(layer, "weight")
    if weight_param is not None:
        weight_param.data = torch.from_numpy(kernel)

    if "bias" in params:
        bias = _to_numpy(params["bias"])
        bias_param = _get_param_by_name(layer, "bias")
        if bias_param is not None:
            bias_param.data = torch.from_numpy(bias)


def transfer_group_norm(layer: Layer, params: ParamDict) -> None:
    import torch

    gamma = _to_numpy(params["scale"])
    beta = _to_numpy(params["bias"])

    weight_param = _get_param_by_name(layer, "weight")
    bias_param = _get_param_by_name(layer, "bias")

    if weight_param is not None:
        weight_param.data = torch.from_numpy(gamma)
    if bias_param is not None:
        bias_param.data = torch.from_numpy(beta)


def transfer_batch_norm(layer: Layer, params: ParamDict) -> None:
    import torch

    gamma = _to_numpy(params["scale"])
    beta = _to_numpy(params["bias"])

    weight_param = _get_param_by_name(layer, "weight")
    bias_param = _get_param_by_name(layer, "bias")

    if weight_param is not None:
        weight_param.data = torch.from_numpy(gamma)
    if bias_param is not None:
        bias_param.data = torch.from_numpy(beta)

    if "mean" in params and "var" in params:
        mean = _to_numpy(params["mean"])
        var = _to_numpy(params["var"])

        if hasattr(layer, "running_mean") and layer.running_mean is not None:
            layer.running_mean.data = torch.from_numpy(mean)
        if hasattr(layer, "running_var") and layer.running_var is not None:
            layer.running_var.data = torch.from_numpy(var)


def verify_conv2d(
    layer: Layer, params: ParamDict, tolerance: Tolerance
) -> tuple[bool, list[str]]:
    kernel = _to_numpy(params["kernel"])
    
    if len(kernel.shape) == 4:
        kernel = np.transpose(kernel, (3, 2, 0, 1))
    elif len(kernel.shape) == 2:
        kernel = np.transpose(kernel, (1, 0))
        kernel = kernel.reshape(kernel.shape[0], kernel.shape[1], 1, 1)
    else:
        return (False, [f"Unexpected kernel shape: {kernel.shape}"])

    weight_param = _get_param_by_name(layer, "weight")
    results = []

    if weight_param is not None:
        weight_np = weight_param.detach().cpu().numpy()
        success, msg = compare_arrays(weight_np, kernel, "kernel", tolerance)
        results.append((success, msg))

    if "bias" in params:
        bias = _to_numpy(params["bias"])
        bias_param = _get_param_by_name(layer, "bias")
        if bias_param is not None:
            bias_np = bias_param.detach().cpu().numpy()
            success_bias, msg_bias = compare_arrays(bias_np, bias, "bias", tolerance)
            results.append((success_bias, msg_bias))

    all_success = all(s for s, _ in results) if results else False
    messages = [m for _, m in results]
    return all_success, messages


def verify_conv2d_transpose(
    layer: Layer, params: ParamDict, tolerance: Tolerance
) -> tuple[bool, list[str]]:
    kernel = _to_numpy(params["kernel"])
    kernel = np.transpose(kernel, (2, 3, 0, 1))

    weight_param = _get_param_by_name(layer, "weight")
    results = []

    if weight_param is not None:
        weight_np = weight_param.detach().cpu().numpy()
        success, msg = compare_arrays(weight_np, kernel, "kernel", tolerance)
        results.append((success, msg))

    if "bias" in params:
        bias = _to_numpy(params["bias"])
        bias_param = _get_param_by_name(layer, "bias")
        if bias_param is not None:
            bias_np = bias_param.detach().cpu().numpy()
            success_bias, msg_bias = compare_arrays(bias_np, bias, "bias", tolerance)
            results.append((success_bias, msg_bias))

    all_success = all(s for s, _ in results) if results else False
    messages = [m for _, m in results]
    return all_success, messages


def verify_linear(
    layer: Layer, params: ParamDict, tolerance: Tolerance
) -> tuple[bool, list[str]]:
    kernel = _to_numpy(params["kernel"])
    kernel = np.transpose(kernel, (1, 0))

    weight_param = _get_param_by_name(layer, "weight")
    results = []

    if weight_param is not None:
        weight_np = weight_param.detach().cpu().numpy()
        success, msg = compare_arrays(weight_np, kernel, "kernel", tolerance)
        results.append((success, msg))

    if "bias" in params:
        bias = _to_numpy(params["bias"])
        bias_param = _get_param_by_name(layer, "bias")
        if bias_param is not None:
            bias_np = bias_param.detach().cpu().numpy()
            success_bias, msg_bias = compare_arrays(bias_np, bias, "bias", tolerance)
            results.append((success_bias, msg_bias))

    all_success = all(s for s, _ in results) if results else False
    messages = [m for _, m in results]
    return all_success, messages


def verify_group_norm(
    layer: Layer, params: ParamDict, tolerance: Tolerance
) -> tuple[bool, list[str]]:
    gamma = _to_numpy(params["scale"])
    beta = _to_numpy(params["bias"])

    weight_param = _get_param_by_name(layer, "weight")
    bias_param = _get_param_by_name(layer, "bias")

    results = []

    if weight_param is not None:
        weight_np = weight_param.detach().cpu().numpy()
        success_gamma, msg_gamma = compare_arrays(weight_np, gamma, "scale", tolerance)
        results.append((success_gamma, msg_gamma))

    if bias_param is not None:
        bias_np = bias_param.detach().cpu().numpy()
        success_beta, msg_beta = compare_arrays(bias_np, beta, "bias", tolerance)
        results.append((success_beta, msg_beta))

    all_success = all(s for s, _ in results) if results else False
    messages = [m for _, m in results]
    return all_success, messages


def verify_batch_norm(
    layer: Layer, params: ParamDict, tolerance: Tolerance
) -> tuple[bool, list[str]]:
    gamma = _to_numpy(params["scale"])
    beta = _to_numpy(params["bias"])

    weight_param = _get_param_by_name(layer, "weight")
    bias_param = _get_param_by_name(layer, "bias")

    results = []

    if weight_param is not None:
        weight_np = weight_param.detach().cpu().numpy()
        success_gamma, msg_gamma = compare_arrays(weight_np, gamma, "scale", tolerance)
        results.append((success_gamma, msg_gamma))

    if bias_param is not None:
        bias_np = bias_param.detach().cpu().numpy()
        success_beta, msg_beta = compare_arrays(bias_np, beta, "bias", tolerance)
        results.append((success_beta, msg_beta))

    if "mean" in params and "var" in params:
        mean = _to_numpy(params["mean"])
        var = _to_numpy(params["var"])

        if hasattr(layer, "running_mean") and layer.running_mean is not None:
            mean_np = layer.running_mean.detach().cpu().numpy()
            success_mean, msg_mean = compare_arrays(mean_np, mean, "mean", tolerance)
            results.append((success_mean, msg_mean))

        if hasattr(layer, "running_var") and layer.running_var is not None:
            var_np = layer.running_var.detach().cpu().numpy()
            success_var, msg_var = compare_arrays(var_np, var, "var", tolerance)
            results.append((success_var, msg_var))

    all_success = all(s for s, _ in results) if results else False
    messages = [m for _, m in results]
    return all_success, messages


def create_pytorch_registry() -> ConversionRegistry:
    registry = ConversionRegistry()

    registry.register_transfer("Conv2d", transfer_conv2d)
    registry.register_transfer("ConvTranspose2d", transfer_conv2d_transpose)
    registry.register_transfer("Linear", transfer_linear)
    registry.register_transfer("GroupNorm", transfer_group_norm)
    registry.register_transfer("BatchNorm2d", transfer_batch_norm)

    registry.register_verify("Conv2d", verify_conv2d)
    registry.register_verify("ConvTranspose2d", verify_conv2d_transpose)
    registry.register_verify("Linear", verify_linear)
    registry.register_verify("GroupNorm", verify_group_norm)
    registry.register_verify("BatchNorm2d", verify_batch_norm)

    return registry

