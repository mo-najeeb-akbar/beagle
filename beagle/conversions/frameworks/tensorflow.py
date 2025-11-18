from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from beagle.conversions.dispatch import ConversionRegistry
from beagle.conversions.types import Layer, ParamDict, Tolerance
from beagle.conversions.verify import compare_arrays


def _to_numpy(arr: any) -> np.ndarray:
    return np.array(jnp.asarray(arr))


def transfer_conv2d(layer: Layer, params: ParamDict) -> None:
    kernel = _to_numpy(params["kernel"])

    if "bias" in params:
        bias = _to_numpy(params["bias"])
        layer.set_weights([kernel, bias])
    else:
        layer.set_weights([kernel])


def transfer_conv2d_transpose(layer: Layer, params: ParamDict) -> None:
    kernel = _to_numpy(params["kernel"])
    kernel = np.transpose(kernel, (0, 1, 3, 2))
    kernel = kernel[::-1, ::-1, :, :]

    if "bias" in params:
        bias = _to_numpy(params["bias"])
        layer.set_weights([kernel, bias])
    else:
        layer.set_weights([kernel])


def transfer_dense(layer: Layer, params: ParamDict) -> None:
    kernel = _to_numpy(params["kernel"])

    if "bias" in params:
        bias = _to_numpy(params["bias"])
        layer.set_weights([kernel, bias])
    else:
        layer.set_weights([kernel])


def transfer_group_norm(layer: Layer, params: ParamDict) -> None:
    gamma = _to_numpy(params["scale"])
    beta = _to_numpy(params["bias"])
    layer.set_weights([gamma, beta])


def transfer_batch_norm(layer: Layer, params: ParamDict) -> None:
    gamma = _to_numpy(params["scale"])
    beta = _to_numpy(params["bias"])

    if "mean" in params and "var" in params:
        mean = _to_numpy(params["mean"])
        var = _to_numpy(params["var"])
    else:
        shape = gamma.shape
        mean = np.zeros(shape, dtype=np.float32)
        var = np.ones(shape, dtype=np.float32)

    layer.set_weights([gamma, beta, mean, var])


def verify_conv2d(
    layer: Layer, params: ParamDict, tolerance: Tolerance
) -> tuple[bool, list[str]]:
    weights = layer.get_weights()
    kernel = _to_numpy(params["kernel"])

    results = []
    success, msg = compare_arrays(weights[0], kernel, "kernel", tolerance)
    results.append((success, msg))

    if "bias" in params and len(weights) > 1:
        bias = _to_numpy(params["bias"])
        success_bias, msg_bias = compare_arrays(weights[1], bias, "bias", tolerance)
        results.append((success_bias, msg_bias))

    all_success = all(s for s, _ in results)
    messages = [m for _, m in results]
    return all_success, messages


def verify_conv2d_transpose(
    layer: Layer, params: ParamDict, tolerance: Tolerance
) -> tuple[bool, list[str]]:
    weights = layer.get_weights()
    kernel = _to_numpy(params["kernel"])
    kernel = np.transpose(kernel, (0, 1, 3, 2))
    kernel = kernel[::-1, ::-1, :, :]

    results = []
    success, msg = compare_arrays(weights[0], kernel, "kernel", tolerance)
    results.append((success, msg))

    if "bias" in params and len(weights) > 1:
        bias = _to_numpy(params["bias"])
        success_bias, msg_bias = compare_arrays(weights[1], bias, "bias", tolerance)
        results.append((success_bias, msg_bias))

    all_success = all(s for s, _ in results)
    messages = [m for _, m in results]
    return all_success, messages


def verify_dense(
    layer: Layer, params: ParamDict, tolerance: Tolerance
) -> tuple[bool, list[str]]:
    weights = layer.get_weights()
    kernel = _to_numpy(params["kernel"])

    results = []
    success, msg = compare_arrays(weights[0], kernel, "kernel", tolerance)
    results.append((success, msg))

    if "bias" in params and len(weights) > 1:
        bias = _to_numpy(params["bias"])
        success_bias, msg_bias = compare_arrays(weights[1], bias, "bias", tolerance)
        results.append((success_bias, msg_bias))

    all_success = all(s for s, _ in results)
    messages = [m for _, m in results]
    return all_success, messages


def verify_group_norm(
    layer: Layer, params: ParamDict, tolerance: Tolerance
) -> tuple[bool, list[str]]:
    weights = layer.get_weights()
    gamma = _to_numpy(params["scale"])
    beta = _to_numpy(params["bias"])

    results = []
    success_gamma, msg_gamma = compare_arrays(weights[0], gamma, "scale", tolerance)
    results.append((success_gamma, msg_gamma))

    success_beta, msg_beta = compare_arrays(weights[1], beta, "bias", tolerance)
    results.append((success_beta, msg_beta))

    all_success = all(s for s, _ in results)
    messages = [m for _, m in results]
    return all_success, messages


def verify_batch_norm(
    layer: Layer, params: ParamDict, tolerance: Tolerance
) -> tuple[bool, list[str]]:
    weights = layer.get_weights()
    gamma = _to_numpy(params["scale"])
    beta = _to_numpy(params["bias"])

    results = []
    success_gamma, msg_gamma = compare_arrays(weights[0], gamma, "scale", tolerance)
    results.append((success_gamma, msg_gamma))

    success_beta, msg_beta = compare_arrays(weights[1], beta, "bias", tolerance)
    results.append((success_beta, msg_beta))

    if "mean" in params and "var" in params and len(weights) >= 4:
        mean = _to_numpy(params["mean"])
        var = _to_numpy(params["var"])

        success_mean, msg_mean = compare_arrays(weights[2], mean, "mean", tolerance)
        results.append((success_mean, msg_mean))

        success_var, msg_var = compare_arrays(weights[3], var, "var", tolerance)
        results.append((success_var, msg_var))

    all_success = all(s for s, _ in results)
    messages = [m for _, m in results]
    return all_success, messages


def create_tf_registry() -> ConversionRegistry:
    registry = ConversionRegistry()

    registry.register_transfer("Conv2D", transfer_conv2d)
    registry.register_transfer("Conv2DTranspose", transfer_conv2d_transpose)
    registry.register_transfer("Dense", transfer_dense)
    registry.register_transfer("GroupNormalization", transfer_group_norm)
    registry.register_transfer("BatchNormalization", transfer_batch_norm)

    registry.register_verify("Conv2D", verify_conv2d)
    registry.register_verify("Conv2DTranspose", verify_conv2d_transpose)
    registry.register_verify("Dense", verify_dense)
    registry.register_verify("GroupNormalization", verify_group_norm)
    registry.register_verify("BatchNormalization", verify_batch_norm)

    return registry

