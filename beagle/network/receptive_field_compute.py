"""Core receptive field computation functions.

Based on "Computing Receptive Fields of Convolutional Neural Networks"
by Araujo, Norris, and Sim (2019): https://distill.pub/2019/computing-receptive-fields/
"""

from __future__ import annotations

from beagle.network.receptive_field_helpers import (
    compute_effective_kernel_size,
    compute_stride_product,
    extract_padding_1d,
    extract_value_for_dim,
)
from beagle.network.receptive_field_types import LayerConfig, ReceptiveFieldInfo


def compute_receptive_field_size_1d(layers: list[LayerConfig], dim: int = 0) -> int:
    """Compute receptive field size for single-path network (pure function).

    Implements the closed-form formula:
    r_0 = sum_{l=1}^{L} ((k_l - 1) * prod_{i=1}^{l-1} s_i) + 1

    Where k_l is the effective kernel size (accounting for dilation).

    Args:
        layers: Sequence of layer configurations from input to output
        dim: Dimension index (0 for height, 1 for width)

    Returns:
        Receptive field size in the input
    """
    if not layers:
        return 1

    rf_size = 1
    for layer_idx, layer in enumerate(layers):
        # Extract parameters for this dimension
        k = extract_value_for_dim(layer.kernel_size, dim)
        d = extract_value_for_dim(layer.dilation, dim)

        # Compute effective kernel size with dilation
        k_eff = compute_effective_kernel_size(k, d)

        # Get product of strides for all previous layers
        prev_strides = [
            extract_value_for_dim(layers[i].stride, dim) for i in range(layer_idx)
        ]
        stride_prod = compute_stride_product(prev_strides)

        # Add contribution of this layer
        rf_size += (k_eff - 1) * stride_prod

    return rf_size


def compute_receptive_field_stride_1d(layers: list[LayerConfig], dim: int = 0) -> int:
    """Compute effective stride from input to output (pure function).

    The effective stride is the product of all layer strides:
    stride_0 = prod_{i=1}^{L} s_i

    Args:
        layers: Sequence of layer configurations from input to output
        dim: Dimension index (0 for height, 1 for width)

    Returns:
        Effective stride from input to output
    """
    strides = [extract_value_for_dim(layer.stride, dim) for layer in layers]
    return compute_stride_product(strides)


def compute_receptive_field_padding_1d(layers: list[LayerConfig], dim: int = 0) -> int:
    """Compute effective padding from input (pure function).

    Implements:
    padding_0 = sum_{l=1}^{L} p_l * prod_{i=1}^{l-1} s_i

    Args:
        layers: Sequence of layer configurations from input to output
        dim: Dimension index (0 for height, 1 for width)

    Returns:
        Effective padding from input
    """
    if not layers:
        return 0

    total_padding = 0
    for layer_idx, layer in enumerate(layers):
        # Extract left padding for this dimension
        p_left, _ = extract_padding_1d(layer.padding, dim)

        # Get product of strides for all previous layers
        prev_strides = [
            extract_value_for_dim(layers[i].stride, dim) for i in range(layer_idx)
        ]
        stride_prod = compute_stride_product(prev_strides)

        # Add contribution of this layer
        total_padding += p_left * stride_prod

    return total_padding


def compute_receptive_field(
    layers: list[LayerConfig],
) -> ReceptiveFieldInfo:
    """Compute receptive field for a single-path network (pure function).

    This function computes the receptive field size, effective stride, and
    effective padding for a fully-convolutional network with a single path
    from input to output.

    For 2D inputs (images), computations are performed independently for each
    spatial dimension (height and width).

    Args:
        layers: Sequence of layer configurations from input to output

    Returns:
        ReceptiveFieldInfo containing size, stride, and padding
    """
    if not layers:
        return ReceptiveFieldInfo(size=1, stride=1, padding=0)

    # Check if layers are 1D or 2D by examining first layer
    # Integers are treated as 2D (square kernels), only explicit single-element
    # tuples are treated as 1D
    first_k = layers[0].kernel_size
    is_1d = isinstance(first_k, tuple) and len(first_k) == 1

    if is_1d:
        size = compute_receptive_field_size_1d(layers, dim=0)
        stride = compute_receptive_field_stride_1d(layers, dim=0)
        padding = compute_receptive_field_padding_1d(layers, dim=0)
        return ReceptiveFieldInfo(size=size, stride=stride, padding=padding)
    else:
        # 2D: compute for both dimensions independently
        size_h = compute_receptive_field_size_1d(layers, dim=0)
        size_w = compute_receptive_field_size_1d(layers, dim=1)
        stride_h = compute_receptive_field_stride_1d(layers, dim=0)
        stride_w = compute_receptive_field_stride_1d(layers, dim=1)
        padding_h = compute_receptive_field_padding_1d(layers, dim=0)
        padding_w = compute_receptive_field_padding_1d(layers, dim=1)
        return ReceptiveFieldInfo(
            size=(size_h, size_w),
            stride=(stride_h, stride_w),
            padding=(padding_h, padding_w),
        )
