"""Helper functions for receptive field computation.

Based on "Computing Receptive Fields of Convolutional Neural Networks"
by Araujo, Norris, and Sim (2019): https://distill.pub/2019/computing-receptive-fields/
"""

from __future__ import annotations

from functools import reduce


def normalize_to_2d(value: int | tuple[int, ...]) -> tuple[int, int]:
    """Normalize scalar or tuple to 2D tuple (pure function)."""
    if isinstance(value, int):
        return (value, value)
    elif len(value) == 1:
        # 1D tuple: (x,) -> treat as scalar
        return (value[0], value[0])
    return (value[0], value[1])


def extract_value_for_dim(value: int | tuple[int, ...], dim: int) -> int:
    """Extract value for a specific dimension (pure function).

    Args:
        value: Scalar or tuple value
        dim: Dimension index

    Returns:
        Value for the specified dimension
    """
    if isinstance(value, int):
        return value
    elif len(value) == 1:
        # 1D: only has dimension 0
        return value[0]
    else:
        # 2D or higher: extract the specific dimension
        return value[dim]


def extract_padding_1d(
    padding: int | tuple[int, ...] | tuple[tuple[int, int], tuple[int, int]],
    dim: int = 0,
) -> tuple[int, int]:
    """Extract left and right padding for a dimension (pure function).

    Args:
        padding: Padding specification (scalar, 1D tuple, or 2D tuple)
        dim: Dimension index (0 for height/first dim, 1 for width/second dim)

    Returns:
        Tuple of (left_padding, right_padding) for the specified dimension
    """
    if isinstance(padding, int):
        return (padding, padding)
    elif len(padding) == 1:
        # Single element tuple: (padding,) for 1D case
        return (padding[0], padding[0])
    elif len(padding) == 2 and isinstance(padding[0], int):
        # Single dimension padding: (left, right)
        return padding  # type: ignore
    else:
        # 2D padding: ((top, bottom), (left, right))
        return padding[dim]  # type: ignore


def compute_effective_kernel_size(kernel_size: int, dilation: int) -> int:
    """Compute effective kernel size with dilation (pure function).

    For dilated (atrous) convolutions, the effective kernel size is:
    effective_kernel_size = dilation * (kernel_size - 1) + 1

    Args:
        kernel_size: Original kernel size
        dilation: Dilation factor

    Returns:
        Effective kernel size accounting for dilation
    """
    return dilation * (kernel_size - 1) + 1


def compute_stride_product(strides: list[int]) -> int:
    """Compute product of strides up to layer l-1 (pure function).

    This implements: prod_{i=1}^{l-1} s_i

    Args:
        strides: List of stride values for layers 1 to l-1

    Returns:
        Product of all strides
    """
    return reduce(lambda x, y: x * y, strides, 1)

