"""Utility functions for receptive field computation.

Based on "Computing Receptive Fields of Convolutional Neural Networks"
by Araujo, Norris, and Sim (2019): https://distill.pub/2019/computing-receptive-fields/
"""

from __future__ import annotations

from beagle.network.receptive_field_compute import compute_receptive_field
from beagle.network.receptive_field_types import LayerConfig, ReceptiveFieldInfo


def compute_output_position_in_input(
    output_position: int,
    rf_info: ReceptiveFieldInfo,
    dim: int = 0,
) -> tuple[int, int]:
    """Compute input region for an output feature (pure function).

    Given a position in the output feature map, compute the corresponding
    receptive field region [start, end) in the input.

    Implements:
    start = output_position * stride - padding
    end = start + rf_size

    Args:
        output_position: Position in output feature map (0-indexed)
        rf_info: Receptive field information for the network
        dim: Dimension index (0 for height, 1 for width)

    Returns:
        Tuple of (start, end) indices in input (end is exclusive)
    """
    if isinstance(rf_info.size, int):
        size = rf_info.size
        stride = rf_info.stride  # type: ignore
        padding = rf_info.padding  # type: ignore
    else:
        size = rf_info.size[dim]
        stride = rf_info.stride[dim]  # type: ignore
        padding = rf_info.padding[dim]  # type: ignore

    start = output_position * stride - padding
    end = start + size
    return (start, end)


def create_layer_from_module(
    module_type: str,
    kernel_size: int | tuple[int, int] = 1,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = 0,
    dilation: int | tuple[int, int] = 1,
    name: str = "",
) -> LayerConfig:
    """Create LayerConfig from common module types (pure function).

    Helper to create layer configurations for common operations:
    - Conv2d, Conv1d: use provided parameters
    - MaxPool2d, AvgPool2d: use provided parameters
    - ReLU, BatchNorm, Dropout: kernel_size=1, stride=1, padding=0

    Args:
        module_type: Type of module (e.g., "Conv2d", "MaxPool2d", "ReLU")
        kernel_size: Kernel size
        stride: Stride
        padding: Padding
        dilation: Dilation
        name: Optional name

    Returns:
        LayerConfig for the module
    """
    # Elementwise operations don't change spatial dimensions
    elementwise_ops = {
        "ReLU",
        "LeakyReLU",
        "GELU",
        "SiLU",
        "BatchNorm2d",
        "BatchNorm1d",
        "LayerNorm",
        "Dropout",
        "Identity",
    }

    if module_type in elementwise_ops:
        return LayerConfig(
            kernel_size=1, stride=1, padding=0, dilation=1, name=name or module_type
        )

    return LayerConfig(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        name=name or module_type,
    )


def print_receptive_field_report(
    layers: list[LayerConfig],
    rf_info: ReceptiveFieldInfo | None = None,
) -> None:
    """Print detailed receptive field analysis (impure function - I/O).

    Args:
        layers: Sequence of layer configurations
        rf_info: Pre-computed receptive field info (computed if None)
    """
    if rf_info is None:
        rf_info = compute_receptive_field(layers)

    print("\n" + "=" * 70)
    print("RECEPTIVE FIELD ANALYSIS")
    print("=" * 70)
    print(f"\nFinal Receptive Field: {rf_info}")
    print(f"\nNumber of layers: {len(layers)}")

    # Compute cumulative receptive field at each layer
    print("\n" + "-" * 70)
    print("Layer-by-layer receptive field growth:")
    print("-" * 70)
    print(f"{'Layer':<20} {'Type':<15} {'k':<8} {'s':<8} {'RF Size':<15}")
    print("-" * 70)

    for i in range(len(layers)):
        partial_rf = compute_receptive_field(layers[: i + 1])
        layer = layers[i]
        layer_name = layer.name or f"Layer {i+1}"

        # Determine layer type from name or parameters
        if layer.kernel_size == 1 and layer.stride == 1:
            layer_type = "Elementwise"
        elif "pool" in layer.name.lower():
            layer_type = "Pooling"
        elif "conv" in layer.name.lower():
            layer_type = "Convolution"
        else:
            layer_type = "Operation"

        k_str = (
            f"{layer.kernel_size}"
            if isinstance(layer.kernel_size, int)
            else f"{layer.kernel_size[0]}x{layer.kernel_size[1]}"
        )
        s_str = (
            f"{layer.stride}"
            if isinstance(layer.stride, int)
            else f"{layer.stride[0]}x{layer.stride[1]}"
        )
        rf_str = (
            f"{partial_rf.size}"
            if isinstance(partial_rf.size, int)
            else f"{partial_rf.size[0]}x{partial_rf.size[1]}"
        )

        print(f"{layer_name:<20} {layer_type:<15} {k_str:<8} {s_str:<8} {rf_str:<15}")

    print("=" * 70 + "\n")

