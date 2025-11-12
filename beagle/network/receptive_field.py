"""Receptive field computation for convolutional neural networks.

This module re-exports all receptive field functionality from the submodules.

Based on "Computing Receptive Fields of Convolutional Neural Networks"
by Araujo, Norris, and Sim (2019): https://distill.pub/2019/computing-receptive-fields/
"""

from beagle.network.receptive_field_compute import (
    compute_receptive_field,
    compute_receptive_field_padding_1d,
    compute_receptive_field_size_1d,
    compute_receptive_field_stride_1d,
)
from beagle.network.receptive_field_helpers import (
    compute_effective_kernel_size,
    compute_stride_product,
)
from beagle.network.receptive_field_types import LayerConfig, ReceptiveFieldInfo
from beagle.network.receptive_field_utils import (
    compute_output_position_in_input,
    create_layer_from_module,
    print_receptive_field_report,
)

__all__ = [
    "LayerConfig",
    "ReceptiveFieldInfo",
    "compute_receptive_field",
    "compute_receptive_field_size_1d",
    "compute_receptive_field_stride_1d",
    "compute_receptive_field_padding_1d",
    "compute_effective_kernel_size",
    "compute_stride_product",
    "compute_output_position_in_input",
    "create_layer_from_module",
    "print_receptive_field_report",
]

