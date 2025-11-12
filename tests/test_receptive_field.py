"""Tests for receptive field computation."""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from beagle.network.receptive_field import (
    LayerConfig,
    ReceptiveFieldInfo,
    compute_effective_kernel_size,
    compute_output_position_in_input,
    compute_receptive_field,
    compute_receptive_field_padding_1d,
    compute_receptive_field_size_1d,
    compute_receptive_field_stride_1d,
    compute_stride_product,
    create_layer_from_module,
)


def test_layer_config_validation():
    """Test LayerConfig validates parameters correctly."""
    # Valid configurations
    LayerConfig(kernel_size=3, stride=1, padding=0)
    LayerConfig(kernel_size=(3, 3), stride=(1, 1), padding=((1, 1), (1, 1)))

    # Invalid kernel size
    with pytest.raises(ValueError, match="kernel_size must be positive"):
        LayerConfig(kernel_size=0)

    with pytest.raises(ValueError, match="kernel_size must be positive"):
        LayerConfig(kernel_size=(3, 0))

    # Invalid stride
    with pytest.raises(ValueError, match="stride must be positive"):
        LayerConfig(kernel_size=3, stride=0)


def test_compute_effective_kernel_size():
    """Test effective kernel size computation with dilation."""
    # No dilation
    assert compute_effective_kernel_size(3, 1) == 3
    assert compute_effective_kernel_size(5, 1) == 5

    # With dilation
    assert compute_effective_kernel_size(3, 2) == 5
    assert compute_effective_kernel_size(3, 3) == 7
    assert compute_effective_kernel_size(5, 2) == 9


def test_compute_stride_product():
    """Test stride product computation."""
    assert compute_stride_product([]) == 1
    assert compute_stride_product([2]) == 2
    assert compute_stride_product([2, 2]) == 4
    assert compute_stride_product([2, 2, 2]) == 8
    assert compute_stride_product([4, 2, 1]) == 8


def test_single_layer_receptive_field():
    """Test receptive field for single layer."""
    # Single conv layer, no padding
    layer = LayerConfig(kernel_size=3, stride=1, padding=0)
    rf = compute_receptive_field([layer])
    assert rf.size == (3, 3)
    assert rf.stride == (1, 1)
    assert rf.padding == (0, 0)

    # Single conv layer with stride
    layer = LayerConfig(kernel_size=3, stride=2, padding=0)
    rf = compute_receptive_field([layer])
    assert rf.size == (3, 3)
    assert rf.stride == (2, 2)

    # Single conv layer with padding
    layer = LayerConfig(kernel_size=3, stride=1, padding=1)
    rf = compute_receptive_field([layer])
    assert rf.size == (3, 3)
    assert rf.padding == (1, 1)


def test_two_layer_receptive_field():
    """Test receptive field for two-layer network."""
    # Two 3x3 convs
    layers = [
        LayerConfig(kernel_size=3, stride=1, padding=0, name="conv1"),
        LayerConfig(kernel_size=3, stride=1, padding=0, name="conv2"),
    ]
    rf = compute_receptive_field(layers)
    # RF = 1 + (3-1)*1 + (3-1)*1 = 5
    assert rf.size == (5, 5)
    assert rf.stride == (1, 1)

    # 3x3 conv followed by 2x2 pool (stride 2)
    layers = [
        LayerConfig(kernel_size=3, stride=1, padding=0, name="conv1"),
        LayerConfig(kernel_size=2, stride=2, padding=0, name="pool1"),
    ]
    rf = compute_receptive_field(layers)
    # RF = 1 + (3-1)*1 + (2-1)*1 = 4
    assert rf.size == (4, 4)
    assert rf.stride == (2, 2)


def test_alexnet_style_receptive_field():
    """Test receptive field for AlexNet-style architecture.

    AlexNet conv1: 11x11 kernel, stride 4, padding 2
    Following the Distill article example.
    """
    layers = [
        LayerConfig(kernel_size=11, stride=4, padding=2, name="conv1"),
    ]
    rf = compute_receptive_field(layers)
    assert rf.size == (11, 11)
    assert rf.stride == (4, 4)
    assert rf.padding == (2, 2)

    # Add pooling layer: 3x3, stride 2
    layers.append(LayerConfig(kernel_size=3, stride=2, padding=0, name="pool1"))
    rf = compute_receptive_field(layers)
    # RF = 1 + (11-1)*1 + (3-1)*4 = 1 + 10 + 8 = 19
    assert rf.size == (19, 19)
    assert rf.stride == (8, 8)

    # Add conv2: 5x5, stride 1, padding 2
    layers.append(LayerConfig(kernel_size=5, stride=1, padding=2, name="conv2"))
    rf = compute_receptive_field(layers)
    # RF = 1 + (11-1)*1 + (3-1)*4 + (5-1)*8 = 1 + 10 + 8 + 32 = 51
    assert rf.size == (51, 51)
    assert rf.stride == (8, 8)


def test_vgg_style_receptive_field():
    """Test receptive field for VGG-style architecture with multiple 3x3 convs."""
    # VGG block: two 3x3 convs followed by 2x2 pool
    layers = [
        LayerConfig(kernel_size=3, stride=1, padding=1, name="conv1_1"),
        LayerConfig(kernel_size=3, stride=1, padding=1, name="conv1_2"),
        LayerConfig(kernel_size=2, stride=2, padding=0, name="pool1"),
    ]
    rf = compute_receptive_field(layers)
    # RF = 1 + (3-1)*1 + (3-1)*1 + (2-1)*1 = 1 + 2 + 2 + 1 = 6
    assert rf.size == (6, 6)
    assert rf.stride == (2, 2)

    # Add second VGG block
    layers.extend(
        [
            LayerConfig(kernel_size=3, stride=1, padding=1, name="conv2_1"),
            LayerConfig(kernel_size=3, stride=1, padding=1, name="conv2_2"),
            LayerConfig(kernel_size=2, stride=2, padding=0, name="pool2"),
        ]
    )
    rf = compute_receptive_field(layers)
    # RF = 6 + (3-1)*2 + (3-1)*2 + (2-1)*2 = 6 + 4 + 4 + 2 = 16
    assert rf.size == (16, 16)
    assert rf.stride == (4, 4)


def test_dilated_convolution_receptive_field():
    """Test receptive field with dilated (atrous) convolutions."""
    # 3x3 conv with dilation=2
    layer = LayerConfig(kernel_size=3, stride=1, padding=0, dilation=2)
    rf = compute_receptive_field([layer])
    # Effective kernel size = 2*(3-1) + 1 = 5
    assert rf.size == (5, 5)

    # Two dilated convs
    layers = [
        LayerConfig(kernel_size=3, stride=1, padding=0, dilation=2, name="conv1"),
        LayerConfig(kernel_size=3, stride=1, padding=0, dilation=2, name="conv2"),
    ]
    rf = compute_receptive_field(layers)
    # RF = 1 + (5-1)*1 + (5-1)*1 = 1 + 4 + 4 = 9
    assert rf.size == (9, 9)


def test_asymmetric_kernels():
    """Test receptive field with asymmetric kernels."""
    # 3x5 kernel
    layer = LayerConfig(kernel_size=(3, 5), stride=1, padding=0)
    rf = compute_receptive_field([layer])
    assert rf.size == (3, 5)

    # Stack asymmetric kernels
    layers = [
        LayerConfig(kernel_size=(3, 5), stride=1, padding=0),
        LayerConfig(kernel_size=(5, 3), stride=1, padding=0),
    ]
    rf = compute_receptive_field(layers)
    assert rf.size == (7, 7)


def test_asymmetric_strides():
    """Test receptive field with asymmetric strides."""
    layers = [
        LayerConfig(kernel_size=3, stride=(2, 1), padding=0),
        LayerConfig(kernel_size=3, stride=(1, 2), padding=0),
    ]
    rf = compute_receptive_field(layers)
    # Height: 1 + (3-1)*1 + (3-1)*2 = 7
    # Width: 1 + (3-1)*1 + (3-1)*1 = 5
    assert rf.size == (7, 5)
    assert rf.stride == (2, 2)


def test_elementwise_operations():
    """Test that elementwise operations don't change receptive field."""
    layers = [
        LayerConfig(kernel_size=3, stride=1, padding=0, name="conv1"),
        create_layer_from_module("ReLU"),
        create_layer_from_module("BatchNorm2d"),
        LayerConfig(kernel_size=3, stride=1, padding=0, name="conv2"),
    ]
    rf = compute_receptive_field(layers)
    # Should be same as without ReLU and BatchNorm
    assert rf.size == (5, 5)


def test_create_layer_from_module():
    """Test helper function to create LayerConfig from module types."""
    # Elementwise operations
    relu = create_layer_from_module("ReLU")
    assert relu.kernel_size == 1
    assert relu.stride == 1

    bn = create_layer_from_module("BatchNorm2d")
    assert bn.kernel_size == 1
    assert bn.stride == 1

    # Convolutional layer
    conv = create_layer_from_module("Conv2d", kernel_size=3, stride=2, padding=1)
    assert conv.kernel_size == 3
    assert conv.stride == 2
    assert conv.padding == 1


def test_compute_output_position_in_input():
    """Test mapping output position to input region."""
    # Simple case: 3x3 kernel, stride 1, no padding
    layers = [LayerConfig(kernel_size=3, stride=1, padding=0)]
    rf_info = compute_receptive_field(layers)

    # First output pixel (0,0) maps to input region [0, 3)
    start, end = compute_output_position_in_input(0, rf_info, dim=0)
    assert start == 0
    assert end == 3

    # Second output pixel (1,0) maps to input region [1, 4)
    start, end = compute_output_position_in_input(1, rf_info, dim=0)
    assert start == 1
    assert end == 4

    # With stride 2
    layers = [LayerConfig(kernel_size=3, stride=2, padding=0)]
    rf_info = compute_receptive_field(layers)

    start, end = compute_output_position_in_input(0, rf_info, dim=0)
    assert start == 0
    assert end == 3

    start, end = compute_output_position_in_input(1, rf_info, dim=0)
    assert start == 2
    assert end == 5

    # With padding
    layers = [LayerConfig(kernel_size=3, stride=1, padding=1)]
    rf_info = compute_receptive_field(layers)

    start, end = compute_output_position_in_input(0, rf_info, dim=0)
    assert start == -1  # Starts in padding
    assert end == 2


def test_empty_layer_list():
    """Test receptive field with no layers."""
    rf = compute_receptive_field([])
    assert rf.size == 1
    assert rf.stride == 1
    assert rf.padding == 0


def test_receptive_field_info_str():
    """Test string representation of ReceptiveFieldInfo."""
    rf = ReceptiveFieldInfo(size=(51, 51), stride=(8, 8), padding=(2, 2))
    str_repr = str(rf)
    assert "51x51" in str_repr
    assert "stride=(8, 8)" in str_repr
    assert "padding=(2, 2)" in str_repr


def test_1d_convolution():
    """Test receptive field for 1D convolutions."""
    # Single 1D conv (explicit tuple with single element)
    layer = LayerConfig(kernel_size=(3,), stride=(1,), padding=(0,))
    rf = compute_receptive_field([layer])
    assert isinstance(rf.size, int)
    assert rf.size == 3

    # Multiple 1D convs
    layers = [
        LayerConfig(kernel_size=(3,), stride=(1,), padding=(0,)),
        LayerConfig(kernel_size=(3,), stride=(1,), padding=(0,)),
    ]
    rf = compute_receptive_field(layers)
    assert rf.size == 5


# Property-based tests


@given(
    kernel_size=st.integers(min_value=1, max_value=11),
    stride=st.integers(min_value=1, max_value=5),
    padding=st.integers(min_value=0, max_value=5),
)
def test_single_layer_receptive_field_equals_kernel(
    kernel_size: int, stride: int, padding: int
):
    """Property: Single layer receptive field size equals kernel size."""
    layer = LayerConfig(kernel_size=kernel_size, stride=stride, padding=padding)
    rf = compute_receptive_field([layer])
    assert rf.size == (kernel_size, kernel_size)


@given(
    num_layers=st.integers(min_value=1, max_value=10),
    kernel_size=st.integers(min_value=1, max_value=5),
)
def test_stacking_unit_stride_layers_grows_linearly(
    num_layers: int, kernel_size: int
):
    """Property: Stacking unit-stride layers grows RF linearly."""
    layers = [
        LayerConfig(kernel_size=kernel_size, stride=1, padding=0)
        for _ in range(num_layers)
    ]
    rf = compute_receptive_field(layers)

    # Expected: 1 + sum_{l=1}^{L} (k-1) = 1 + L*(k-1)
    expected_size = 1 + num_layers * (kernel_size - 1)
    assert rf.size == (expected_size, expected_size)


@given(strides=st.lists(st.integers(min_value=1, max_value=3), min_size=1, max_size=5))
def test_stride_product_is_positive(strides: list[int]):
    """Property: Stride product is always positive."""
    product = compute_stride_product(strides)
    assert product > 0


@given(
    dilation=st.integers(min_value=1, max_value=10),
    kernel_size=st.integers(min_value=1, max_value=10),
)
def test_dilation_increases_effective_kernel(dilation: int, kernel_size: int):
    """Property: Dilation increases or maintains effective kernel size."""
    eff_k = compute_effective_kernel_size(kernel_size, dilation)
    assert eff_k >= kernel_size


@given(
    num_layers=st.integers(min_value=1, max_value=5),
    stride=st.integers(min_value=1, max_value=3),
)
def test_adding_layers_increases_receptive_field(num_layers: int, stride: int):
    """Property: Adding more layers increases or maintains RF size."""
    rf_sizes = []
    layers = []

    for i in range(num_layers):
        layers.append(LayerConfig(kernel_size=3, stride=stride, padding=0))
        rf = compute_receptive_field(layers)
        rf_sizes.append(rf.size[0] if isinstance(rf.size, tuple) else rf.size)

    # RF should be monotonically increasing
    for i in range(1, len(rf_sizes)):
        assert rf_sizes[i] >= rf_sizes[i - 1]


def test_compute_receptive_field_size_1d():
    """Test 1D receptive field size computation directly."""
    layers = [
        LayerConfig(kernel_size=3, stride=1, padding=0),
        LayerConfig(kernel_size=3, stride=1, padding=0),
    ]
    size = compute_receptive_field_size_1d(layers, dim=0)
    assert size == 5


def test_compute_receptive_field_stride_1d():
    """Test 1D stride computation directly."""
    layers = [
        LayerConfig(kernel_size=3, stride=2, padding=0),
        LayerConfig(kernel_size=3, stride=2, padding=0),
    ]
    stride = compute_receptive_field_stride_1d(layers, dim=0)
    assert stride == 4


def test_compute_receptive_field_padding_1d():
    """Test 1D padding computation directly."""
    layers = [
        LayerConfig(kernel_size=3, stride=1, padding=1),
        LayerConfig(kernel_size=3, stride=2, padding=1),
    ]
    padding = compute_receptive_field_padding_1d(layers, dim=0)
    # padding = 1*1 + 1*1 = 2
    assert padding == 2

