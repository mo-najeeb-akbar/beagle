# Receptive Field Computation

This guide explains how to use the receptive field computation utilities in the `beagle.network` module.

## Overview

The receptive field of a convolutional neural network is the size of the region in the input that produces a feature in the output. Understanding receptive fields is crucial for:

- Designing networks for specific tasks (e.g., small vs large objects)
- Debugging network architectures
- Understanding feature scales at different layers
- Ensuring your network can "see" objects at the right scale

This implementation is based on the paper [**"Computing Receptive Fields of Convolutional Neural Networks"**](https://distill.pub/2019/computing-receptive-fields/) by Araujo, Norris, and Sim (2019).

## Quick Start

```python
from beagle.network import LayerConfig, compute_receptive_field

# Define your network architecture
layers = [
    LayerConfig(kernel_size=7, stride=2, padding=3, name="conv1"),
    LayerConfig(kernel_size=3, stride=2, padding=1, name="pool1"),
    LayerConfig(kernel_size=3, stride=1, padding=1, name="conv2"),
]

# Compute receptive field
rf_info = compute_receptive_field(layers)

print(f"Receptive field size: {rf_info.size}")      # e.g., (17, 17)
print(f"Effective stride: {rf_info.stride}")        # e.g., (4, 4)
print(f"Effective padding: {rf_info.padding}")      # e.g., (7, 7)
```

## Core API

### LayerConfig

`LayerConfig` represents the configuration of a single layer:

```python
from beagle.network import LayerConfig

layer = LayerConfig(
    kernel_size=3,          # int or tuple[int, int]
    stride=1,               # int or tuple[int, int]
    padding=1,              # int or tuple[int, int]
    dilation=1,             # int or tuple[int, int] (for dilated convs)
    name="conv1"            # optional name for debugging
)
```

**Parameters:**
- `kernel_size`: Spatial extent of the kernel
- `stride`: Stride of the operation
- `padding`: Padding applied (can be scalar, tuple, or 2D tuple for asymmetric padding)
- `dilation`: Dilation factor for atrous convolutions (default: 1)
- `name`: Optional layer name for debugging

### compute_receptive_field

Computes the receptive field for a sequence of layers:

```python
from beagle.network import compute_receptive_field, ReceptiveFieldInfo

rf_info: ReceptiveFieldInfo = compute_receptive_field(layers)
```

Returns a `ReceptiveFieldInfo` object with:
- `size`: Receptive field size (int for 1D, tuple for 2D)
- `stride`: Effective stride from input to output
- `padding`: Effective padding from input

### create_layer_from_module

Helper to create `LayerConfig` for common layer types:

```python
from beagle.network import create_layer_from_module

# Elementwise operations (no spatial change)
relu = create_layer_from_module("ReLU")
bn = create_layer_from_module("BatchNorm2d")

# Convolutional layers
conv = create_layer_from_module("Conv2d", kernel_size=3, stride=2, padding=1)
```

Supported module types:
- Elementwise: `ReLU`, `LeakyReLU`, `GELU`, `SiLU`, `BatchNorm2d`, `LayerNorm`, `Dropout`, `Identity`
- Others: specify parameters explicitly

### print_receptive_field_report

Prints a detailed analysis of receptive field growth:

```python
from beagle.network import print_receptive_field_report

print_receptive_field_report(layers)
```

Output example:
```
======================================================================
RECEPTIVE FIELD ANALYSIS
======================================================================

Final Receptive Field: Receptive Field: 51x51 pixels, stride=(8, 8), padding=(2, 2)

Number of layers: 3

----------------------------------------------------------------------
Layer-by-layer receptive field growth:
----------------------------------------------------------------------
Layer                Type            k        s        RF Size        
----------------------------------------------------------------------
conv1                Convolution     11       4        11x11          
pool1                Pooling         3        2        19x19          
conv2                Convolution     5        1        51x51          
======================================================================
```

## Common Use Cases

### 1. AlexNet-Style Architecture

```python
from beagle.network import LayerConfig, compute_receptive_field

layers = [
    LayerConfig(kernel_size=11, stride=4, padding=2, name="conv1"),
    LayerConfig(kernel_size=3, stride=2, padding=0, name="pool1"),
    LayerConfig(kernel_size=5, stride=1, padding=2, name="conv2"),
]

rf_info = compute_receptive_field(layers)
# Result: 51x51 receptive field with 8x8 stride
```

### 2. VGG-Style Architecture

```python
from beagle.network import LayerConfig, create_layer_from_module

layers = [
    LayerConfig(kernel_size=3, stride=1, padding=1, name="conv1_1"),
    create_layer_from_module("ReLU"),
    LayerConfig(kernel_size=3, stride=1, padding=1, name="conv1_2"),
    create_layer_from_module("ReLU"),
    LayerConfig(kernel_size=2, stride=2, padding=0, name="pool1"),
]

rf_info = compute_receptive_field(layers)
# Result: 6x6 receptive field
```

### 3. Dilated (Atrous) Convolutions

```python
layers = [
    LayerConfig(kernel_size=3, stride=1, padding=0, dilation=1),
    LayerConfig(kernel_size=3, stride=1, padding=0, dilation=2),
    LayerConfig(kernel_size=3, stride=1, padding=0, dilation=4),
    LayerConfig(kernel_size=3, stride=1, padding=0, dilation=8),
]

rf_info = compute_receptive_field(layers)
# Result: 31x31 receptive field with stride=1 (no downsampling!)
```

### 4. Mapping Output Position to Input Region

```python
from beagle.network import compute_output_position_in_input

# Get input region for output position (5, 3)
start_h, end_h = compute_output_position_in_input(5, rf_info, dim=0)  # height
start_w, end_w = compute_output_position_in_input(3, rf_info, dim=1)  # width

print(f"Output (5, 3) maps to input region [{start_h}:{end_h}, {start_w}:{end_w}]")
```

## Advanced Features

### 1D Convolutions

For 1D convolutions, use explicit 1-element tuples:

```python
# 1D convolution
layer = LayerConfig(kernel_size=(3,), stride=(1,), padding=(0,))
rf_info = compute_receptive_field([layer])
# rf_info.size is an int (not tuple) for 1D
```

### Asymmetric Kernels and Strides

```python
# Asymmetric kernel: 3x5
layer = LayerConfig(kernel_size=(3, 5), stride=1, padding=0)

# Asymmetric strides: 2x1
layer = LayerConfig(kernel_size=3, stride=(2, 1), padding=0)

# Asymmetric padding: top/bottom = 1, left/right = 2
layer = LayerConfig(kernel_size=3, stride=1, padding=((1, 1), (2, 2)))
```

## Examples

See `examples/receptive_field_demo.py` for comprehensive examples including:

- AlexNet-style architecture
- VGG-style architecture  
- ResNet-style blocks
- Dilated convolutions
- MobileNet-style architecture
- Custom networks

Run the demo:

```bash
make run CMD='python examples/receptive_field_demo.py'
```

## Mathematical Background

The receptive field size is computed using the closed-form formula:

```
r_0 = sum_{l=1}^{L} ((k_l - 1) * prod_{i=1}^{l-1} s_i) + 1
```

Where:
- `k_l` is the kernel size at layer `l` (accounting for dilation)
- `s_i` is the stride at layer `i`
- `r_0` is the receptive field size at the input

For more details, see the [Distill article](https://distill.pub/2019/computing-receptive-fields/).

## Limitations

### Multi-Path Networks

This implementation computes receptive fields for **single-path networks** (e.g., AlexNet, VGG, sequential blocks). 

For multi-path networks with skip connections (e.g., ResNet, DenseNet, Inception), you can:
1. Compute the receptive field for each path separately
2. The effective receptive field depends on the merge operation (add, concatenate)
3. For addition: RF is typically the maximum of the paths
4. For concatenation: both paths contribute to the output

### Training vs. Inference

The computed receptive field is the **theoretical receptive field**. The **effective receptive field** (which pixels actually influence the output) may be smaller, especially for:

- Networks initialized with small weights
- Networks with many non-linearities
- Deep networks where gradients fade

See [Luo et al. 2016](https://arxiv.org/abs/1701.04128) for more on effective receptive fields.

## Troubleshooting

### "kernel_size must be positive"

Ensure all kernel sizes, strides are positive integers:

```python
# Bad
LayerConfig(kernel_size=0, stride=1)

# Good
LayerConfig(kernel_size=1, stride=1)
```

### Asymmetric padding not working

For asymmetric padding, use explicit tuple format:

```python
# 2D padding: ((top, bottom), (left, right))
layer = LayerConfig(
    kernel_size=3, 
    stride=1, 
    padding=((1, 2), (1, 2))  # top=1, bottom=2, left=1, right=2
)
```

### Very large receptive fields

If your receptive field is unexpectedly large, check:
1. Stride values (strides compound multiplicatively)
2. Dilation values (dilation increases effective kernel size)
3. Number of layers (each layer adds to RF)

## Reference

Araujo, A., Norris, W., & Sim, J. (2019). Computing Receptive Fields of Convolutional Neural Networks. *Distill*, 4(11), e21. https://doi.org/10.23915/distill.00021

