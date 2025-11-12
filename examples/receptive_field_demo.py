"""Demonstration of receptive field computation for CNNs.

This script demonstrates how to compute receptive fields for various
convolutional neural network architectures using the beagle library.

Based on: https://distill.pub/2019/computing-receptive-fields/
"""

from beagle.network.receptive_field import (
    LayerConfig,
    compute_output_position_in_input,
    compute_receptive_field,
    create_layer_from_module,
    print_receptive_field_report,
)


def alexnet_example() -> None:
    """Demonstrate receptive field computation for AlexNet-style architecture."""
    print("\n" + "=" * 80)
    print("ALEXNET-STYLE ARCHITECTURE")
    print("=" * 80)

    layers = [
        LayerConfig(kernel_size=11, stride=4, padding=2, name="conv1"),
        LayerConfig(kernel_size=3, stride=2, padding=0, name="pool1"),
        LayerConfig(kernel_size=5, stride=1, padding=2, name="conv2"),
        LayerConfig(kernel_size=3, stride=2, padding=0, name="pool2"),
        LayerConfig(kernel_size=3, stride=1, padding=1, name="conv3"),
        LayerConfig(kernel_size=3, stride=1, padding=1, name="conv4"),
        LayerConfig(kernel_size=3, stride=1, padding=1, name="conv5"),
        LayerConfig(kernel_size=3, stride=2, padding=0, name="pool5"),
    ]

    print_receptive_field_report(layers)

    # Show input region for specific output positions
    rf_info = compute_receptive_field(layers)
    print("\nMapping output positions to input regions:")
    print("-" * 70)
    for out_pos in [0, 1, 2, 5, 10]:
        start, end = compute_output_position_in_input(out_pos, rf_info, dim=0)
        print(
            f"Output position ({out_pos}, 0) maps to input region "
            f"[{start}:{end}, :] (height dimension)"
        )


def vgg_example() -> None:
    """Demonstrate receptive field computation for VGG-style architecture."""
    print("\n" + "=" * 80)
    print("VGG-STYLE ARCHITECTURE (2 blocks)")
    print("=" * 80)

    layers = [
        # Block 1
        LayerConfig(kernel_size=3, stride=1, padding=1, name="conv1_1"),
        create_layer_from_module("ReLU", name="relu1_1"),
        LayerConfig(kernel_size=3, stride=1, padding=1, name="conv1_2"),
        create_layer_from_module("ReLU", name="relu1_2"),
        LayerConfig(kernel_size=2, stride=2, padding=0, name="pool1"),
        # Block 2
        LayerConfig(kernel_size=3, stride=1, padding=1, name="conv2_1"),
        create_layer_from_module("ReLU", name="relu2_1"),
        LayerConfig(kernel_size=3, stride=1, padding=1, name="conv2_2"),
        create_layer_from_module("ReLU", name="relu2_2"),
        LayerConfig(kernel_size=2, stride=2, padding=0, name="pool2"),
    ]

    print_receptive_field_report(layers)


def resnet_block_example() -> None:
    """Demonstrate receptive field for a ResNet-style block (main path only).

    Note: This computes the receptive field for the main convolution path.
    For multi-path networks (including skip connections), receptive field
    computation is more complex and depends on the specific merge operation.
    """
    print("\n" + "=" * 80)
    print("RESNET-STYLE BLOCK (main path only)")
    print("=" * 80)

    # ResNet bottleneck block (main path)
    layers = [
        # Bottleneck: 1x1 -> 3x3 -> 1x1
        LayerConfig(kernel_size=1, stride=1, padding=0, name="conv1_1x1"),
        create_layer_from_module("BatchNorm2d", name="bn1"),
        create_layer_from_module("ReLU", name="relu1"),
        LayerConfig(kernel_size=3, stride=1, padding=1, name="conv2_3x3"),
        create_layer_from_module("BatchNorm2d", name="bn2"),
        create_layer_from_module("ReLU", name="relu2"),
        LayerConfig(kernel_size=1, stride=1, padding=0, name="conv3_1x1"),
        create_layer_from_module("BatchNorm2d", name="bn3"),
    ]

    print_receptive_field_report(layers)


def dilated_conv_example() -> None:
    """Demonstrate receptive field with dilated (atrous) convolutions."""
    print("\n" + "=" * 80)
    print("DILATED CONVOLUTION EXAMPLE")
    print("=" * 80)

    layers = [
        LayerConfig(kernel_size=3, stride=1, padding=0, dilation=1, name="conv1_d1"),
        LayerConfig(kernel_size=3, stride=1, padding=0, dilation=2, name="conv2_d2"),
        LayerConfig(kernel_size=3, stride=1, padding=0, dilation=4, name="conv3_d4"),
        LayerConfig(kernel_size=3, stride=1, padding=0, dilation=8, name="conv4_d8"),
    ]

    print_receptive_field_report(layers)

    rf_info = compute_receptive_field(layers)
    print(
        "\nNote: Dilated convolutions achieve large receptive fields "
        "without downsampling."
    )
    print(
        f"This 4-layer network has RF={rf_info.size[0]}x{rf_info.size[1]} "
        f"but stride={rf_info.stride[0]} (no downsampling)."
    )


def mobilenet_example() -> None:
    """Demonstrate receptive field for MobileNet-style architecture."""
    print("\n" + "=" * 80)
    print("MOBILENET-STYLE ARCHITECTURE (first few layers)")
    print("=" * 80)

    layers = [
        # Standard conv
        LayerConfig(kernel_size=3, stride=2, padding=1, name="conv1"),
        create_layer_from_module("BatchNorm2d", name="bn1"),
        create_layer_from_module("ReLU", name="relu1"),
        # Depthwise separable conv
        LayerConfig(kernel_size=3, stride=1, padding=1, name="conv2_dw"),
        create_layer_from_module("BatchNorm2d", name="bn2"),
        create_layer_from_module("ReLU", name="relu2"),
        LayerConfig(kernel_size=1, stride=1, padding=0, name="conv2_pw"),
        create_layer_from_module("BatchNorm2d", name="bn3"),
        create_layer_from_module("ReLU", name="relu3"),
        # Another depthwise separable with stride
        LayerConfig(kernel_size=3, stride=2, padding=1, name="conv3_dw"),
        create_layer_from_module("BatchNorm2d", name="bn4"),
        create_layer_from_module("ReLU", name="relu4"),
        LayerConfig(kernel_size=1, stride=1, padding=0, name="conv3_pw"),
    ]

    print_receptive_field_report(layers)


def custom_network_example() -> None:
    """Show how to analyze a custom network."""
    print("\n" + "=" * 80)
    print("CUSTOM NETWORK EXAMPLE")
    print("=" * 80)

    # Define your network architecture
    layers = [
        LayerConfig(kernel_size=7, stride=2, padding=3, name="stem_conv"),
        create_layer_from_module("ReLU"),
        LayerConfig(kernel_size=3, stride=2, padding=1, name="conv1"),
        create_layer_from_module("ReLU"),
        LayerConfig(kernel_size=3, stride=2, padding=1, name="conv2"),
        create_layer_from_module("ReLU"),
        LayerConfig(kernel_size=3, stride=1, padding=1, name="conv3"),
    ]

    rf_info = compute_receptive_field(layers)

    print(f"\nNetwork has {len(layers)} layers")
    print(f"Final receptive field: {rf_info.size[0]}x{rf_info.size[1]} pixels")
    print(f"Effective stride: {rf_info.stride[0]}x{rf_info.stride[1]}")
    print(f"Effective padding: {rf_info.padding[0]}, {rf_info.padding[1]}")

    # Calculate how much the input image is downsampled
    downsample_factor = rf_info.stride[0]
    print(f"\nInput image downsampling factor: {downsample_factor}x")
    print(
        f"Example: 224x224 input -> "
        f"{224 // downsample_factor}x{224 // downsample_factor} output"
    )


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RECEPTIVE FIELD COMPUTATION FOR CONVOLUTIONAL NEURAL NETWORKS")
    print("Based on: https://distill.pub/2019/computing-receptive-fields/")
    print("=" * 80)

    # Run all examples
    alexnet_example()
    vgg_example()
    resnet_block_example()
    dilated_conv_example()
    mobilenet_example()
    custom_network_example()

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80 + "\n")

