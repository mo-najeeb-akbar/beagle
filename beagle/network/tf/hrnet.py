from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow import keras


def _fuse_blocks(
    blocks: list[tf.Tensor], channels: int, use_concatenation: bool
) -> tf.Tensor:
    """Fuse multiple resolution blocks together."""
    if use_concatenation:
        fused = keras.layers.Concatenate(axis=-1)(blocks)
    else:
        fused = keras.layers.Add()(blocks)
    fused = keras.layers.Activation("relu")(fused)
    fused = keras.layers.Conv2D(
        filters=channels, kernel_size=3, padding="same", use_bias=False
    )(fused)
    fused = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(fused)
    return keras.layers.Activation("relu")(fused)


def _upsample_blocks(x: tf.Tensor, steps_up: int) -> tf.Tensor:
    """Upsample feature maps with bilinear interpolation."""
    for k in range(steps_up):
        channels = x.shape[-1]
        x = keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        x = keras.layers.Conv2D(
            filters=channels, kernel_size=3, padding="same", use_bias=False
        )(x)
        x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        if k < steps_up - 1:
            x = keras.layers.Activation("relu")(x)
    return x


def _downsample_blocks(
    x: tf.Tensor, steps_down: int, override_use_relu: bool
) -> tf.Tensor:
    """Downsample feature maps with strided convolutions."""
    for k in range(steps_down):
        channels = x.shape[-1]
        x = keras.layers.Conv2D(
            filters=channels, kernel_size=3, strides=2, padding="same", use_bias=False
        )(x)
        x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        if (k < steps_down - 1) or override_use_relu:
            x = keras.layers.Activation("relu")(x)
    return x


def _down_with_skip(x: tf.Tensor, features: int) -> tf.Tensor:
    """Downsample with skip connection."""
    h = keras.layers.Conv2D(
        filters=features, kernel_size=3, strides=2, padding="same", use_bias=False
    )(x)
    h = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(h)
    x = keras.layers.Conv2D(
        filters=features, kernel_size=1, strides=2, padding="same", use_bias=False
    )(x)
    return keras.layers.Activation("relu")(x + h)


def _down_with_skip_beginning(x: tf.Tensor, features: int) -> tf.Tensor:
    """Initial downsampling with skip connection."""
    h = keras.layers.Conv2D(
        filters=features, kernel_size=3, strides=2, padding="same", use_bias=False
    )(x)
    h = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(h)
    x = keras.layers.Conv2D(
        filters=features, kernel_size=1, strides=2, padding="same", use_bias=False
    )(x)
    return keras.layers.Activation("relu")(x + h)


def _resnet_block(x: tf.Tensor) -> tf.Tensor:
    """Residual block with normalization."""
    channels = x.shape[-1]
    h = keras.layers.Conv2D(
        filters=channels, kernel_size=3, padding="same", use_bias=False
    )(x)
    h = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(h)
    h = keras.layers.Activation("relu")(h)
    h = keras.layers.Conv2D(
        filters=channels, kernel_size=3, padding="same", use_bias=False
    )(h)
    h = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(h)
    return x + h


def _basic_block(x: tf.Tensor, channels: int, kernel_size: int = 3) -> tf.Tensor:
    """Basic conv-norm-relu block."""
    x = keras.layers.Conv2D(
        filters=channels, kernel_size=kernel_size, padding="same", use_bias=False
    )(x)
    x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    return keras.layers.Activation("relu")(x)


def build_hrnet_backbone(
    x: tf.Tensor, num_stages: int, features: int, target_res: float
) -> tf.Tensor:
    """Build HRNet backbone with multi-resolution processing.

    Args:
        x: Input tensor
        num_stages: Number of multi-resolution stages
        features: Base number of features
        target_res: Target resolution scale factor

    Returns:
        Backbone output tensor at target resolution
    """
    x = _down_with_skip_beginning(x, features * 2)
    x = _down_with_skip_beginning(x, features)

    channels = x.shape[-1]
    blocks = [x]
    block_sizes = [1]

    for stage in range(num_stages):
        resolution_groups = defaultdict(list)
        new_blocks = []
        new_block_sizes = []
        lowest_res = 1 / (2 ** (stage + 1))

        for block_idx, (block, block_size) in enumerate(zip(blocks, block_sizes)):
            num_steps_up = int(np.abs(np.log2(block_size)))
            num_steps_down = int(np.abs(np.log2(lowest_res / block_size)))

            for step in range(num_steps_up):
                current_block = block
                curr_res = block_size * (2 ** (step + 1))
                if stage == (num_stages - 1):
                    if target_res == curr_res:
                        current_block = _upsample_blocks(current_block, step + 1)
                        resolution_groups[curr_res].append(current_block)
                else:
                    current_block = _upsample_blocks(current_block, step + 1)
                    resolution_groups[curr_res].append(current_block)

            for step in range(num_steps_down):
                current_block = block
                curr_res = block_size * (1 / (2 ** (step + 1)))
                use_relu_last_ds = step == (num_steps_down - 1)
                if stage == (num_stages - 1):
                    if target_res == curr_res:
                        current_block = _downsample_blocks(
                            current_block, step + 1, use_relu_last_ds
                        )
                        resolution_groups[curr_res].append(current_block)
                else:
                    current_block = _downsample_blocks(
                        current_block, step + 1, use_relu_last_ds
                    )
                    resolution_groups[curr_res].append(current_block)

            if stage == (num_stages - 1):
                if target_res == block_size:
                    current_block = block
                    current_block = _resnet_block(current_block)
                    resolution_groups[block_size].append(current_block)
            else:
                current_block = block
                current_block = _resnet_block(current_block)
                resolution_groups[block_size].append(current_block)

        for res_idx, (resolution, resolution_blocks) in enumerate(resolution_groups.items()):
            if len(resolution_blocks) == 1:
                new_blocks.append(resolution_blocks[0])
                new_block_sizes.append(resolution)
            else:
                use_concat_last_step = stage == (num_stages - 1)
                fused_block = _fuse_blocks(
                    resolution_blocks, channels, use_concat_last_step
                )
                new_blocks.append(fused_block)
                new_block_sizes.append(resolution)

        blocks = new_blocks
        block_sizes = new_block_sizes

    return blocks[-1]


def build_hrnet_monet(
    x: tf.Tensor,
    num_stages: int,
    features: int,
    target_res: float,
    outputs: list[tuple[int, bool] | tuple[int, bool, int]],
) -> list[tf.Tensor]:
    """Build multi-output network based on HRNet backbone.

    Args:
        x: Input tensor
        num_stages: Number of multi-resolution stages
        features: Base number of features
        target_res: Target resolution scale factor
        outputs: List of output descriptors. Each descriptor is either:
            - (num_outputs, use_sigmoid) for no upsampling
            - (num_outputs, use_sigmoid, upsample_steps) for upsampling

    Returns:
        List of output tensors (one per head) plus backbone output as last element
    """
    backbone_out = build_hrnet_backbone(x, num_stages, features, target_res)

    outputs_tf = []
    for head_idx, descriptor in enumerate(outputs):
        if len(descriptor) == 2:
            num_outs, use_sigmoid = descriptor
            upsample_steps = 0
        else:
            num_outs, use_sigmoid, upsample_steps = descriptor

        head = _basic_block(backbone_out, features)
        head = _basic_block(head, features)
        head = _basic_block(head, num_outs, kernel_size=1)
        if upsample_steps > 0:
            head = _upsample_blocks(head, upsample_steps)
        head = keras.layers.Conv2D(filters=num_outs, kernel_size=1, use_bias=True)(head)
        if use_sigmoid:
            head = keras.layers.Activation("sigmoid")(head)
        outputs_tf.append(head)

    outputs_tf.append(backbone_out)
    return outputs_tf


class HRNetBB(keras.Model):
    """High-Resolution Network backbone."""

    def __init__(
        self, num_stages: int, features: int, target_res: float, name: str | None = None
    ) -> None:
        super().__init__(name=name)
        self.num_stages = num_stages
        self.features = features
        self.target_res = target_res

    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Process input through multi-resolution stages."""
        return build_hrnet_backbone(x, self.num_stages, self.features, self.target_res)


class MoNet(keras.Model):
    """Multi-output network based on HRNet backbone.
    
    Supports multiple output heads with optional upsampling.
    Output descriptor format: (num_outputs, use_sigmoid, upsample_steps)
    where upsample_steps is optional (defaults to 0).
    """

    def __init__(
        self,
        num_stages: int,
        features: int,
        target_res: float,
        train_bb: bool,
        outputs: list[tuple[int, bool] | tuple[int, bool, int]],
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.num_stages = num_stages
        self.features = features
        self.target_res = target_res
        self.train_bb = train_bb
        self.output_descriptors = outputs
        self.backbone = HRNetBB(num_stages, features, target_res)

    def call(self, x: tf.Tensor, training: bool = True) -> list[tf.Tensor]:
        """Process input through backbone and multiple output heads.

        Returns:
            List of output tensors (one per head) plus backbone output as last element
        """
        backbone_out = self.backbone(x, training=(training and self.train_bb))

        outputs = []
        for head_idx, descriptor in enumerate(self.output_descriptors):
            if len(descriptor) == 2:
                num_outs, use_sigmoid = descriptor
                upsample_steps = 0
            else:
                num_outs, use_sigmoid, upsample_steps = descriptor

            head = _basic_block(backbone_out, self.features)
            head = _basic_block(head, self.features)
            head = _basic_block(head, num_outs, kernel_size=1)
            if upsample_steps > 0:
                head = _upsample_blocks(head, upsample_steps)
            out = keras.layers.Conv2D(filters=num_outs, kernel_size=1, use_bias=True)(head)

            if use_sigmoid:
                out = keras.layers.Activation("sigmoid")(out)
            outputs.append(out)

        outputs.append(backbone_out)
        return outputs

