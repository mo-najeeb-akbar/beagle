from collections import defaultdict
from functools import partial
from typing import Any

import numpy as np
import jax.numpy as jnp
from jax import image
import flax.linen as nn

ModuleDef = Any


def _up_blocks(
    self: Any, x: jnp.ndarray, steps_up: int, train: bool, name_prefix: str = "up"
) -> jnp.ndarray:
    """Upsample feature maps with bilinear interpolation."""
    norm = partial(self.norm, use_running_average=not train)
    for k in range(steps_up):
        batch, height, width, channels = x.shape
        x = image.resize(
            x,
            shape=(batch, height * 2, width * 2, channels),
            method="bilinear",
        )
        x = nn.Conv(
            features=channels,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
        )(x)
        x = norm()(x)
        if k < steps_up - 1:
            x = nn.relu(x)
    return x


def _down_blocks(
    self: Any, x: jnp.ndarray, steps_down: int, override_use_relu: bool, train: bool, name_prefix: str = "down"
) -> jnp.ndarray:
    """Downsample feature maps with strided convolutions."""
    norm = partial(self.norm, use_running_average=not train)
    for k in range(steps_down):
        batch, height, width, channels = x.shape
        x = nn.Conv(
            features=channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
        )(x)
        x = norm()(x)
        if (k < steps_down - 1) or override_use_relu:
            x = nn.relu(x)
    return x


def _resnet_block(self: Any, x: jnp.ndarray, train: bool, name_prefix: str = "resnet") -> jnp.ndarray:
    """Residual block with normalization."""
    norm = partial(self.norm, use_running_average=not train)
    batch, height, width, channels = x.shape
    h = nn.Conv(
        features=channels,
        kernel_size=(3, 3),
        padding="SAME",
        use_bias=False,
    )(x)
    h = nn.relu(norm()(h))
    h = nn.Conv(
        features=channels,
        kernel_size=(3, 3),
        padding="SAME",
        use_bias=False,
    )(h)
    h = norm()(h)
    return x + h


def _down_with_skip(
    self: Any, x: jnp.ndarray, features: int, train: bool, name_prefix: str = "down_skip"
) -> jnp.ndarray:
    """Downsample with skip connection."""
    norm = partial(self.norm, use_running_average=not train)
    h = nn.Conv(
        features=features,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="SAME",
        use_bias=False,
    )(x)
    h = norm()(h)
    x = nn.Conv(
        features=features,
        kernel_size=(1, 1),
        strides=(2, 2),
        padding="SAME",
        use_bias=False,
    )(x)
    return nn.relu(x + h)


def _down_with_skip_beginning(
    self: Any, x: jnp.ndarray, features: int, train: bool, name_prefix: str = "stem"
) -> jnp.ndarray:
    """Initial downsampling with skip connection."""
    norm = partial(self.norm, use_running_average=not train)
    h = nn.Conv(
        features=features,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="SAME",
        use_bias=False,
    )(x)
    h = norm()(h)
    x = nn.Conv(
        features=features,
        kernel_size=(1, 1),
        strides=(2, 2),
        padding="SAME",
        use_bias=False,
    )(x)
    return nn.relu(x + h)


def _fuse_blocks(
    self: Any,
    blocks: list[jnp.ndarray],
    channels: int,
    use_concatenation: bool,
    train: bool,
    name_prefix: str = "fuse",
) -> jnp.ndarray:
    """Fuse multiple resolution blocks together."""
    norm = partial(self.norm, use_running_average=not train)

    if use_concatenation:
        fused = jnp.concatenate(blocks, axis=-1)
    else:
        fused = sum(blocks)
    fused = nn.relu(fused)
    fused = nn.Conv(
        features=channels,
        kernel_size=(3, 3),
        padding="SAME",
        use_bias=False,
    )(fused)
    fused = norm()(fused)
    fused = nn.relu(fused)
    return fused


def _basic_block(
    self: Any,
    x: jnp.ndarray,
    channels: int,
    train: bool,
    kernel_size: int = 3,
    name_prefix: str = "basic",
) -> jnp.ndarray:
    """Basic conv-norm-relu block."""
    norm = partial(self.norm, use_running_average=not train)
    x = nn.Conv(
        features=channels,
        kernel_size=(kernel_size, kernel_size),
        padding="SAME",
        use_bias=False,
    )(x)
    x = norm()(x)
    x = nn.relu(x)
    return x


class HRNetBackbone(nn.Module):
    """High-Resolution Network backbone for multi-scale feature extraction.

    Processes input through multiple stages of parallel multi-resolution blocks,
    maintaining high-resolution representations throughout the network.

    Attributes:
        num_stages: Number of HRNet stages (default: 3)
        features: Base number of features (default: 32)
        target_res: Target output resolution as fraction of input (default: 1.0)
                   1.0 = same resolution, 0.5 = half resolution, etc.

    Returns:
        Dict with keys:
            - 'features': Multi-scale features at target_res [B, H', W', C]

    Example:
        >>> backbone = HRNetBackbone(num_stages=3, features=32, target_res=1.0)
        >>> outputs = backbone(images, train=True)
        >>> features = outputs['features']  # [B, H, W, C]
    """

    num_stages: int
    features: int
    target_res: float

    def setup(self) -> None:
        self.norm = partial(
            nn.BatchNorm, momentum=0.9, epsilon=1e-5, dtype=jnp.float32
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> dict[str, jnp.ndarray]:
        """Process input through multi-resolution stages.

        Args:
            x: Input image [B, H, W, C]
            train: Training mode (affects batch normalization)

        Returns:
            dict with 'features' key containing fused multi-scale features
        """
        # Initial stem: downsample 4x total (2x + 2x)
        x = _down_with_skip_beginning(self, x, self.features * 2, train, name_prefix="stem_0")
        x = _down_with_skip_beginning(self, x, self.features, train, name_prefix="stem_1")

        # Multi-resolution parallel processing
        batch, height, width, channels = x.shape
        blocks = [x]
        block_sizes = [1]

        for stage in range(self.num_stages):
            resolution_groups = defaultdict(list)
            new_blocks = []
            new_block_sizes = []
            lowest_res = 1 / (2 ** (stage + 1))

            for block_idx, (block, block_size) in enumerate(zip(blocks, block_sizes)):
                num_steps_up = int(np.abs(np.log2(block_size)))
                num_steps_down = int(np.abs(np.log2(lowest_res / block_size)))

                # Upsample branches
                for step in range(num_steps_up):
                    current_block = block
                    curr_res = block_size * (2 ** (step + 1))
                    name = f"stage_{stage}_block_{block_idx}_up_res_{curr_res}"
                    if stage == (self.num_stages - 1):
                        if self.target_res == curr_res:
                            current_block = _up_blocks(
                                self, current_block, step + 1, train, name_prefix=name
                            )
                            resolution_groups[curr_res].append(current_block)
                    else:
                        current_block = _up_blocks(self, current_block, step + 1, train, name_prefix=name)
                        resolution_groups[curr_res].append(current_block)

                # Downsample branches
                for step in range(num_steps_down):
                    current_block = block
                    curr_res = block_size * (1 / (2 ** (step + 1)))
                    use_relu_last_ds = step == (num_steps_down - 1)
                    name = f"stage_{stage}_block_{block_idx}_down_res_{curr_res}"
                    if stage == (self.num_stages - 1):
                        if self.target_res == curr_res:
                            current_block = _down_blocks(
                                self, current_block, step + 1, use_relu_last_ds, train, name_prefix=name
                            )
                            resolution_groups[curr_res].append(current_block)
                    else:
                        current_block = _down_blocks(
                            self, current_block, step + 1, use_relu_last_ds, train, name_prefix=name
                        )
                        resolution_groups[curr_res].append(current_block)

                # Residual refinement
                name = f"stage_{stage}_block_{block_idx}_resnet_res_{block_size}"
                if stage == (self.num_stages - 1):
                    if self.target_res == block_size:
                        current_block = block
                        current_block = _resnet_block(self, current_block, train, name_prefix=name)
                        resolution_groups[block_size].append(current_block)
                else:
                    current_block = block
                    current_block = _resnet_block(self, current_block, train, name_prefix=name)
                    resolution_groups[block_size].append(current_block)

            # Fuse multi-resolution features
            for res_idx, (resolution, resolution_blocks) in enumerate(resolution_groups.items()):
                if len(resolution_blocks) == 1:
                    new_blocks.append(resolution_blocks[0])
                    new_block_sizes.append(resolution)
                else:
                    use_concat_last_step = stage == (self.num_stages - 1)
                    name = f"stage_{stage}_fuse_res_{resolution}_idx_{res_idx}"
                    fused_block = _fuse_blocks(
                        self, resolution_blocks, channels, use_concat_last_step, train, name_prefix=name
                    )
                    new_blocks.append(fused_block)
                    new_block_sizes.append(resolution)

            blocks = new_blocks
            block_sizes = new_block_sizes

        backbone_out = blocks[-1]

        # Return dict instead of raw tensor
        return {'features': backbone_out}


class SegmentationHead(nn.Module):
    """Segmentation decoder head with optional upsampling.

    Takes features from a backbone and decodes to per-pixel class predictions.
    Supports configurable upsampling and activation functions.

    Attributes:
        num_classes: Number of segmentation classes
        features: Number of intermediate features (default: 32)
        upsample_steps: Number of 2x upsampling steps (default: 0)
        use_sigmoid: Apply sigmoid activation instead of softmax (default: False)
        output_key: Name of output dict key (default: 'logits')

    Returns:
        Dict with single key (specified by output_key parameter):
            - <output_key>: Segmentation logits/probabilities [B, H', W', num_classes]

    Example:
        >>> head = SegmentationHead(num_classes=3, upsample_steps=2, output_key='logits')
        >>> outputs = head(backbone_features, train=True)
        >>> logits = outputs['logits']  # [B, H*4, W*4, 3]
    """

    num_classes: int
    features: int = 32
    upsample_steps: int = 0
    use_sigmoid: bool = False
    output_key: str = 'logits'

    def setup(self) -> None:
        self.norm = partial(
            nn.BatchNorm, momentum=0.9, epsilon=1e-5, dtype=jnp.float32
        )

    @nn.compact
    def __call__(self, features: jnp.ndarray, train: bool = True) -> dict[str, jnp.ndarray]:
        """Decode features to segmentation output.

        Args:
            features: Input features from backbone [B, H, W, C]
            train: Training mode (affects batch normalization)

        Returns:
            dict with output_key -> logits [B, H', W', num_classes]
        """
        # Decoder blocks: refine features for segmentation
        x = _basic_block(self, features, self.features, train, name_prefix="decode_0")
        x = _basic_block(self, x, self.features, train, name_prefix="decode_1")
        x = _basic_block(self, x, self.num_classes, train, kernel_size=1, name_prefix="decode_2")

        # Optional upsampling to reach target resolution
        if self.upsample_steps > 0:
            x = _up_blocks(self, x, self.upsample_steps, train, name_prefix="upsample")

        # Final 1x1 conv for class logits
        output = nn.Conv(features=self.num_classes, kernel_size=(1, 1), use_bias=True)(x)

        # Optional activation (sigmoid for binary/multi-label)
        if self.use_sigmoid:
            output = nn.sigmoid(output)

        # Return dict with configurable key name
        return {self.output_key: output}


class EmbedNet(nn.Module):
    """Embedding network with configurable backbone."""

    output_dim: int
    num_features: int
    train_bb: bool
    backbone: Any

    def setup(self) -> None:
        self.norm = partial(
            nn.BatchNorm, momentum=0.9, epsilon=1e-5, dtype=jnp.float32
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Extract normalized embedding from input."""
        norm = partial(self.norm, use_running_average=not train)

        x = self.backbone(x, (train and self.train_bb))
        for i in range(5):
            x = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                use_bias=False,
            )(x)
            x = norm()(x)

        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        x = nn.Dense(features=self.output_dim, use_bias=True)(x)
        x = x / (jnp.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        return x
