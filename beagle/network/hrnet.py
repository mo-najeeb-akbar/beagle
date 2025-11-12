from collections import defaultdict
from functools import partial
from typing import Any

import numpy as np
import jax.numpy as jnp
from jax import image
import flax.linen as nn

ModuleDef = Any


def _up_blocks(
    self: Any, x: jnp.ndarray, steps_up: int, train: bool
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
    self: Any, x: jnp.ndarray, steps_down: int, override_use_relu: bool, train: bool
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


def _resnet_block(self: Any, x: jnp.ndarray, train: bool) -> jnp.ndarray:
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
    self: Any, x: jnp.ndarray, features: int, train: bool
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
    self: Any, x: jnp.ndarray, features: int, train: bool
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
) -> jnp.ndarray:
    """Fuse multiple resolution blocks together."""
    norm = partial(self.norm, use_running_average=not train)

    if use_concatenation:
        fused = jnp.concatenate(blocks, axis=-1)
    else:
        fused = sum(blocks)
    fused = nn.relu(fused)
    fused = nn.Conv(
        features=channels, kernel_size=(3, 3), padding="SAME", use_bias=False
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
    name: str | None = None,
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


class HRNetBB(nn.Module):
    """High-Resolution Network backbone."""

    num_stages: int
    features: int
    target_res: float

    def setup(self) -> None:
        self.norm = partial(
            nn.BatchNorm, momentum=0.9, epsilon=1e-5, dtype=jnp.float32
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Process input through multi-resolution stages."""
        x = _down_with_skip_beginning(self, x, self.features * 2, train)
        x = _down_with_skip_beginning(self, x, self.features, train)

        batch, height, width, channels = x.shape
        blocks = [x]
        block_sizes = [1]
        for stage in range(self.num_stages):
            resolution_groups = defaultdict(list)
            new_blocks = []
            new_block_sizes = []
            lowest_res = 1 / (2 ** (stage + 1))
            for block, block_size in zip(blocks, block_sizes):
                num_steps_up = int(np.abs(np.log2(block_size)))
                num_steps_down = int(np.abs(np.log2(lowest_res / block_size)))

                for step in range(num_steps_up):
                    current_block = block
                    curr_res = block_size * (2 ** (step + 1))
                    if stage == (self.num_stages - 1):
                        if self.target_res == curr_res:
                            current_block = _up_blocks(
                                self, current_block, step + 1, train
                            )
                            resolution_groups[curr_res].append(current_block)
                    else:
                        current_block = _up_blocks(self, current_block, step + 1, train)
                        resolution_groups[curr_res].append(current_block)

                for step in range(num_steps_down):
                    current_block = block
                    curr_res = block_size * (1 / (2 ** (step + 1)))
                    use_relu_last_ds = step == (num_steps_down - 1)
                    if stage == (self.num_stages - 1):
                        if self.target_res == curr_res:
                            current_block = _down_blocks(
                                self, current_block, step + 1, use_relu_last_ds, train
                            )
                            resolution_groups[curr_res].append(current_block)
                    else:
                        current_block = _down_blocks(
                            self, current_block, step + 1, use_relu_last_ds, train
                        )
                        resolution_groups[curr_res].append(current_block)

                if stage == (self.num_stages - 1):
                    if self.target_res == block_size:
                        current_block = block
                        current_block = _resnet_block(self, current_block, train)
                        resolution_groups[block_size].append(current_block)
                else:
                    current_block = block
                    current_block = _resnet_block(self, current_block, train)
                    resolution_groups[block_size].append(current_block)

            for resolution, resolution_blocks in resolution_groups.items():
                if len(resolution_blocks) == 1:
                    new_blocks.append(resolution_blocks[0])
                    new_block_sizes.append(resolution)
                else:
                    use_concat_last_step = stage == (self.num_stages - 1)
                    fused_block = _fuse_blocks(
                        self, resolution_blocks, channels, use_concat_last_step, train
                    )
                    new_blocks.append(fused_block)
                    new_block_sizes.append(resolution)

            blocks = new_blocks
            block_sizes = new_block_sizes

        backbone_out = blocks[-1]

        return backbone_out


class MoNet(nn.Module):
    """Multi-output network based on HRNet backbone."""

    num_stages: int
    features: int
    target_res: float
    train_bb: bool
    outputs: list

    def setup(self) -> None:
        self.norm = partial(
            nn.BatchNorm, momentum=0.9, epsilon=1e-5, dtype=jnp.float32
        )
        self.backbone = HRNetBB(self.num_stages, self.features, self.target_res)

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> list[jnp.ndarray]:
        """Process input through backbone and multiple output heads."""
        backbone_out = self.backbone(x, (train and self.train_bb))

        outputs = []
        for descriptor in self.outputs:
            num_outs, use_sigmoid = descriptor
            x = _basic_block(self, backbone_out, self.features, train)
            x = _basic_block(self, x, self.features, train)
            x = _basic_block(self, x, num_outs, train, kernel_size=1)
            out = nn.Conv(features=num_outs, kernel_size=(1, 1), use_bias=True)(x)
            if use_sigmoid:
                out = nn.sigmoid(out)
            outputs.append(out)

        outputs.append(backbone_out)
        return outputs

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
        for _ in range(5):
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
