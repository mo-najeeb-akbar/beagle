from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding with MLP projection."""

    dim: int

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Generate time embeddings.

        Args:
            t: Timesteps [batch]

        Returns:
            Time embeddings [batch, dim]
        """
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)

        emb = nn.Dense(self.dim * 4)(emb)
        emb = nn.gelu(emb)
        emb = nn.Dense(self.dim)(emb)
        return emb


class ResBlock(nn.Module):
    """ResNet block with time embedding injection."""

    out_channels: int
    dropout: float = 0.1

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, time_emb: jnp.ndarray, train: bool = True
    ) -> jnp.ndarray:
        h = nn.GroupNorm(num_groups=32)(x)
        h = nn.gelu(h)
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding="SAME")(h)

        time_emb = nn.Dense(self.out_channels)(nn.gelu(time_emb))
        h = h + time_emb[:, None, None, :]

        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.gelu(h)
        h = nn.Dropout(self.dropout, deterministic=not train)(h)
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding="SAME")(h)

        if x.shape[-1] != self.out_channels:
            x = nn.Conv(self.out_channels, kernel_size=(1, 1))(x)

        return x + h


class SelfAttentionBlock(nn.Module):
    """Self-attention block for spatial features."""

    num_heads: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch, height, width, channels = x.shape

        h = nn.GroupNorm(num_groups=32)(x)
        h = h.reshape(batch, height * width, channels)

        h = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=channels, out_features=channels
        )(h, h)

        h = h.reshape(batch, height, width, channels)
        return x + h


class CrossAttentionBlock(nn.Module):
    """Cross-attention to context embeddings."""

    num_heads: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray, context: jnp.ndarray) -> jnp.ndarray:
        """Apply cross-attention.

        Args:
            x: Spatial features (batch, height, width, channels)
            context: Context embeddings (batch, context_dim)

        Returns:
            Output features with same shape as x
        """
        batch, height, width, channels = x.shape

        h = nn.GroupNorm(num_groups=32)(x)
        h = h.reshape(batch, height * width, channels)

        context_seq = context[:, None, :]

        h = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=channels,
        )(h, context_seq)

        h = h.reshape(batch, height, width, channels)
        return x + h


class DownBlock(nn.Module):
    """Downsampling block with optional attention."""

    out_channels: int
    num_res_blocks: int = 2
    use_attention: bool = False
    dropout: float = 0.1
    context_dim: Optional[int] = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        time_emb: jnp.ndarray,
        context: Optional[jnp.ndarray] = None,
        train: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        h = x
        for _ in range(self.num_res_blocks):
            h = ResBlock(self.out_channels, self.dropout)(h, time_emb, train)
            if self.use_attention:
                h = SelfAttentionBlock()(h)
                if context is not None:
                    h = CrossAttentionBlock()(h, context)

        h_down = nn.Conv(
            self.out_channels, kernel_size=(3, 3), strides=(2, 2), padding="SAME"
        )(h)

        return h_down, h


class UpBlock(nn.Module):
    """Upsampling block with optional attention."""

    out_channels: int
    num_res_blocks: int = 2
    use_attention: bool = False
    dropout: float = 0.1
    context_dim: Optional[int] = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        skip: jnp.ndarray,
        time_emb: jnp.ndarray,
        context: Optional[jnp.ndarray] = None,
        train: bool = True,
    ) -> jnp.ndarray:
        batch, height, width, channels = x.shape
        h = jax.image.resize(
            x, (batch, height * 2, width * 2, channels), method="nearest"
        )
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding="SAME")(h)

        h = jnp.concatenate([h, skip], axis=-1)

        for _ in range(self.num_res_blocks):
            h = ResBlock(self.out_channels, self.dropout)(h, time_emb, train)
            if self.use_attention:
                h = SelfAttentionBlock()(h)
                if context is not None:
                    h = CrossAttentionBlock()(h, context)

        return h


class DenoisingUNet(nn.Module):
    """U-Net for denoising latent representations with cross-attention conditioning."""

    base_channels: int = 64
    channel_multipliers: Sequence[int] = (1, 2)
    num_res_blocks: int = 1
    attention_levels: Sequence[bool] = (False, True)
    dropout: float = 0.2
    context_dim: int = 1024

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        context: jnp.ndarray,
        train: bool = True,
    ) -> jnp.ndarray:
        """Denoise latent tensor conditioned on context.

        Args:
            x: Noisy latent tensor (batch, height, width, latent_channels)
            t: Timesteps (batch,)
            context: Context embeddings (batch, context_dim)
            train: Training mode

        Returns:
            Predicted noise (same shape as x)
        """
        time_emb = TimeEmbedding(self.base_channels)(t)

        h = nn.Conv(self.base_channels, kernel_size=(3, 3), padding="SAME")(x)

        skip_connections = []
        for i, (mult, use_attn) in enumerate(
            zip(self.channel_multipliers, self.attention_levels)
        ):
            out_ch = self.base_channels * mult
            h, skip = DownBlock(
                out_ch, self.num_res_blocks, use_attn, self.dropout, self.context_dim
            )(h, time_emb, context, train)
            skip_connections.append(skip)

        h = ResBlock(self.base_channels * self.channel_multipliers[-1], self.dropout)(
            h, time_emb, train
        )
        h = SelfAttentionBlock()(h)
        h = CrossAttentionBlock()(h, context)
        h = ResBlock(self.base_channels * self.channel_multipliers[-1], self.dropout)(
            h, time_emb, train
        )

        for i, (mult, use_attn) in enumerate(
            zip(reversed(self.channel_multipliers), reversed(self.attention_levels))
        ):
            out_ch = self.base_channels * mult
            skip = skip_connections.pop()
            h = UpBlock(
                out_ch, self.num_res_blocks, use_attn, self.dropout, self.context_dim
            )(h, skip, time_emb, context, train)

        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.gelu(h)
        h = nn.Conv(x.shape[-1], kernel_size=(3, 3), padding="SAME")(h)

        return h


