import flax.linen as nn
import jax.numpy as jnp


class Attention(nn.Module):
    """Multi-head attention with QK normalization."""

    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        head_dim = self.embed_dim // self.num_heads
        scale = head_dim**-0.5

        B, P, D = x.shape

        qkv = nn.Dense(self.embed_dim * 3, use_bias=False)(x)
        qkv = qkv.reshape(B, P, 3, self.num_heads, head_dim).transpose(
            (2, 0, 3, 1, 4)
        )
        q, k, v = qkv
        q = nn.LayerNorm()(q)
        k = nn.LayerNorm()(k)
        q = q * scale

        attn = q @ jnp.swapaxes(k, -2, -1)
        attn = nn.softmax(attn, axis=-1)

        x = attn @ v
        x = jnp.swapaxes(x, 1, 2).reshape(B, P, D)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.embed_dim, use_bias=True)(x)

        return x


class LayerScale(nn.Module):
    """Layer-wise learnable scaling parameter."""

    dim: int
    init_values: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gamma = self.param(
            "gamma",
            lambda rng, shape: jnp.ones(shape) * self.init_values,
            (self.dim,),
        )
        return x * gamma


class MLP(nn.Module):
    """Two-layer MLP with GELU activation."""

    hidden_features: int
    out_features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        o = nn.Dense(self.hidden_features, use_bias=True)(x)
        o = nn.gelu(o)
        o = nn.Dense(self.out_features)(o)

        return o


class Block(nn.Module):
    """Transformer block with attention and MLP, both with LayerScale."""

    embed_dim: int
    num_heads: int
    mlp_ratio: float

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        o = nn.LayerNorm()(x)
        o = Attention(embed_dim=self.embed_dim, num_heads=self.num_heads)(o)
        o = LayerScale(dim=self.embed_dim)(o)

        x = x + o

        o = nn.LayerNorm()(x)
        o = MLP(
            hidden_features=int(self.mlp_ratio * self.embed_dim),
            out_features=self.embed_dim,
        )(o)
        o = LayerScale(dim=self.embed_dim)(o)

        x = x + o

        return x

