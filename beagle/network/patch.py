import jax.numpy as jnp
import flax.linen as nn


class PatchEmbed(nn.Module):
    """Convert image to patches using conv with stride=patch_size."""

    img_size: int
    patch_dim: int
    embed_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B = x.shape[0]
        x = nn.Conv(
            self.embed_dim,
            kernel_size=(self.patch_dim, self.patch_dim),
            strides=(self.patch_dim, self.patch_dim),
            padding="VALID",
        )(x)
        x = x.reshape((B, -1, self.embed_dim))

        return x
