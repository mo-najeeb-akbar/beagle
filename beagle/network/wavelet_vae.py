from functools import partial

import jax
import jax.numpy as jnp
from jax import image, random
import flax.linen as nn

from beagle.wavelets import waverec2


class ResidualBlock(nn.Module):
    """Residual block with group normalization."""

    filters: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        norm = partial(nn.GroupNorm, num_groups=8)
        act = nn.swish
        skip = x

        x = nn.Conv(self.filters, (3, 3), padding=1, use_bias=False, name="conv1")(x)
        x = norm(name="gn1")(x)
        x = act(x)

        x = nn.Conv(self.filters, (3, 3), padding=1, use_bias=False, name="conv2")(x)
        x = norm(name="gn2")(x)

        return act(x + skip)

class Encoder(nn.Module):
    """VAE encoder with downsampling conv blocks."""

    latent_dim: int
    features: int

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, training: bool = True
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        norm = partial(nn.GroupNorm, num_groups=8)
        act = nn.swish

        for i in range(5):
            x = nn.Conv(
                self.features,
                (3, 3),
                strides=(2, 2),
                padding=1,
                use_bias=False,
                name=f"conv_layers.{i}",
            )(x)
            x = norm(name=f"gn_layers.{i}")(x)
            x = act(x)
            x = ResidualBlock(self.features)(x, training=training)

        mu = nn.Dense(self.latent_dim, name="dense_mu")(x)
        log_var = nn.Dense(self.latent_dim, name="dense_logvar")(x)

        return mu, log_var

class Decoder(nn.Module):
    """VAE decoder with upsampling conv blocks."""

    latent_dim: int
    bottle_neck: int
    features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        norm = partial(nn.GroupNorm, num_groups=8)
        act = nn.swish

        for i in range(5):
            batch, height, width, channels = x.shape
            x = image.resize(
                x, shape=(batch, height * 2, width * 2, channels), method="bilinear"
            )
            x = nn.Conv(
                features=self.features,
                kernel_size=(3, 3),
                padding="SAME",
                use_bias=False,
            )(x)
            x = norm(name=f"gn_layers.{i}")(x)
            x = act(x)
            x = ResidualBlock(self.features)(x, training=training)

        x = nn.Conv(4, (3, 3), padding=1, name="out_conv")(x)

        return x

class VAE(nn.Module):
    """Wavelet-based VAE for image data."""

    latent_dim: int = 128
    base_features: int = 32
    block_size: int = 8

    def setup(self) -> None:
        self.Encoder = Encoder(self.latent_dim, self.base_features)
        self.Decoder = Decoder(self.latent_dim, self.block_size, self.base_features)

    def encode(
        self, x: jnp.ndarray, training: bool = True
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Encode image to latent distribution."""
        return self.Encoder(x, training=training)

    def decode(self, z: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Decode latent to wavelet coefficients."""
        x_recon = self.Decoder(z, training=training)
        return x_recon

    def decode_full(self, z: jnp.ndarray) -> jnp.ndarray:
        """Decode latent to full image via wavelet reconstruction."""
        x_recon = self.Decoder(z, training=False)
        # Rearrange channels from [LL, HL, LH, HH] to [LL, LH, HL, HH]
        x_recon_reordered = jnp.stack([
            x_recon[..., 0],  # LL
            x_recon[..., 2],  # LH
            x_recon[..., 1],  # HL
            x_recon[..., 3],  # HH
        ], axis=-1)
        x_recon = waverec2(x_recon_reordered, wavelet="haar")
        return x_recon

    def reparameterize(
        self, key: jax.random.PRNGKey, mu: jnp.ndarray, log_var: jnp.ndarray
    ) -> jnp.ndarray:
        """Reparameterization (currently disabled, returns mu only)."""
        return mu

    def __call__(
        self, x: jnp.ndarray, key: jax.random.PRNGKey, training: bool = True
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass through VAE.

        Returns:
            reconstructed: Full reconstructed image
            x_recon: Wavelet coefficients
            mu: Latent mean
            log_var: Latent log variance
        """
        mu, log_var = self.encode(x, training)
        z = self.reparameterize(key, mu, log_var)
        x_recon = self.decode(z, training)
        # Rearrange channels from [LL, HL, LH, HH] to [LL, LH, HL, HH]
        x_recon_reordered = jnp.stack([
            x_recon[..., 0],  # LL
            x_recon[..., 2],  # LH
            x_recon[..., 1],  # HL
            x_recon[..., 3],  # HH
        ], axis=-1)
        reconstructed = waverec2(x_recon_reordered, wavelet="haar")
        return reconstructed, x_recon, mu, log_var
