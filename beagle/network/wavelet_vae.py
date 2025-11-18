from functools import partial

import jax
import jax.numpy as jnp
from jax import image, random
import flax.linen as nn

from beagle.network.wavelets import HaarWaveletConv, HaarWaveletConvTranspose


class ResidualBlock(nn.Module):
    """Residual block with group normalization."""

    filters: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        norm = partial(nn.GroupNorm, num_groups=8)
        act = nn.swish
        skip = x

        x = nn.Conv(self.filters, (3, 3), padding='SAME', use_bias=False, name="conv1")(x)
        x = norm(name="gn1")(x)
        x = act(x)

        x = nn.Conv(self.filters, (3, 3), padding='SAME', use_bias=False, name="conv2")(x)
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

        x = HaarWaveletConv(name="haar_conv")(x)
        x = nn.GroupNorm(num_groups=4, name="gn_haar")(x)
        for i in range(5):
            x = nn.Conv(
                self.features,
                (3, 3),
                strides=(2, 2),
                padding='SAME',
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
                padding='SAME',
                use_bias=False,
                name=f"conv_layers.{i}"
            )(x)
            x = norm(name=f"gn_layers.{i}")(x)
            x = act(x)
            x = ResidualBlock(self.features)(x, training=training)

        x_haar = nn.Conv(4, (3, 3), padding='SAME', name="out_conv")(x)
        x_recon = HaarWaveletConvTranspose(name="haar_conv_transpose")(x_haar)

        return x_recon, x_haar

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
        x_recon, x_haar = self.Decoder(z, training=training)
        return x_recon, x_haar


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
        x_recon, x_haar = self.decode(z, training)
        return x_recon, x_haar, mu, log_var
