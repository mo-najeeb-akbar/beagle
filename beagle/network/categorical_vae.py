from typing import Sequence

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax import random
import flax.linen as nn


class Encoder(nn.Module):
    """VAE encoder for categorical data with positional embeddings."""

    latent_dim: int
    embedding_dim: int
    num_categories: int
    hidden_dims: Sequence[int]
    mlp_dims: Sequence[int] = (32, 32)
    dropout_rate: float = 0.1
    use_batch_norm: bool = True

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, pos_embed: jnp.ndarray, train: bool = True
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Encode categorical data to latent distribution.

        Args:
            x: Categorical indices [batch_size, input_dim]
            pos_embed: Positional embeddings [input_dim, embedding_dim]
            train: Whether in training mode

        Returns:
            mu: Latent mean [batch_size, latent_dim]
            log_var: Latent log variance [batch_size, latent_dim]
        """
        x = jax.nn.one_hot(x, self.num_categories)
        for i, hidden_dim in enumerate(self.mlp_dims):
            x = nn.Dense(hidden_dim, name=f"mlp_{i}")(x)
            x = nn.swish(x)

        x = x + pos_embed[None, :, :]

        attn_weights = nn.Dense(1, name="attn")(x)
        attn_weights = jnn.softmax(attn_weights, axis=1)
        x = jnp.sum(x * attn_weights, axis=1)

        for i, hidden_dim in enumerate(self.hidden_dims):
            residual = x

            x = nn.Dense(hidden_dim, use_bias=not self.use_batch_norm)(x)
            x = nn.LayerNorm()(x)
            x = nn.swish(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

            if residual.shape[-1] == hidden_dim:
                x = x + residual

        mu = nn.Dense(self.latent_dim, name="dense_mu")(x)
        log_var = nn.Dense(self.latent_dim, name="dense_logvar")(x)

        return mu, log_var


class Decoder(nn.Module):
    """VAE decoder for categorical data with positional embeddings."""

    latent_dim: int
    input_dim: int
    num_categories: int
    hidden_dims: Sequence[int]
    mlp_dims: Sequence[int] = (32, 32)
    dropout_rate: float = 0.1
    use_batch_norm: bool = True

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, pos_embed: jnp.ndarray, train: bool = True
    ) -> jnp.ndarray:
        """Decode latent vector to categorical logits.

        Args:
            x: Latent vector [batch_size, latent_dim]
            pos_embed: Positional embeddings [input_dim, embedding_dim]
            train: Whether in training mode

        Returns:
            logits: Unnormalized log probabilities [batch_size, input_dim, num_categories]
        """
        for i, hidden_dim in enumerate(self.hidden_dims):
            residual = x

            x = nn.Dense(hidden_dim, use_bias=not self.use_batch_norm)(x)
            x = nn.LayerNorm()(x)
            x = nn.swish(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

            if residual.shape[-1] == hidden_dim:
                x = x + residual

        x = nn.Dense(32)(x)
        x = x[:, None, :]
        pos_embed = pos_embed[None, :, :]
        x = x + pos_embed

        for i, hidden_dim in enumerate(self.mlp_dims):
            x = nn.Dense(hidden_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.swish(x)

        logits = nn.Dense(self.num_categories, name="out_dense")(x)

        return logits


class CategoricalVAE(nn.Module):
    """VAE for categorical sequence data with positional embeddings."""

    input_dim: int
    embedding_dim: int
    num_categories: int
    latent_dim: int = 32
    encoder_hidden_dims: Sequence[int] = (512, 256, 128)
    decoder_hidden_dims: Sequence[int] = (128, 64, 64)
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    use_vae_sampling: bool = True

    def setup(self) -> None:
        decoder_dims = self.decoder_hidden_dims
        if decoder_dims is None:
            decoder_dims = self.encoder_hidden_dims[::-1]

        self.embedding = nn.Embed(num_embeddings=self.input_dim, features=32)

        self.encoder = Encoder(
            latent_dim=self.latent_dim,
            embedding_dim=self.embedding_dim,
            num_categories=self.num_categories,
            hidden_dims=self.encoder_hidden_dims,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm,
        )

        self.decoder = Decoder(
            latent_dim=self.latent_dim,
            input_dim=self.input_dim,
            num_categories=self.num_categories,
            hidden_dims=decoder_dims,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm,
        )

    def reparameterize(
        self, key: jax.random.PRNGKey, mu: jnp.ndarray, log_var: jnp.ndarray
    ) -> jnp.ndarray:
        """Reparameterization trick for VAE (pure function given key)."""
        if self.use_vae_sampling:
            std = jnp.exp(0.5 * log_var)
            eps = random.normal(key, mu.shape)
            return mu + std * eps
        else:
            return mu

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """Encode to latent mean (inference mode)."""
        pos_indices = jnp.arange(x.shape[1])
        pos_embed = self.embedding(pos_indices)
        mu, _ = self.encoder(x, pos_embed, False)
        return mu

    def __call__(
        self, x: jnp.ndarray, key: jax.random.PRNGKey, training: bool = True
    ) -> dict[str, jnp.ndarray]:
        """Forward pass through VAE.

        Args:
            x: Categorical data [batch_size, num_genes], integers in [0, num_categories-1]
            key: JAX random key for sampling
            training: Whether in training mode

        Returns:
            Dict with keys:
                - 'logits': Unnormalized log probabilities [batch_size, num_genes, num_categories]
                - 'mu': Latent mean [batch_size, latent_dim]
                - 'log_var': Latent log variance [batch_size, latent_dim]
                - 'latent_normalized': Normalized latent vector [batch_size, latent_dim]
        """
        pos_indices = jnp.arange(x.shape[1])
        pos_embed = self.embedding(pos_indices)
        mu, log_var = self.encoder(x, pos_embed, training)

        z = self.reparameterize(key, mu, log_var)
        z_normalized = mu / (jnp.linalg.norm(mu, axis=-1, keepdims=True) + 1e-8)

        logits = self.decoder(mu, pos_embed, training)

        return {
            'logits': logits,
            'mu': mu,
            'log_var': log_var,
            'latent_normalized': z_normalized
        }
    