import jax
import jax.numpy as jnp
from flax import linen as nn


class CompactVAE(nn.Module):
    """Compact VAE for independent categorical genes."""

    latent_dim: int = 64
    num_categories: int = 8
    dropout_rate: float = 0.5

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, rng: jax.random.PRNGKey, training: bool = True
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Encode categorical genes to latent space and decode.

        Args:
            x: Categorical values [batch, num_genes]
            rng: Random key for reparameterization
            training: Whether in training mode

        Returns:
            logits: Reconstruction logits [batch, num_genes, num_categories]
            mu: Latent mean [batch, latent_dim]
            logvar: Latent log variance [batch, latent_dim]
            z: Sampled latent (or mu if not training) [batch, latent_dim]
        """
        batch_size, num_genes = x.shape

        h = nn.Embed(num_embeddings=self.num_categories, features=4)(x)
        h = nn.Dropout(self.dropout_rate, deterministic=not training)(h)

        h = jnp.mean(h, axis=1)

        h = nn.Dense(128)(h)
        h = nn.relu(h)
        h = nn.Dropout(self.dropout_rate, deterministic=not training)(h)

        h = nn.Dense(64)(h)
        h = nn.relu(h)

        mu = nn.Dense(self.latent_dim)(h)
        logvar = nn.Dense(self.latent_dim)(h)

        if training:
            std = jnp.exp(0.5 * logvar)
            eps = jax.random.normal(rng, mu.shape)
            z = mu + eps * std
        else:
            z = mu

        h = nn.Dense(64)(z)
        h = nn.relu(h)
        h = nn.Dropout(self.dropout_rate, deterministic=not training)(h)

        h = nn.Dense(128)(h)
        h = nn.relu(h)
        h = nn.Dropout(self.dropout_rate, deterministic=not training)(h)

        logits = nn.Dense(num_genes * self.num_categories)(h)
        logits = logits.reshape(batch_size, num_genes, self.num_categories)

        return logits, mu, logvar, z
    