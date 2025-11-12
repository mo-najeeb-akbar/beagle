import jax
import jax.numpy as jnp
import flax.linen as nn


class DenseEncoder(nn.Module):
    """Dense encoder with categorical embedding and reconstruction head."""

    input_dim: int
    embed_dim: int
    num_categories: int
    project_dim: int
    recon_sample_ratio: float = 0.1
    dropout_rate: float = 0.5

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, rng: jax.random.PRNGKey, training: bool = True
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Encode categorical input to normalized latent vector with reconstruction.

        Args:
            x: Categorical indices [batch, input_dim]
            rng: Random key for sampling reconstruction positions
            training: Whether in training mode

        Returns:
            h: Normalized latent vector [batch, project_dim]
            recon_logits: Reconstruction logits [batch, num_recon, num_categories]
            targets: Original values at sampled positions [batch, num_recon]
        """
        embedded = nn.Embed(
            num_embeddings=self.num_categories, features=self.embed_dim
        )(x)

        if training:
            embedded = nn.Dropout(rate=self.dropout_rate)(embedded, deterministic=False)

        pooled = jnp.mean(embedded, axis=1)

        h = nn.Dense(self.project_dim // 2, name="project1")(pooled)
        h = nn.gelu(h)

        if training:
            h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=False)

        h = nn.Dense(self.project_dim, name="project2")(h)

        if training:
            h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=False)

        h = h / (jnp.linalg.norm(h, axis=-1, keepdims=True) + 1e-8)

        sample_ratio = self.recon_sample_ratio if training else 0.5
        num_recon = int(self.input_dim * sample_ratio)
        recon_indices = jax.random.choice(
            rng, self.input_dim, shape=(num_recon,), replace=False
        )

        h_tiled = jnp.tile(h[:, None, :], (1, num_recon, 1))

        pos_embed = nn.Embed(num_embeddings=self.input_dim, features=8)(recon_indices)
        pos_embed = jnp.tile(pos_embed[None, :, :], (h.shape[0], 1, 1))

        decoder_input = jnp.concatenate([h_tiled, pos_embed], axis=-1)

        decoded = nn.Dense(64)(decoder_input)
        decoded = nn.gelu(decoded)

        if training:
            decoded = nn.Dropout(rate=0.3)(decoded, deterministic=False)

        recon_logits = nn.Dense(self.num_categories)(decoded)

        return h, recon_logits, x[:, recon_indices]

    def get_gene_importance_scores(self, params: dict) -> jnp.ndarray:
        """Get learned importance scores for each gene.

        Args:
            params: Model parameters dictionary

        Returns:
            Softmax-normalized importance weights [num_genes]
        """
        gene_importance = params["importance"]
        return jax.nn.softmax(gene_importance)
