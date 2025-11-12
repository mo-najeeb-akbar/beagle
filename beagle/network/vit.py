import jax
import jax.numpy as jnp
import flax.linen as nn

from beagle.network.attention import Block
from beagle.network.patch import PatchEmbed
from beagle.network.position import get_2d_sincos_pos_embed

class MaskedViT(nn.Module):
    """Masked Vision Transformer for self-supervised learning."""

    img_size: int = 128
    patch_size: int = 16
    in_channels: int = 1
    embed_dim: int = 256
    depth: int = 2
    num_heads: int = 8
    decoder_embed_dim: int = 32
    decoder_depth: int = 2
    decoder_num_heads: int = 8
    mlp_ratio: float = 4.0

    def setup(self) -> None:
        self.patch_embed = PatchEmbed(self.img_size, self.patch_size, self.embed_dim)
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.grid_size = self.img_size // self.patch_size

        pos_embed = get_2d_sincos_pos_embed(
            self.embed_dim, self.grid_size, cls_token=False, expand_first_dim=True
        )
        self.pos_embed = self.variable(
            "params", "pos_embed", lambda: jnp.array(pos_embed)
        )

        self.blocks = nn.Sequential(
            [
                Block(self.embed_dim, self.num_heads, self.mlp_ratio)
                for _ in range(self.depth)
            ]
        )
        self.norm = nn.LayerNorm()

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_embed_dim,
            self.grid_size,
            cls_token=False,
            expand_first_dim=True,
        )
        self.decoder_pos_embed = self.variable(
            "params", "decoder_pos_embed", lambda: jnp.array(decoder_pos_embed)
        )

        self.decoder_embed = nn.Dense(self.decoder_embed_dim, use_bias=True)
        self.mask_token = self.param(
            "mask_token",
            nn.initializers.normal(0.02),
            (1, 1, self.decoder_embed_dim),
        )
        self.decoder_blocks = nn.Sequential(
            [
                Block(self.decoder_embed_dim, self.decoder_num_heads, self.mlp_ratio)
                for _ in range(self.decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm()
        self.decoder_pred = nn.Dense(
            self.patch_size**2 * self.in_channels, use_bias=True
        )
    

    def random_masking(
        self, x: jnp.ndarray, mask_ratio: float, rng: jax.random.PRNGKey
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Randomly mask patches (pure function given rng)."""
        B, P, D = x.shape
        len_keep = int(P * (1 - mask_ratio))

        rng_perm = jax.random.split(rng, B)
        ids_shuffle = jax.vmap(lambda r: jax.random.permutation(r, P))(rng_perm)

        ids_restore = jnp.empty_like(ids_shuffle)
        ids_restore = ids_restore.at[jnp.arange(B)[:, None], ids_shuffle].set(
            jnp.arange(P)[None, :]
        )

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = jnp.take_along_axis(x, ids_keep[..., None], axis=1)

        mask = jnp.arange(P) >= len_keep
        mask = jnp.take_along_axis(mask[None, :], ids_restore, axis=1).astype(
            jnp.float32
        )

        return x_masked, mask, ids_restore

    def encode(
        self, x: jnp.ndarray, mask_ratio: float, rng: jax.random.PRNGKey
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Encode with masking."""
        x = self.patch_embed(x)
        x = x + self.pos_embed.value
        x, mask, ids_restore = self.random_masking(x, mask_ratio, rng)
        x = self.blocks(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def decode(
        self, x: jnp.ndarray, mask: jnp.ndarray, ids_restore: jnp.ndarray
    ) -> jnp.ndarray:
        """Decode masked patches."""
        x = self.decoder_embed(x)
        num_mask_tokens = ids_restore.shape[1] - x.shape[1]
        mask_tokens = jnp.broadcast_to(
            self.mask_token, (x.shape[0], num_mask_tokens, self.decoder_embed_dim)
        )
        x = jnp.concatenate([x, mask_tokens], axis=1)
        x = jnp.take_along_axis(x, ids_restore[..., None], axis=1)

        x = x + self.decoder_pos_embed.value
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x

    def patchify(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convert image to patches (pure function).

        Args:
            x: Image tensor (B, H, W, C)

        Returns:
            Patches tensor (B, num_patches, patch_size**2 * C)
        """
        B, H, W, C = x.shape
        p = self.patch_size

        num_patches_per_side = H // p

        x = x.reshape(B, num_patches_per_side, p, num_patches_per_side, p, C)
        x = jnp.einsum("nhpwqc->nhwpqc", x)
        x = x.reshape(B, num_patches_per_side * num_patches_per_side, p * p * C)

        return x

    def unpatchify(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convert patches back to image (pure function).

        Args:
            x: Patches tensor (B, num_patches, patch_size**2 * C)

        Returns:
            Image tensor (B, H, W, C)
        """
        B = x.shape[0]
        p = self.patch_size
        h = w = self.img_size // p
        C = self.in_channels

        x = x.reshape(B, h, w, p, p, C)
        x = jnp.einsum("nhwpqc->nhpwqc", x)
        x = x.reshape(B, h * p, w * p, C)

        return x

    def __call__(
        self, x: jnp.ndarray, mask_ratio: float, rng: jax.random.PRNGKey
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass with masking.

        Returns:
            decoded: Predicted patches
            mask: Binary mask (1 = masked, 0 = kept)
            target: Original patches
        """
        encoded, mask, ids_restore = self.encode(x, mask_ratio, rng)
        decoded = self.decode(encoded, mask, ids_restore)
        target = self.patchify(x)
        return decoded, mask, target