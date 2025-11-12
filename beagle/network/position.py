import jax.numpy as jnp


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: jnp.ndarray) -> jnp.ndarray:
    """Generate 1D sinusoidal positional embeddings (pure function).

    Args:
        embed_dim: Output dimension for each position (must be even)
        pos: Array of positions to encode, shape (M,)

    Returns:
        Positional embeddings of shape (M, embed_dim)
    """
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = jnp.einsum("m,d->md", pos, omega)

    emb_sin = jnp.sin(out)
    emb_cos = jnp.cos(out)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int, grid: jnp.ndarray
) -> jnp.ndarray:
    """Generate 2D positional embeddings from grid (pure function).

    Args:
        embed_dim: Output dimension (must be even)
        grid: 2D grid of shape [2, 1, grid_size, grid_size]

    Returns:
        Positional embeddings of shape (H*W, embed_dim)
    """
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = jnp.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
    expand_first_dim: bool = True,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Generate 2D sinusoidal positional embeddings (pure function).

    Args:
        embed_dim: Embedding dimension
        grid_size: Grid height and width (assumes square grid)
        cls_token: If True, prepend a zero vector for class token
        expand_first_dim: If True, add batch dimension
        dtype: Data type for embeddings

    Returns:
        Positional embeddings of shape [1, grid_size*grid_size, embed_dim]
        or [grid_size*grid_size, embed_dim] depending on expand_first_dim
    """
    grid_h = jnp.arange(grid_size, dtype=dtype)
    grid_w = jnp.arange(grid_size, dtype=dtype)
    grid = jnp.meshgrid(grid_w, grid_h)
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token:
        pos_embed = jnp.concatenate(
            [jnp.zeros([1, embed_dim], dtype=dtype), pos_embed], axis=0
        )

    if expand_first_dim:
        pos_embed = pos_embed[jnp.newaxis, ...]

    return pos_embed