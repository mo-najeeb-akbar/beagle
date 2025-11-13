"""Wavelet filter definitions (pure JAX arrays)."""

from __future__ import annotations

import jax.numpy as jnp


# Haar wavelet filters (orthogonal, shortest possible)
HAAR_DEC_LO = jnp.array([0.7071067811865476, 0.7071067811865476], dtype=jnp.float32)
HAAR_DEC_HI = jnp.array([-0.7071067811865476, 0.7071067811865476], dtype=jnp.float32)

# Daubechies 2 (db2) filters
DB2_DEC_LO = jnp.array([
    -0.1294095226, 0.2241438680, 0.8365163037, 0.4829629131
], dtype=jnp.float32)
DB2_DEC_HI = jnp.array([
    -0.4829629131, 0.8365163037, -0.2241438680, -0.1294095226
], dtype=jnp.float32)

# Daubechies 4 (db4) filters
DB4_DEC_LO = jnp.array([
    -0.0105974018, 0.0328830117, 0.0308413818, -0.1870348117,
    -0.0279837694, 0.6308807679, 0.7148465706, 0.2303778133
], dtype=jnp.float32)
DB4_DEC_HI = jnp.array([
    -0.2303778133, 0.7148465706, -0.6308807679, -0.0279837694,
    0.1870348117, 0.0308413818, -0.0328830117, -0.0105974018
], dtype=jnp.float32)


WAVELETS = {
    'haar': (HAAR_DEC_LO, HAAR_DEC_HI),
    'db2': (DB2_DEC_LO, DB2_DEC_HI),
    'db4': (DB4_DEC_LO, DB4_DEC_HI),
}


def get_filters(
    wavelet: str,
    flip: bool = True,
    dtype: jnp.dtype | None = None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get decomposition filters for wavelet (pure).
    
    Args:
        wavelet: Wavelet name ('haar', 'db2', 'db4')
        flip: Whether to flip filters for convolution (default True for decomposition)
        dtype: Target dtype for filters (None = keep float32, useful for mixed precision)
        
    Returns:
        (dec_lo, dec_hi) decomposition filters in requested dtype
        
    Raises:
        ValueError: If wavelet not supported
    """
    if wavelet not in WAVELETS:
        raise ValueError(
            f"Wavelet '{wavelet}' not supported. "
            f"Available: {list(WAVELETS.keys())}"
        )
    dec_lo, dec_hi = WAVELETS[wavelet]
    
    # Cast to target dtype if specified (for mixed precision)
    if dtype is not None and dtype != jnp.float32:
        dec_lo = dec_lo.astype(dtype)
        dec_hi = dec_hi.astype(dtype)
    
    if flip:
        dec_lo = jnp.flip(dec_lo)
        dec_hi = jnp.flip(dec_hi)
    
    return dec_lo, dec_hi


def construct_2d_filters(lo: jnp.ndarray, hi: jnp.ndarray) -> jnp.ndarray:
    """Construct 2D filters from 1D filters using outer products (pure).
    
    Args:
        lo: 1D lowpass filter of shape (L,)
        hi: 1D highpass filter of shape (L,)
        
    Returns:
        4D filter tensor of shape (4, 1, L, L) containing:
        [LL, LH, HL, HH] subbands
    """
    ll = jnp.outer(lo, lo)
    lh = jnp.outer(hi, lo)
    hl = jnp.outer(lo, hi)
    hh = jnp.outer(hi, hi)
    
    filt = jnp.stack([ll, lh, hl, hh], axis=0)
    return filt[:, jnp.newaxis, :, :]

