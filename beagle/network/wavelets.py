from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


class HaarWaveletConv(nn.Module):
    """Haar wavelet decomposition using Conv2D (single channel only).
    
    Applies 2D Haar wavelet transform using strided convolution, producing
    4 subbands (LL, LH, HL, HH) from a single-channel input.
    
    Input shape: (B, H, W, 1)
    Output shape: (B, H//2, W//2, 4)
    """
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply Haar wavelet decomposition.
        
        Args:
            x: Input images of shape (B, H, W, 1)
            
        Returns:
            Wavelet coefficients of shape (B, H//2, W//2, 4)
            Channels are ordered: [LL, LH, HL, HH]
        """
        s = 0.5
        
        # Flax Conv kernel shape: (h, w, in_channels, out_channels)
        # Decomposition filters (with flip): lo=[0.7071, 0.7071], hi=[0.7071, -0.7071]
        # LL = outer(lo,lo), LH = outer(hi,lo), HL = outer(lo,hi), HH = outer(hi,hi)
        filters = np.zeros((2, 2, 1, 4), dtype=np.float32)
        filters[0, 0, 0, :] = [s, s, s, s]
        filters[0, 1, 0, :] = [s, s, -s, -s]
        filters[1, 0, 0, :] = [s, -s, s, -s]
        filters[1, 1, 0, :] = [s, -s, -s, s]
        
        return nn.Conv(
            features=4,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding='VALID',
            use_bias=False,
            kernel_init=lambda *args: jax.lax.stop_gradient(jnp.array(filters, dtype=jnp.float32)),
        )(x)


class HaarWaveletConvTranspose(nn.Module):
    """Haar wavelet reconstruction using ConvTranspose2D (single channel only).
    
    Reconstructs single-channel image from 4 Haar wavelet subbands using
    transposed convolution.
    
    Input shape: (B, H, W, 4)
    Output shape: (B, H*2, W*2, 1)
    """
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply Haar wavelet reconstruction.
        
        Args:
            x: Wavelet coefficients of shape (B, H, W, 4)
               Channels are ordered: [LL, LH, HL, HH]
               
        Returns:
            Reconstructed images of shape (B, H*2, W*2, 1)
        """
        s = 0.5
        
        # Flax ConvTranspose kernel shape: (h, w, in_channels, out_channels)
        # Reconstruction filters (no flip): lo=[0.7071, 0.7071], hi=[-0.7071, 0.7071]
        # LL = outer(lo,lo), LH = outer(hi,lo), HL = outer(lo,hi), HH = outer(hi,hi)
        filters = np.zeros((2, 2, 4, 1), dtype=np.float32)
        filters[0, 0, :, 0] = [s, -s, -s, s]
        filters[0, 1, :, 0] = [s, -s, s, -s]
        filters[1, 0, :, 0] = [s, s, -s, -s]
        filters[1, 1, :, 0] = [s, s, s, s]
        
        return nn.ConvTranspose(
            features=1,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding='SAME',
            use_bias=False,
            kernel_init=lambda *args: jax.lax.stop_gradient(jnp.array(filters, dtype=jnp.float32)),
        )(x)

