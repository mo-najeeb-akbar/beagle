"""Convolution-based wavelet transforms (pure JAX operations)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from beagle.wavelets.filters import get_filters, construct_2d_filters


def pad_symmetric(data: jnp.ndarray, filt_len: int) -> jnp.ndarray:
    """Pad data symmetrically for wavelet transform (pure).
    
    Args:
        data: Input of shape (B, C, H, W)
        filt_len: Length of wavelet filter
        
    Returns:
        Symmetrically padded data
    """
    pad_size = (2 * filt_len - 3) // 2
    
    # Ensure even dimensions after padding
    pad_h = 1 if data.shape[-2] % 2 != 0 else 0
    pad_w = 1 if data.shape[-1] % 2 != 0 else 0
    
    return jnp.pad(
        data,
        ((0, 0), (0, 0), (pad_size, pad_size + pad_h), (pad_size, pad_size + pad_w)),
        mode='symmetric'
    )


def wavedec2_single(
    images: jnp.ndarray,
    wavelet: str = "haar"
) -> jnp.ndarray:
    """Single-level 2D wavelet decomposition (pure, non-differentiable).
    
    Computes one level of 2D discrete wavelet transform using separable
    convolution. Gradients are stopped to prevent training interference.
    
    Args:
        images: Input of shape (B, H, W, C)
        wavelet: Wavelet name ('haar', 'db2', 'db4')
        
    Returns:
        Wavelets of shape (B, H/2, W/2, C*4) with subbands:
        [LL, LH, HL, HH] concatenated along channel dimension
        
    Note:
        This function applies stop_gradient internally - use for
        fixed preprocessing, not learnable transforms.
    """
    # Get filters matching input dtype (critical for mixed precision!)
    dec_lo, dec_hi = get_filters(wavelet, flip=True, dtype=images.dtype)
    dec_filt = construct_2d_filters(dec_lo, dec_hi)
    
    # Transpose to NCHW format for conv
    data = jnp.transpose(images, (0, 3, 1, 2))
    num_channels = data.shape[1]
    
    # Replicate filters for each input channel
    # For feature_group_count=C, filter shape: (4*C, 1, H, W)
    # Each channel gets its own set of 4 subband filters
    if num_channels > 1:
        dec_filt = jnp.tile(dec_filt, (num_channels, 1, 1, 1))
    
    # Pad symmetrically
    data = pad_symmetric(data, dec_lo.shape[0])
    
    # Separable 2D convolution with stride 2 (downsampling)
    # Use feature_group_count to apply filters independently per channel
    res = jax.lax.conv_general_dilated(
        lhs=data,
        rhs=dec_filt,
        padding="VALID",
        window_strides=[2, 2],
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        feature_group_count=num_channels,
        precision=jax.lax.Precision("highest"),
    )
    
    # Split into 4 subbands and rearrange
    # res shape: (B, 4*C, H/2, W/2)
    res_ll, res_lh, res_hl, res_hh = jnp.split(res, 4, axis=1)
    
    # Concatenate along channel: (B, 4*C, H/2, W/2)
    wavelets = jnp.concatenate([res_ll, res_lh, res_hl, res_hh], axis=1)
    
    # Transpose back to NHWC: (B, H/2, W/2, 4*C)
    wavelets = jnp.transpose(wavelets, (0, 2, 3, 1))
    
    # Stop gradients - non-differentiable preprocessing
    return jax.lax.stop_gradient(wavelets)


def waverec2_single(
    coeffs: jnp.ndarray,
    wavelet: str = "haar"
) -> jnp.ndarray:
    """Single-level 2D wavelet reconstruction (pure, non-differentiable).
    
    Reconstructs image from wavelet coefficients. This is the inverse of wavedec2_single.
    
    Args:
        coeffs: Wavelet coefficients of shape (B, H, W, C*4) where subbands
                [LL, LH, HL, HH] are concatenated along channel dimension
        wavelet: Wavelet name ('haar', 'db2', 'db4')
        
    Returns:
        Reconstructed images of shape (B, H*2, W*2, C)
        
    Note:
        This function applies stop_gradient internally - use for
        fixed preprocessing, not learnable transforms.
    """
    # Get reconstruction filters matching input dtype (critical for mixed precision!)
    rec_lo, rec_hi = get_filters(wavelet, flip=False, dtype=coeffs.dtype)
    
    # Construct 2D reconstruction filters
    filt_ll = jnp.outer(rec_lo, rec_lo)
    filt_lh = jnp.outer(rec_hi, rec_lo)
    filt_hl = jnp.outer(rec_lo, rec_hi)
    filt_hh = jnp.outer(rec_hi, rec_hi)
    
    # Reshape for convolution: (1, 1, fh, fw)
    filt_ll = filt_ll[jnp.newaxis, jnp.newaxis, :, :]
    filt_lh = filt_lh[jnp.newaxis, jnp.newaxis, :, :]
    filt_hl = filt_hl[jnp.newaxis, jnp.newaxis, :, :]
    filt_hh = filt_hh[jnp.newaxis, jnp.newaxis, :, :]
    
    # Transpose to NCHW format
    data = jnp.transpose(coeffs, (0, 3, 1, 2))
    batch_size, num_channels, height, width = data.shape
    num_output_channels = num_channels // 4
    
    # Reshape to separate the 4 subbands: (B, C, 4, H, W)
    data = data.reshape(batch_size, num_output_channels, 4, height, width)
    
    # Process each output channel independently
    outputs = []
    for c in range(num_output_channels):
        # Get the 4 subbands for this channel
        channel_data = data[:, c, :, :, :]  # (B, 4, H, W)
        coeffs_ll = channel_data[:, 0:1, :, :]  # (B, 1, H, W)
        coeffs_lh = channel_data[:, 1:2, :, :]
        coeffs_hl = channel_data[:, 2:3, :, :]
        coeffs_hh = channel_data[:, 3:4, :, :]
        
        # Use transpose convolution to upsample each subband
        # This does upsampling + filtering in one operation
        rec_ll = jax.lax.conv_transpose(
            lhs=coeffs_ll, rhs=filt_ll, strides=[2, 2], padding="SAME",
            dimension_numbers=("NCHW", "IOHW", "NCHW"),
            transpose_kernel=False, precision=jax.lax.Precision("highest")
        )
        rec_lh = jax.lax.conv_transpose(
            lhs=coeffs_lh, rhs=filt_lh, strides=[2, 2], padding="SAME",
            dimension_numbers=("NCHW", "IOHW", "NCHW"),
            transpose_kernel=False, precision=jax.lax.Precision("highest")
        )
        rec_hl = jax.lax.conv_transpose(
            lhs=coeffs_hl, rhs=filt_hl, strides=[2, 2], padding="SAME",
            dimension_numbers=("NCHW", "IOHW", "NCHW"),
            transpose_kernel=False, precision=jax.lax.Precision("highest")
        )
        rec_hh = jax.lax.conv_transpose(
            lhs=coeffs_hh, rhs=filt_hh, strides=[2, 2], padding="SAME",
            dimension_numbers=("NCHW", "IOHW", "NCHW"),
            transpose_kernel=False, precision=jax.lax.Precision("highest")
        )
        
        # Sum all contributions: (B, 1, 2*H, 2*W)
        reconstructed_channel = rec_ll + rec_lh + rec_hl + rec_hh
        outputs.append(reconstructed_channel)
    
    # Concatenate all output channels: (B, C, 2*H, 2*W)
    result = jnp.concatenate(outputs, axis=1)
    
    # Transpose back to NHWC: (B, 2*H, 2*W, C)
    result = jnp.transpose(result, (0, 2, 3, 1))
    
    # Stop gradients - non-differentiable preprocessing
    return jax.lax.stop_gradient(result)

