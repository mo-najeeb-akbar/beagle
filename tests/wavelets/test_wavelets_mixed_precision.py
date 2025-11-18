"""Tests for wavelet operations with mixed precision dtypes."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as random

from beagle.wavelets import wavedec2, waverec2
from beagle.wavelets.filters import get_filters, construct_2d_filters


def test_get_filters_with_dtype_bfloat16():
    """Test that filters can be cast to bfloat16."""
    lo_f32, hi_f32 = get_filters("haar", dtype=None)
    lo_bf16, hi_bf16 = get_filters("haar", dtype=jnp.bfloat16)
    
    assert lo_f32.dtype == jnp.float32
    assert hi_f32.dtype == jnp.float32
    assert lo_bf16.dtype == jnp.bfloat16
    assert hi_bf16.dtype == jnp.bfloat16
    
    # Values should be close (within bfloat16 precision)
    assert jnp.allclose(lo_f32, lo_bf16.astype(jnp.float32), rtol=1e-2)
    assert jnp.allclose(hi_f32, hi_bf16.astype(jnp.float32), rtol=1e-2)


def test_get_filters_with_dtype_float16():
    """Test that filters can be cast to float16."""
    lo_f32, hi_f32 = get_filters("haar", dtype=None)
    lo_f16, hi_f16 = get_filters("haar", dtype=jnp.float16)
    
    assert lo_f32.dtype == jnp.float32
    assert hi_f32.dtype == jnp.float32
    assert lo_f16.dtype == jnp.float16
    assert hi_f16.dtype == jnp.float16


def test_construct_2d_filters_preserves_dtype():
    """Test that 2D filter construction preserves dtype."""
    # bfloat16
    lo_bf16, hi_bf16 = get_filters("haar", dtype=jnp.bfloat16)
    filt_bf16 = construct_2d_filters(lo_bf16, hi_bf16)
    assert filt_bf16.dtype == jnp.bfloat16
    
    # float16
    lo_f16, hi_f16 = get_filters("haar", dtype=jnp.float16)
    filt_f16 = construct_2d_filters(lo_f16, hi_f16)
    assert filt_f16.dtype == jnp.float16
    
    # float32 (default)
    lo_f32, hi_f32 = get_filters("haar", dtype=None)
    filt_f32 = construct_2d_filters(lo_f32, hi_f32)
    assert filt_f32.dtype == jnp.float32


def test_wavedec2_with_bfloat16_input():
    """Test wavelet decomposition with bfloat16 input."""
    key = random.key(0)
    x_f32 = random.normal(key, (2, 64, 64, 3))
    x_bf16 = x_f32.astype(jnp.bfloat16)
    
    # Should not raise dtype mismatch error
    wavelets_bf16 = wavedec2(x_bf16, wavelet="haar")
    
    # Output should match input dtype
    assert wavelets_bf16.dtype == jnp.bfloat16
    
    # Should have correct shape (4 subbands per channel)
    assert wavelets_bf16.shape == (2, 32, 32, 12)  # 3 channels * 4 subbands


def test_wavedec2_with_float16_input():
    """Test wavelet decomposition with float16 input."""
    key = random.key(1)
    x_f32 = random.normal(key, (2, 64, 64, 1))
    x_f16 = x_f32.astype(jnp.float16)
    
    # Should not raise dtype mismatch error
    wavelets_f16 = wavedec2(x_f16, wavelet="haar")
    
    # Output should match input dtype
    assert wavelets_f16.dtype == jnp.float16
    
    # Should have correct shape
    assert wavelets_f16.shape == (2, 32, 32, 4)


def test_waverec2_with_bfloat16_input():
    """Test wavelet reconstruction with bfloat16 input."""
    key = random.key(2)
    coeffs_f32 = random.normal(key, (2, 32, 32, 4))
    coeffs_bf16 = coeffs_f32.astype(jnp.bfloat16)
    
    # Should not raise dtype mismatch error
    recon_bf16 = waverec2(coeffs_bf16, wavelet="haar")
    
    # Output should match input dtype
    assert recon_bf16.dtype == jnp.bfloat16
    
    # Should have correct shape (upsampled)
    assert recon_bf16.shape == (2, 64, 64, 1)


def test_wavedec2_waverec2_roundtrip_bfloat16():
    """Test decomposition -> reconstruction roundtrip with bfloat16."""
    key = random.key(3)
    x_orig = random.normal(key, (1, 64, 64, 1))
    x_bf16 = x_orig.astype(jnp.bfloat16)
    
    # Decompose
    wavelets = wavedec2(x_bf16, wavelet="haar")
    assert wavelets.dtype == jnp.bfloat16
    
    # Reconstruct
    x_recon = waverec2(wavelets, wavelet="haar")
    assert x_recon.dtype == jnp.bfloat16
    
    # Should be close to original (within bfloat16 precision)
    x_recon_f32 = x_recon.astype(jnp.float32)
    assert jnp.allclose(x_orig, x_recon_f32, rtol=1e-2, atol=1e-2)


def test_wavedec2_waverec2_roundtrip_float16():
    """Test decomposition -> reconstruction roundtrip with float16."""
    key = random.key(4)
    x_orig = random.normal(key, (1, 32, 32, 2))
    x_f16 = x_orig.astype(jnp.float16)
    
    # Decompose
    wavelets = wavedec2(x_f16, wavelet="haar")
    assert wavelets.dtype == jnp.float16
    
    # Reconstruct
    x_recon = waverec2(wavelets, wavelet="haar")
    assert x_recon.dtype == jnp.float16
    
    # Should be close to original (within float16 precision)
    x_recon_f32 = x_recon.astype(jnp.float32)
    assert jnp.allclose(x_orig, x_recon_f32, rtol=1e-2, atol=1e-2)


def test_wavedec2_mixed_precision_vs_float32():
    """Test that mixed precision gives similar results to float32."""
    key = random.key(5)
    x_f32 = random.normal(key, (2, 128, 128, 3))
    x_bf16 = x_f32.astype(jnp.bfloat16)
    
    # Decompose in both precisions
    wavelets_f32 = wavedec2(x_f32, wavelet="haar")
    wavelets_bf16 = wavedec2(x_bf16, wavelet="haar")
    
    # Convert bfloat16 result to float32 for comparison
    wavelets_bf16_f32 = wavelets_bf16.astype(jnp.float32)
    
    # Results should be close (within bfloat16 precision)
    # bfloat16 has ~7 bits of precision, allow ~3% relative error
    # Note: edge values in convolution can have higher relative error
    assert jnp.allclose(wavelets_f32, wavelets_bf16_f32, rtol=0.03, atol=0.01)


def test_db2_filters_with_bfloat16():
    """Test db2 wavelet filters work with bfloat16."""
    key = random.key(6)
    x = random.normal(key, (1, 64, 64, 1)).astype(jnp.bfloat16)
    
    # Should work with db2 wavelet
    wavelets = wavedec2(x, wavelet="db2")
    assert wavelets.dtype == jnp.bfloat16
    # db2 has 4-tap filters, output is slightly larger than Haar
    assert wavelets.shape[0] == 1
    assert wavelets.shape[-1] == 4
    assert 32 <= wavelets.shape[1] <= 34  # Filter length affects size
    
    # Reconstruct
    recon = waverec2(wavelets, wavelet="db2")
    assert recon.dtype == jnp.bfloat16
    # Reconstruction size depends on filter length
    assert recon.shape[0] == 1
    assert recon.shape[-1] == 1
    assert 64 <= recon.shape[1] <= 68


def test_db4_filters_with_float16():
    """Test db4 wavelet filters work with float16."""
    key = random.key(7)
    x = random.normal(key, (1, 64, 64, 1)).astype(jnp.float16)
    
    # Should work with db4 wavelet
    wavelets = wavedec2(x, wavelet="db4")
    assert wavelets.dtype == jnp.float16
    # db4 has 8-tap filters, output is larger than Haar
    assert wavelets.shape[0] == 1
    assert wavelets.shape[-1] == 4
    assert 32 <= wavelets.shape[1] <= 36  # Filter length affects size
    
    # Reconstruct
    recon = waverec2(wavelets, wavelet="db4")
    assert recon.dtype == jnp.float16
    # Reconstruction size depends on filter length
    assert recon.shape[0] == 1
    assert recon.shape[-1] == 1
    assert 64 <= recon.shape[1] <= 72

