"""Tests for beagle.network.wavelets module (Haar Conv implementations)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beagle.network.wavelets import HaarWaveletConv, HaarWaveletConvTranspose
from beagle.wavelets import wavedec2, waverec2


def test_conv_matches_existing_implementation() -> None:
    """Test that Conv2D-based Haar matches existing wavedec2."""
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (2, 32, 32, 1), dtype=jnp.float32)
    
    coeffs_existing = wavedec2(x, wavelet="haar")
    
    conv_model = HaarWaveletConv()
    params = conv_model.init(key, x)
    coeffs_conv = conv_model.apply(params, x)
    
    np.testing.assert_allclose(
        coeffs_conv,
        coeffs_existing,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Conv2D implementation doesn't match existing"
    )


def test_conv_transpose_matches_existing_implementation() -> None:
    """Test that ConvTranspose2D-based Haar matches existing waverec2."""
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (2, 32, 32, 1), dtype=jnp.float32)
    
    coeffs_existing = wavedec2(x, wavelet="haar")
    reconstructed_existing = waverec2(coeffs_existing, wavelet="haar")
    
    conv_model = HaarWaveletConv()
    params = conv_model.init(key, x)
    coeffs_conv = conv_model.apply(params, x)
    
    conv_transpose_model = HaarWaveletConvTranspose()
    params_transpose = conv_transpose_model.init(key, coeffs_conv)
    reconstructed_conv = conv_transpose_model.apply(params_transpose, coeffs_conv)
    
    np.testing.assert_allclose(
        reconstructed_conv,
        reconstructed_existing,
        rtol=1e-5,
        atol=1e-6,
        err_msg="ConvTranspose2D implementation doesn't match existing"
    )


@pytest.mark.parametrize("shape", [
    (1, 32, 32, 1),
    (2, 64, 64, 1),
    (4, 128, 128, 1),
])
def test_perfect_reconstruction_haar(shape: tuple[int, int, int, int]) -> None:
    """Test that Haar wavelet decomposition and reconstruction are perfect inverses."""
    key = jax.random.PRNGKey(42)
    images = jax.random.normal(key, shape, dtype=jnp.float32)
    
    coeffs = wavedec2(images, wavelet="haar")
    reconstructed = waverec2(coeffs, wavelet="haar")
    
    np.testing.assert_allclose(
        reconstructed,
        images,
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"Perfect reconstruction failed for shape {shape}"
    )


def test_subband_separation() -> None:
    """Test that wavelet transform correctly separates into 4 subbands."""
    key = jax.random.PRNGKey(123)
    images = jax.random.normal(key, (1, 32, 32, 1), dtype=jnp.float32)
    
    coeffs = wavedec2(images, wavelet="haar")
    
    assert coeffs.shape == (1, 16, 16, 4), f"Expected (1, 16, 16, 4), got {coeffs.shape}"
    
    ll, lh, hl, hh = jnp.split(coeffs, 4, axis=-1)
    
    assert ll.shape == (1, 16, 16, 1)
    assert lh.shape == (1, 16, 16, 1)
    assert hl.shape == (1, 16, 16, 1)
    assert hh.shape == (1, 16, 16, 1)


def test_batch_processing() -> None:
    """Test that batch processing works correctly."""
    key = jax.random.PRNGKey(456)
    images = jax.random.normal(key, (8, 32, 32, 1), dtype=jnp.float32)
    
    coeffs = wavedec2(images, wavelet="haar")
    assert coeffs.shape == (8, 16, 16, 4)
    
    reconstructed = waverec2(coeffs, wavelet="haar")
    np.testing.assert_allclose(reconstructed, images, rtol=1e-5, atol=1e-6)


def test_constant_image() -> None:
    """Test with constant image - all energy should go to LL subband."""
    constant_value = 5.0
    images = jnp.ones((1, 32, 32, 1)) * constant_value
    
    coeffs = wavedec2(images, wavelet="haar")
    ll, lh, hl, hh = jnp.split(coeffs, 4, axis=-1)
    
    assert jnp.abs(lh).max() < 1e-5
    assert jnp.abs(hl).max() < 1e-5
    assert jnp.abs(hh).max() < 1e-5
    
    reconstructed = waverec2(coeffs, wavelet="haar")
    np.testing.assert_allclose(reconstructed, images, rtol=1e-5, atol=1e-6)


def test_checkerboard_pattern() -> None:
    """Test with checkerboard - should have high frequency content."""
    size = 8
    pattern = jnp.array([[((i + j) % 2) * 2.0 - 1.0 for j in range(size)] 
                         for i in range(size)], dtype=jnp.float32)
    images = pattern[jnp.newaxis, :, :, jnp.newaxis]
    
    coeffs = wavedec2(images, wavelet="haar")
    ll, lh, hl, hh = jnp.split(coeffs, 4, axis=-1)
    
    assert jnp.abs(hh).max() > 0.1
    
    reconstructed = waverec2(coeffs, wavelet="haar")
    np.testing.assert_allclose(reconstructed, images, rtol=1e-5, atol=1e-6)


def test_energy_preservation() -> None:
    """Test that wavelet transform preserves energy (Parseval's theorem)."""
    key = jax.random.PRNGKey(789)
    images = jax.random.normal(key, (1, 64, 64, 1), dtype=jnp.float32)
    
    energy_spatial = jnp.sum(images ** 2)
    coeffs = wavedec2(images, wavelet="haar")
    energy_wavelet = jnp.sum(coeffs ** 2)
    
    np.testing.assert_allclose(
        energy_wavelet,
        energy_spatial,
        rtol=1e-4,
        err_msg="Energy not preserved"
    )


def test_conv_output_shape() -> None:
    """Test that HaarWaveletConv produces correct output shape."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, 64, 64, 1), dtype=jnp.float32)
    
    model = HaarWaveletConv()
    params = model.init(key, x)
    out = model.apply(params, x)
    
    assert out.shape == (4, 32, 32, 4)


def test_conv_transpose_output_shape() -> None:
    """Test that HaarWaveletConvTranspose produces correct output shape."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, 32, 32, 4), dtype=jnp.float32)
    
    model = HaarWaveletConvTranspose()
    params = model.init(key, x)
    out = model.apply(params, x)
    
    assert out.shape == (4, 64, 64, 1)


def test_conv_round_trip() -> None:
    """Test that Conv -> ConvTranspose is near-perfect inverse."""
    key = jax.random.PRNGKey(999)
    x = jax.random.normal(key, (2, 32, 32, 1), dtype=jnp.float32)
    
    # Forward
    conv_model = HaarWaveletConv()
    params_conv = conv_model.init(key, x)
    coeffs = conv_model.apply(params_conv, x)
    
    # Backward
    conv_transpose_model = HaarWaveletConvTranspose()
    params_transpose = conv_transpose_model.init(key, coeffs)
    reconstructed = conv_transpose_model.apply(params_transpose, coeffs)
    
    np.testing.assert_allclose(
        reconstructed,
        x,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Conv round-trip not near-perfect inverse"
    )

