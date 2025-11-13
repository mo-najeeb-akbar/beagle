"""Tests for beagle.wavelets module.

Validates JAX wavelet implementation against PyWavelets reference.
"""

from __future__ import annotations

import pytest
import numpy as np
import jax.numpy as jnp
import pywt

from beagle.wavelets import wavedec2, waverec2, get_filters, WAVELETS


def test_available_wavelets():
    """Test that wavelets dict contains expected entries."""
    assert 'haar' in WAVELETS
    assert 'db2' in WAVELETS
    assert 'db4' in WAVELETS
    assert len(WAVELETS) == 3


def test_get_filters_haar():
    """Test Haar filter retrieval."""
    dec_lo, dec_hi = get_filters('haar')
    
    # Check shapes
    assert dec_lo.shape == (2,)
    assert dec_hi.shape == (2,)
    
    # Check orthogonality: sum of squares = 1
    assert np.allclose(np.sum(dec_lo**2), 1.0, rtol=1e-5)
    assert np.allclose(np.sum(dec_hi**2), 1.0, rtol=1e-5)


def test_get_filters_db2():
    """Test Daubechies 2 filter retrieval."""
    dec_lo, dec_hi = get_filters('db2')
    
    assert dec_lo.shape == (4,)
    assert dec_hi.shape == (4,)
    assert np.allclose(np.sum(dec_lo**2), 1.0, rtol=1e-5)


def test_get_filters_db4():
    """Test Daubechies 4 filter retrieval."""
    dec_lo, dec_hi = get_filters('db4')
    
    assert dec_lo.shape == (8,)
    assert dec_hi.shape == (8,)
    assert np.allclose(np.sum(dec_lo**2), 1.0, rtol=1e-5)


def test_get_filters_invalid():
    """Test that invalid wavelet name raises error."""
    with pytest.raises(ValueError, match="not supported"):
        get_filters('invalid_wavelet')


def test_wavedec2_shape_single_channel():
    """Test output shape for single channel input."""
    B, H, W, C = 4, 128, 128, 1
    images = jnp.ones((B, H, W, C))
    
    wavelets = wavedec2(images, wavelet='haar')
    
    # Should downsample by 2 and expand channels by 4
    assert wavelets.shape == (B, H//2, W//2, C*4)


def test_wavedec2_shape_multi_channel():
    """Test output shape for multi-channel input."""
    B, H, W, C = 2, 64, 64, 3
    images = jnp.ones((B, H, W, C))
    
    wavelets = wavedec2(images, wavelet='haar')
    
    assert wavelets.shape == (B, H//2, W//2, C*4)


def test_wavedec2_shape_odd_dimensions():
    """Test that odd dimensions are handled correctly."""
    B, H, W, C = 1, 127, 127, 1
    images = jnp.ones((B, H, W, C))
    
    wavelets = wavedec2(images, wavelet='haar')
    
    # Odd dimensions should round up: (127+1)/2 = 64
    assert wavelets.shape == (B, 64, 64, C*4)


def test_wavedec2_vs_pywt_haar_simple():
    """Compare Haar transform with PyWavelets on simple input."""
    # Create test image
    np.random.seed(42)
    image = np.random.randn(64, 64).astype(np.float32)
    
    # PyWavelets transform
    pywt_coeffs = pywt.dwt2(image, 'haar', mode='symmetric')
    pywt_ll, (pywt_lh, pywt_hl, pywt_hh) = pywt_coeffs
    
    # Our transform (add batch and channel dims)
    jax_input = jnp.array(image)[jnp.newaxis, :, :, jnp.newaxis]
    jax_wavelets = wavedec2(jax_input, wavelet='haar')
    
    # Extract subbands
    jax_wavelets_np = np.array(jax_wavelets[0, :, :, :])
    jax_ll = jax_wavelets_np[:, :, 0]
    jax_lh = jax_wavelets_np[:, :, 1]
    jax_hl = jax_wavelets_np[:, :, 2]
    jax_hh = jax_wavelets_np[:, :, 3]
    
    # Compare (allow small numerical differences)
    assert np.allclose(jax_ll, pywt_ll, rtol=1e-4, atol=1e-5), \
        f"LL mismatch: max diff = {np.abs(jax_ll - pywt_ll).max()}"
    assert np.allclose(jax_lh, pywt_lh, rtol=1e-4, atol=1e-5), \
        f"LH mismatch: max diff = {np.abs(jax_lh - pywt_lh).max()}"
    assert np.allclose(jax_hl, pywt_hl, rtol=1e-4, atol=1e-5), \
        f"HL mismatch: max diff = {np.abs(jax_hl - pywt_hl).max()}"
    assert np.allclose(jax_hh, pywt_hh, rtol=1e-4, atol=1e-5), \
        f"HH mismatch: max diff = {np.abs(jax_hh - pywt_hh).max()}"


def test_wavedec2_vs_pywt_db2():
    """Compare db2 transform with PyWavelets."""
    np.random.seed(123)
    image = np.random.randn(128, 128).astype(np.float32)
    
    # PyWavelets
    pywt_coeffs = pywt.dwt2(image, 'db2', mode='symmetric')
    pywt_ll, (pywt_lh, pywt_hl, pywt_hh) = pywt_coeffs
    
    # Our transform
    jax_input = jnp.array(image)[jnp.newaxis, :, :, jnp.newaxis]
    jax_wavelets = wavedec2(jax_input, wavelet='db2')
    jax_wavelets_np = np.array(jax_wavelets[0, :, :, :])
    
    jax_ll = jax_wavelets_np[:, :, 0]
    jax_lh = jax_wavelets_np[:, :, 1]
    jax_hl = jax_wavelets_np[:, :, 2]
    jax_hh = jax_wavelets_np[:, :, 3]
    
    # Compare
    assert np.allclose(jax_ll, pywt_ll, rtol=1e-4, atol=1e-5)
    assert np.allclose(jax_lh, pywt_lh, rtol=1e-4, atol=1e-5)
    assert np.allclose(jax_hl, pywt_hl, rtol=1e-4, atol=1e-5)
    assert np.allclose(jax_hh, pywt_hh, rtol=1e-4, atol=1e-5)


def test_wavedec2_vs_pywt_db4():
    """Compare db4 transform with PyWavelets."""
    np.random.seed(456)
    image = np.random.randn(256, 256).astype(np.float32)
    
    # PyWavelets
    pywt_coeffs = pywt.dwt2(image, 'db4', mode='symmetric')
    pywt_ll, (pywt_lh, pywt_hl, pywt_hh) = pywt_coeffs
    
    # Our transform
    jax_input = jnp.array(image)[jnp.newaxis, :, :, jnp.newaxis]
    jax_wavelets = wavedec2(jax_input, wavelet='db4')
    jax_wavelets_np = np.array(jax_wavelets[0, :, :, :])
    
    jax_ll = jax_wavelets_np[:, :, 0]
    jax_lh = jax_wavelets_np[:, :, 1]
    jax_hl = jax_wavelets_np[:, :, 2]
    jax_hh = jax_wavelets_np[:, :, 3]
    
    # Compare
    assert np.allclose(jax_ll, pywt_ll, rtol=1e-4, atol=1e-5)
    assert np.allclose(jax_lh, pywt_lh, rtol=1e-4, atol=1e-5)
    assert np.allclose(jax_hl, pywt_hl, rtol=1e-4, atol=1e-5)
    assert np.allclose(jax_hh, pywt_hh, rtol=1e-4, atol=1e-5)


def test_wavedec2_batched():
    """Test that batched processing works correctly."""
    np.random.seed(789)
    
    # Create batch of 4 different images
    images = np.random.randn(4, 64, 64, 1).astype(np.float32)
    
    # Transform batch
    jax_wavelets = wavedec2(jnp.array(images), wavelet='haar')
    
    # Transform individually and compare
    for i in range(4):
        single = wavedec2(jnp.array(images[i:i+1]), wavelet='haar')
        assert np.allclose(jax_wavelets[i], single[0], rtol=1e-6)


def test_wavedec2_constant_image():
    """Test transform on constant image."""
    # Constant image should have zero high-frequency components
    constant = jnp.ones((1, 64, 64, 1)) * 5.0
    
    wavelets = wavedec2(constant, wavelet='haar')
    wavelets_np = np.array(wavelets[0, :, :, :])
    
    # LL should be non-zero (scaled by filter)
    assert np.abs(wavelets_np[:, :, 0]).mean() > 0
    
    # LH, HL, HH should be near zero
    assert np.allclose(wavelets_np[:, :, 1], 0.0, atol=1e-5)
    assert np.allclose(wavelets_np[:, :, 2], 0.0, atol=1e-5)
    assert np.allclose(wavelets_np[:, :, 3], 0.0, atol=1e-5)


def test_wavedec2_horizontal_edge():
    """Test transform on horizontal edge matches PyWavelets."""
    # Image with horizontal edge
    image = np.zeros((64, 64), dtype=np.float32)
    image[:32, :] = 1.0  # Top half white
    
    # PyWavelets
    pywt_coeffs = pywt.dwt2(image, 'haar', mode='symmetric')
    pywt_ll, (pywt_lh, pywt_hl, pywt_hh) = pywt_coeffs
    
    # Our implementation
    jax_input = jnp.array(image)[jnp.newaxis, :, :, jnp.newaxis]
    wavelets = wavedec2(jax_input, wavelet='haar')
    wavelets_np = np.array(wavelets[0, :, :, :])
    
    # Compare all subbands
    assert np.allclose(wavelets_np[:, :, 0], pywt_ll, rtol=1e-4, atol=1e-5)
    assert np.allclose(wavelets_np[:, :, 1], pywt_lh, rtol=1e-4, atol=1e-5)
    assert np.allclose(wavelets_np[:, :, 2], pywt_hl, rtol=1e-4, atol=1e-5)
    assert np.allclose(wavelets_np[:, :, 3], pywt_hh, rtol=1e-4, atol=1e-5)


def test_wavedec2_vertical_edge():
    """Test transform on vertical edge matches PyWavelets."""
    # Image with vertical edge
    image = np.zeros((64, 64), dtype=np.float32)
    image[:, :32] = 1.0  # Left half white
    
    # PyWavelets
    pywt_coeffs = pywt.dwt2(image, 'haar', mode='symmetric')
    pywt_ll, (pywt_lh, pywt_hl, pywt_hh) = pywt_coeffs
    
    # Our implementation
    jax_input = jnp.array(image)[jnp.newaxis, :, :, jnp.newaxis]
    wavelets = wavedec2(jax_input, wavelet='haar')
    wavelets_np = np.array(wavelets[0, :, :, :])
    
    # Compare all subbands
    assert np.allclose(wavelets_np[:, :, 0], pywt_ll, rtol=1e-4, atol=1e-5)
    assert np.allclose(wavelets_np[:, :, 1], pywt_lh, rtol=1e-4, atol=1e-5)
    assert np.allclose(wavelets_np[:, :, 2], pywt_hl, rtol=1e-4, atol=1e-5)
    assert np.allclose(wavelets_np[:, :, 3], pywt_hh, rtol=1e-4, atol=1e-5)


def test_wavedec2_non_differentiable():
    """Test that gradients are stopped."""
    import jax
    
    def forward(images):
        return wavedec2(images, wavelet='haar').sum()
    
    images = jnp.ones((1, 64, 64, 1))
    
    # This should not raise an error, but gradients should be zero
    grad_fn = jax.grad(forward)
    grads = grad_fn(images)
    
    # Gradients should be stopped (all zeros)
    assert np.allclose(grads, 0.0)


def test_wavedec2_jit_compatible():
    """Test that transform works with JIT compilation."""
    import jax
    
    @jax.jit
    def jitted_transform(images):
        return wavedec2(images, wavelet='haar')
    
    images = jnp.ones((2, 64, 64, 1))
    wavelets = jitted_transform(images)
    
    assert wavelets.shape == (2, 32, 32, 4)


@pytest.mark.parametrize("wavelet", ['haar', 'db2', 'db4'])
def test_wavedec2_all_wavelets_vs_pywt(wavelet):
    """Parametrized test comparing all wavelets to PyWavelets."""
    np.random.seed(42)
    image = np.random.randn(128, 128).astype(np.float32)
    
    # PyWavelets
    pywt_coeffs = pywt.dwt2(image, wavelet, mode='symmetric')
    pywt_ll, (pywt_lh, pywt_hl, pywt_hh) = pywt_coeffs
    
    # Our transform
    jax_input = jnp.array(image)[jnp.newaxis, :, :, jnp.newaxis]
    jax_wavelets = wavedec2(jax_input, wavelet=wavelet)
    jax_wavelets_np = np.array(jax_wavelets[0, :, :, :])
    
    # Extract and compare
    jax_ll = jax_wavelets_np[:, :, 0]
    jax_lh = jax_wavelets_np[:, :, 1]
    jax_hl = jax_wavelets_np[:, :, 2]
    jax_hh = jax_wavelets_np[:, :, 3]
    
    assert np.allclose(jax_ll, pywt_ll, rtol=1e-4, atol=1e-5), \
        f"{wavelet} LL mismatch"
    assert np.allclose(jax_lh, pywt_lh, rtol=1e-4, atol=1e-5), \
        f"{wavelet} LH mismatch"
    assert np.allclose(jax_hl, pywt_hl, rtol=1e-4, atol=1e-5), \
        f"{wavelet} HL mismatch"
    assert np.allclose(jax_hh, pywt_hh, rtol=1e-4, atol=1e-5), \
        f"{wavelet} HH mismatch"


@pytest.mark.parametrize("size", [(64, 64), (128, 128), (256, 256), (127, 127)])
def test_wavedec2_various_sizes(size):
    """Test transform on various input sizes."""
    H, W = size
    np.random.seed(42)
    image = np.random.randn(H, W).astype(np.float32)
    
    jax_input = jnp.array(image)[jnp.newaxis, :, :, jnp.newaxis]
    wavelets = wavedec2(jax_input, wavelet='haar')
    
    # Check output size is approximately half
    expected_h = (H + 1) // 2
    expected_w = (W + 1) // 2
    assert wavelets.shape == (1, expected_h, expected_w, 4)


@pytest.mark.parametrize("wavelet", ["haar"])
def test_waverec2_inverts_wavedec2(wavelet):
    """Test that waverec2 correctly inverts wavedec2."""
    np.random.seed(42)
    image = np.random.randn(64, 64).astype(np.float32)
    
    # Add batch and channel dimensions
    jax_input = jnp.array(image)[jnp.newaxis, :, :, jnp.newaxis]
    
    # Decompose and reconstruct
    coeffs = wavedec2(jax_input, wavelet=wavelet)
    reconstructed = waverec2(coeffs, wavelet=wavelet)
    
    # Check shape matches
    assert reconstructed.shape == jax_input.shape
    
    # Check reconstruction error is small
    mse = np.mean((np.array(reconstructed) - np.array(jax_input))**2)
    assert mse < 0.01, f"{wavelet} reconstruction error too large: {mse}"


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_waverec2_batched(batch_size):
    """Test reconstruction with different batch sizes."""
    np.random.seed(42)
    images = np.random.randn(batch_size, 64, 64, 1).astype(np.float32)
    jax_input = jnp.array(images)
    
    coeffs = wavedec2(jax_input, wavelet='haar')
    reconstructed = waverec2(coeffs, wavelet='haar')
    
    assert reconstructed.shape == jax_input.shape
    mse = np.mean((np.array(reconstructed) - np.array(jax_input))**2)
    assert mse < 0.01


def test_waverec2_multi_channel():
    """Test reconstruction with multi-channel images."""
    np.random.seed(42)
    # 3-channel image (like RGB)
    images = np.random.randn(2, 64, 64, 3).astype(np.float32)
    jax_input = jnp.array(images)
    
    coeffs = wavedec2(jax_input, wavelet='haar')
    # Coeffs should have 12 channels (3 * 4 subbands)
    assert coeffs.shape == (2, 32, 32, 12)
    
    reconstructed = waverec2(coeffs, wavelet='haar')
    
    assert reconstructed.shape == jax_input.shape
    mse = np.mean((np.array(reconstructed) - np.array(jax_input))**2)
    assert mse < 0.01

