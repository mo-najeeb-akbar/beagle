"""Tests for augmentation pipeline."""
from __future__ import annotations

import pytest
import numpy as np

# Only run if albumentations is available
pytest.importorskip("albumentations")

from beagle.augmentations import (
    create_transform,
    apply_transform,
    AugmentConfig,
    MINIMAL_AUGMENT,
    MODERATE_AUGMENT,
    HEAVY_AUGMENT,
)


def test_minimal_augment_config():
    """Test minimal augmentation config has expected values."""
    assert MINIMAL_AUGMENT.flip_horizontal is True
    assert MINIMAL_AUGMENT.flip_vertical is True
    assert MINIMAL_AUGMENT.rotate_90 is True
    assert MINIMAL_AUGMENT.rotation_limit == 0.0


def test_augment_config_is_frozen():
    """Test that AugmentConfig is immutable."""
    config = AugmentConfig(flip_horizontal=True)
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        config.flip_horizontal = False


def test_create_transform_with_minimal():
    """Test creating transform with minimal augmentation."""
    transform = create_transform(MINIMAL_AUGMENT)
    assert transform is not None


def test_apply_transform_preserves_shape():
    """Test that augmentation preserves image shape."""
    transform = create_transform(MINIMAL_AUGMENT)
    image = np.random.rand(64, 64, 3).astype(np.float32)
    
    result = apply_transform(transform, image)
    
    assert 'image' in result
    assert result['image'].shape == image.shape


def test_apply_transform_output_range():
    """Test that augmented images stay in [0, 1] range."""
    transform = create_transform(MODERATE_AUGMENT)
    image = np.random.rand(64, 64, 3).astype(np.float32)
    
    result = apply_transform(transform, image)
    
    assert result['image'].min() >= 0.0
    assert result['image'].max() <= 1.0


def test_apply_transform_with_mask():
    """Test augmentation with mask for segmentation."""
    transform = create_transform(MINIMAL_AUGMENT)
    image = np.random.rand(64, 64, 3).astype(np.float32)
    mask = np.random.rand(64, 64).astype(np.float32)
    
    result = apply_transform(transform, image, mask=mask)
    
    assert 'image' in result
    assert 'mask' in result
    assert result['image'].shape == image.shape
    assert result['mask'].shape == mask.shape


def test_apply_transform_mask_preserves_binary():
    """Test that binary masks remain roughly binary after augmentation."""
    transform = create_transform(MINIMAL_AUGMENT)
    image = np.random.rand(64, 64, 3).astype(np.float32)
    mask = (np.random.rand(64, 64) > 0.5).astype(np.float32)
    
    result = apply_transform(transform, image, mask=mask)
    
    # After geometric transforms, mask should still be mostly binary
    unique_vals = np.unique(result['mask'])
    assert len(unique_vals) <= 10  # Allow some interpolation artifacts


def test_empty_augmentation():
    """Test that empty config creates identity-like transform."""
    config = AugmentConfig(
        flip_horizontal=False,
        flip_vertical=False,
        rotate_90=False,
    )
    transform = create_transform(config)
    image = np.random.rand(64, 64, 3).astype(np.float32)
    
    result = apply_transform(transform, image)
    
    # Should be very close to original (may have tiny numerical differences)
    assert np.allclose(result['image'], image, atol=0.01)


def test_moderate_augment_changes_image():
    """Test that moderate augmentation actually changes the image."""
    transform = create_transform(MODERATE_AUGMENT)
    image = np.random.rand(64, 64, 3).astype(np.float32)
    
    # Apply multiple times, at least one should be different
    results = [apply_transform(transform, image)['image'] for _ in range(10)]
    
    # At least one result should differ from original
    any_different = any(not np.allclose(r, image, atol=0.01) for r in results)
    assert any_different


def test_custom_config():
    """Test creating transform with custom config."""
    config = AugmentConfig(
        flip_horizontal=True,
        rotation_limit=10.0,
        brightness_limit=0.1,
        gaussian_noise_var=0.01,
    )
    transform = create_transform(config)
    image = np.random.rand(64, 64, 3).astype(np.float32)
    
    result = apply_transform(transform, image)
    
    assert result['image'].shape == image.shape


def test_grayscale_image():
    """Test augmentation with grayscale image."""
    transform = create_transform(MINIMAL_AUGMENT)
    image = np.random.rand(64, 64, 1).astype(np.float32)
    
    result = apply_transform(transform, image)
    
    assert result['image'].shape == image.shape


def test_different_sizes():
    """Test augmentation with different image sizes."""
    transform = create_transform(MINIMAL_AUGMENT)
    
    for size in [(32, 32), (64, 64), (128, 256)]:
        image = np.random.rand(*size, 3).astype(np.float32)
        result = apply_transform(transform, image)
        assert result['image'].shape == image.shape

