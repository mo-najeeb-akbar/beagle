"""Tests for flexible preprocessing system."""
from __future__ import annotations

import pytest
import tensorflow as tf
import numpy as np

from beagle.dataset.preprocessing import (
    FieldConfig,
    FieldType,
    create_standardize_fn,
)


def test_field_config_is_frozen():
    """Test that FieldConfig is immutable."""
    config = FieldConfig('image', FieldType.IMAGE)
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        config.standardize = False


def test_field_config_defaults():
    """Test FieldConfig default values."""
    config = FieldConfig('test', FieldType.IMAGE)
    assert config.name == 'test'
    assert config.field_type == FieldType.IMAGE
    assert config.standardize is True
    assert config.stats is None
    assert config.dtype == tf.float32


def test_create_standardize_fn_with_stats():
    """Test standardization function with precomputed stats."""
    configs = {
        'image': FieldConfig(
            'image',
            FieldType.IMAGE,
            standardize=True,
            stats=(0.5, 0.2),
        ),
    }
    
    standardize = create_standardize_fn(configs)
    
    # Create test data
    img = tf.constant([[1.0, 0.5], [0.3, 0.7]], dtype=tf.float32)
    data = {'image': img}
    
    result = standardize(data)
    
    # Check standardization applied
    assert 'image' in result
    expected = (img - 0.5) / (0.2 + 1e-8)
    np.testing.assert_allclose(
        result['image'].numpy(),
        expected.numpy(),
        rtol=1e-5,
    )


def test_create_standardize_fn_no_standardization():
    """Test that fields marked as no-standardize are unchanged."""
    configs = {
        'mask': FieldConfig(
            'mask',
            FieldType.MASK,
            standardize=False,
        ),
    }
    
    standardize = create_standardize_fn(configs)
    
    # Create test data
    mask = tf.constant([[1, 0], [2, 3]], dtype=tf.int32)
    data = {'mask': mask}
    
    result = standardize(data)
    
    # Check mask unchanged
    assert 'mask' in result
    np.testing.assert_array_equal(result['mask'].numpy(), mask.numpy())


def test_create_standardize_fn_multiple_fields():
    """Test standardization with multiple fields of different types."""
    configs = {
        'image': FieldConfig(
            'image',
            FieldType.IMAGE,
            standardize=True,
            stats=(0.5, 0.2),
        ),
        'mask': FieldConfig(
            'mask',
            FieldType.MASK,
            standardize=False,
            dtype=tf.int32,
        ),
        'bbox': FieldConfig(
            'bbox',
            FieldType.VECTOR,
            standardize=True,
            stats=(0.0, 1.0),
        ),
    }
    
    standardize = create_standardize_fn(configs)
    
    # Create test data
    data = {
        'image': tf.constant([[1.0, 0.5]], dtype=tf.float32),
        'mask': tf.constant([[1, 0]], dtype=tf.int32),
        'bbox': tf.constant([0.5, 0.5, 0.2, 0.3], dtype=tf.float32),
    }
    
    result = standardize(data)
    
    # Check all fields present
    assert 'image' in result
    assert 'mask' in result
    assert 'bbox' in result
    
    # Check image standardized
    expected_img = (data['image'] - 0.5) / (0.2 + 1e-8)
    np.testing.assert_allclose(
        result['image'].numpy(),
        expected_img.numpy(),
        rtol=1e-5,
    )
    
    # Check mask unchanged
    np.testing.assert_array_equal(result['mask'].numpy(), data['mask'].numpy())
    
    # Check bbox standardized
    expected_bbox = data['bbox'] / (1.0 + 1e-8)
    np.testing.assert_allclose(
        result['bbox'].numpy(),
        expected_bbox.numpy(),
        rtol=1e-5,
    )


def test_create_standardize_fn_passthrough_unknown_fields():
    """Test that unknown fields are passed through unchanged."""
    configs = {
        'image': FieldConfig(
            'image',
            FieldType.IMAGE,
            standardize=True,
            stats=(0.5, 0.2),
        ),
    }
    
    standardize = create_standardize_fn(configs)
    
    # Create test data with unknown field
    data = {
        'image': tf.constant([[1.0, 0.5]], dtype=tf.float32),
        'unknown': tf.constant([[42.0]], dtype=tf.float32),
    }
    
    result = standardize(data)
    
    # Check both fields present
    assert 'image' in result
    assert 'unknown' in result
    
    # Check unknown field unchanged
    np.testing.assert_array_equal(
        result['unknown'].numpy(),
        data['unknown'].numpy(),
    )


def test_field_type_enum():
    """Test FieldType enum values."""
    assert FieldType.IMAGE.value == 'image'
    assert FieldType.MASK.value == 'mask'
    assert FieldType.LABEL.value == 'label'
    assert FieldType.VECTOR.value == 'vector'
    assert FieldType.RAW.value == 'raw'


def test_create_standardize_fn_dtype_conversion():
    """Test that standardization respects dtype conversion."""
    configs = {
        'data': FieldConfig(
            'data',
            FieldType.RAW,
            standardize=False,
            dtype=tf.float16,
        ),
    }
    
    standardize = create_standardize_fn(configs)
    
    # Create test data with different dtype
    data = {'data': tf.constant([[1.0, 2.0]], dtype=tf.float32)}
    
    result = standardize(data)
    
    # Check dtype converted
    assert result['data'].dtype == tf.float16


def test_create_standardize_fn_handles_zero_std():
    """Test standardization handles zero std gracefully."""
    configs = {
        'image': FieldConfig(
            'image',
            FieldType.IMAGE,
            standardize=True,
            stats=(0.5, 0.0),  # Zero std!
        ),
    }
    
    standardize = create_standardize_fn(configs)
    
    # Create test data
    img = tf.constant([[1.0, 0.5]], dtype=tf.float32)
    data = {'image': img}
    
    result = standardize(data)
    
    # Should not raise division by zero (uses eps=1e-8)
    assert 'image' in result
    assert not tf.reduce_any(tf.math.is_inf(result['image']))
    assert not tf.reduce_any(tf.math.is_nan(result['image']))

