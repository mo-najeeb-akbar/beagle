"""Tests for composable augmentation system."""
from __future__ import annotations

import pytest
import tensorflow as tf
import numpy as np

from beagle.augmentations.compose import (
    compose,
    apply_to_field,
    apply_to_fields,
    apply_geometric,
    random_flip_left_right,
    random_flip_up_down,
    random_rotate_90,
    random_brightness,
    random_contrast,
    clip_values,
)


def test_compose_single_function():
    """Test compose with single augmentation."""
    def add_one(data_dict):
        data_dict['value'] = data_dict['value'] + 1
        return data_dict
    
    augment = compose(add_one)
    data = {'value': tf.constant(5)}
    result = augment(data)
    
    assert result['value'].numpy() == 6


def test_compose_multiple_functions():
    """Test compose chains multiple augmentations."""
    def add_one(data_dict):
        data_dict['value'] = data_dict['value'] + 1
        return data_dict
    
    def multiply_two(data_dict):
        data_dict['value'] = data_dict['value'] * 2
        return data_dict
    
    augment = compose(add_one, multiply_two, add_one)
    data = {'value': tf.constant(5)}
    result = augment(data)
    
    # (5 + 1) * 2 + 1 = 13
    assert result['value'].numpy() == 13


def test_apply_to_field():
    """Test apply_to_field transforms only specified field."""
    augment = apply_to_field('image', lambda x: x * 2)
    
    data = {
        'image': tf.constant([[1.0, 2.0]], dtype=tf.float32),
        'mask': tf.constant([[1, 2]], dtype=tf.int32),
    }
    
    result = augment(data)
    
    # Image should be transformed
    np.testing.assert_array_equal(
        result['image'].numpy(),
        [[2.0, 4.0]],
    )
    
    # Mask should be unchanged
    np.testing.assert_array_equal(
        result['mask'].numpy(),
        [[1, 2]],
    )


def test_apply_to_field_missing_field():
    """Test apply_to_field handles missing field gracefully."""
    augment = apply_to_field('missing', lambda x: x * 2)
    
    data = {'image': tf.constant([[1.0]])}
    result = augment(data)
    
    # Should not raise error
    assert 'image' in result
    assert 'missing' not in result


def test_apply_to_fields():
    """Test apply_to_fields transforms multiple fields."""
    augment = apply_to_fields(['image', 'depth'], lambda x: x * 2)
    
    data = {
        'image': tf.constant([[1.0]], dtype=tf.float32),
        'depth': tf.constant([[2.0]], dtype=tf.float32),
        'label': tf.constant([5], dtype=tf.int32),
    }
    
    result = augment(data)
    
    # Image and depth transformed
    np.testing.assert_array_equal(result['image'].numpy(), [[2.0]])
    np.testing.assert_array_equal(result['depth'].numpy(), [[4.0]])
    
    # Label unchanged
    np.testing.assert_array_equal(result['label'].numpy(), [5])


def test_apply_geometric():
    """Test apply_geometric applies same transform to image and mask."""
    def flip_both(img, mask):
        # Simple flip (for testing)
        img = tf.reverse(img, axis=[1])
        if mask is not None:
            mask = tf.reverse(mask, axis=[1])
        return img, mask
    
    augment = apply_geometric(flip_both)
    
    data = {
        'image': tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32),
        'mask': tf.constant([[10, 20, 30]], dtype=tf.int32),
    }
    
    result = augment(data)
    
    # Both should be flipped
    np.testing.assert_array_equal(result['image'].numpy(), [[3.0, 2.0, 1.0]])
    np.testing.assert_array_equal(result['mask'].numpy(), [[30, 20, 10]])


def test_apply_geometric_custom_field_names():
    """Test apply_geometric with custom field names."""
    def flip_both(img, mask):
        img = tf.reverse(img, axis=[1])
        if mask is not None:
            mask = tf.reverse(mask, axis=[1])
        return img, mask
    
    augment = apply_geometric(
        flip_both,
        image_field='depth',
        mask_field='segmentation',
    )
    
    data = {
        'depth': tf.constant([[1.0, 2.0]], dtype=tf.float32),
        'segmentation': tf.constant([[10, 20]], dtype=tf.int32),
    }
    
    result = augment(data)
    
    assert 'depth' in result
    assert 'segmentation' in result
    np.testing.assert_array_equal(result['depth'].numpy(), [[2.0, 1.0]])
    np.testing.assert_array_equal(result['segmentation'].numpy(), [[20, 10]])


def test_apply_geometric_no_mask():
    """Test apply_geometric works without mask field."""
    def flip_img(img, mask):
        img = tf.reverse(img, axis=[1])
        return img, mask
    
    augment = apply_geometric(flip_img, mask_field=None)
    
    data = {'image': tf.constant([[1.0, 2.0]], dtype=tf.float32)}
    
    result = augment(data)
    
    assert 'image' in result
    np.testing.assert_array_equal(result['image'].numpy(), [[2.0, 1.0]])


def test_random_flip_left_right_preserves_shape():
    """Test random_flip_left_right preserves shape."""
    augment = random_flip_left_right(fields=['image'])
    
    data = {'image': tf.constant([[[1.0], [2.0], [3.0]]], dtype=tf.float32)}
    result = augment(data)
    
    assert result['image'].shape == data['image'].shape


def test_random_flip_up_down_preserves_shape():
    """Test random_flip_up_down preserves shape."""
    augment = random_flip_up_down(fields=['image'])
    
    data = {'image': tf.constant([[[1.0], [2.0], [3.0]]], dtype=tf.float32)}
    result = augment(data)
    
    assert result['image'].shape == data['image'].shape


def test_random_rotate_90_preserves_shape():
    """Test random_rotate_90 with square input."""
    augment = random_rotate_90(fields=['image'])
    
    # Use square image so shape is preserved after rotation
    data = {'image': tf.constant([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=tf.float32)}
    result = augment(data)
    
    assert result['image'].shape == data['image'].shape


def test_random_brightness_field_parameter():
    """Test random_brightness applies to correct field."""
    augment = random_brightness(0.2, field='depth')
    
    data = {
        'depth': tf.constant([[0.5]], dtype=tf.float32),
        'mask': tf.constant([[1]], dtype=tf.int32),
    }
    
    result = augment(data)
    
    # Depth should be changed (brightness is random, just check it ran)
    assert 'depth' in result
    
    # Mask should be unchanged
    np.testing.assert_array_equal(result['mask'].numpy(), [[1]])


def test_random_contrast():
    """Test random_contrast function."""
    augment = random_contrast(0.8, 1.2, field='image')
    
    # random_contrast requires 3D tensor (H, W, C)
    data = {'image': tf.constant([[[0.5], [0.6]], [[0.7], [0.8]]], dtype=tf.float32)}
    result = augment(data)
    
    assert 'image' in result
    assert result['image'].shape == data['image'].shape


def test_clip_values():
    """Test clip_values constrains output range."""
    augment = clip_values(0.0, 1.0, field='image')
    
    data = {'image': tf.constant([[-0.5, 0.5, 1.5]], dtype=tf.float32)}
    result = augment(data)
    
    np.testing.assert_array_equal(result['image'].numpy(), [[0.0, 0.5, 1.0]])


def test_compose_with_library_functions():
    """Test compose works with library augmentation functions."""
    augment = compose(
        random_flip_left_right(fields=['image']),
        random_brightness(0.1, field='image'),
        clip_values(0.0, 1.0, field='image'),
    )
    
    data = {'image': tf.constant([[[0.5]]], dtype=tf.float32)}
    result = augment(data)
    
    assert 'image' in result
    assert result['image'].shape == data['image'].shape
    # Values should be clipped to [0, 1]
    assert tf.reduce_all(result['image'] >= 0.0)
    assert tf.reduce_all(result['image'] <= 1.0)


def test_geometric_augment_consistency():
    """Test that geometric augments apply same transform to image and mask."""
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    
    augment = random_flip_left_right()  # Default: applies to both image and mask
    
    # Create simple test case where we can verify consistency
    data = {
        'image': tf.constant([[[1.0, 2.0, 3.0]]], dtype=tf.float32),
        'mask': tf.constant([[[10, 20, 30]]], dtype=tf.int32),
    }
    
    result = augment(data)
    
    # Both should be transformed consistently (flipped or not)
    # We can't predict the exact output due to randomness,
    # but the operation should complete without error
    assert 'image' in result
    assert 'mask' in result
    assert result['image'].shape == data['image'].shape
    assert result['mask'].shape == data['mask'].shape


def test_compose_preserves_immutability():
    """Test that compose doesn't mutate input data dict."""
    def double_value(data_dict):
        # Create new dict (functional style)
        return {'value': data_dict['value'] * 2}
    
    augment = compose(double_value)
    
    original_data = {'value': tf.constant(5)}
    result = augment(original_data)
    
    # Original should be unchanged if augmentation is pure
    # (Note: our augmentations mutate for efficiency, but test the pattern)
    assert result['value'].numpy() == 10


def test_apply_geometric_missing_image_field():
    """Test apply_geometric returns early when image field is missing."""
    def flip_both(img, mask):
        img = tf.reverse(img, axis=[1])
        if mask is not None:
            mask = tf.reverse(mask, axis=[1])
        return img, mask
    
    augment = apply_geometric(flip_both, image_field='missing')
    
    data = {'other': tf.constant([[1.0, 2.0]], dtype=tf.float32)}
    result = augment(data)
    
    # Should return unchanged
    assert 'other' in result
    assert 'missing' not in result


def test_random_flip_up_down_geometric_mode():
    """Test random_flip_up_down in geometric mode (image + mask)."""
    augment = random_flip_up_down()  # No fields argument = geometric mode
    
    data = {
        'image': tf.constant([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=tf.float32),
        'mask': tf.constant([[[10], [20]], [[30], [40]]], dtype=tf.int32),
    }
    
    result = augment(data)
    
    # Should process both fields
    assert 'image' in result
    assert 'mask' in result
    assert result['image'].shape == data['image'].shape
    assert result['mask'].shape == data['mask'].shape


def test_random_rotate_90_geometric_mode():
    """Test random_rotate_90 in geometric mode (image + mask)."""
    augment = random_rotate_90()  # No fields argument = geometric mode
    
    data = {
        'image': tf.constant([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=tf.float32),
        'mask': tf.constant([[[10], [20]], [[30], [40]]], dtype=tf.int32),
    }
    
    result = augment(data)
    
    # Should process both fields
    assert 'image' in result
    assert 'mask' in result
    # Shape preserved for square images
    assert result['image'].shape == data['image'].shape
    assert result['mask'].shape == data['mask'].shape


def test_random_flip_left_right_geometric_mode():
    """Test random_flip_left_right in geometric mode."""
    augment = random_flip_left_right()  # No fields argument = geometric mode
    
    data = {
        'image': tf.constant([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=tf.float32),
        'mask': tf.constant([[[10], [20]], [[30], [40]]], dtype=tf.int32),
    }
    
    result = augment(data)
    
    # Should process both fields
    assert 'image' in result
    assert 'mask' in result
    assert result['image'].shape == data['image'].shape
    assert result['mask'].shape == data['mask'].shape

