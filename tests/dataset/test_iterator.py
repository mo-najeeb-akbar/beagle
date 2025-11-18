"""Tests for iterator module."""
from __future__ import annotations

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path

from beagle.dataset.iterator import (
    compute_num_crops,
    make_default_image_parser,
    create_iterator,
    create_tfrecord_iterator,
    compute_welford_stats,
)
from beagle.dataset.preprocessing import FieldConfig, FieldType


@pytest.fixture
def sample_tfrecord(tmp_path: Path) -> str:
    """Create a sample TFRecord file for testing."""
    tfrecord_path = tmp_path / "test.tfrecord"
    
    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        for _ in range(10):
            image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img_str = tf.io.encode_png(image).numpy()
            
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_str]))}
                )
            )
            writer.write(example.SerializeToString())
    
    return str(tfrecord_path)


def test_compute_num_crops_basic():
    """Test basic crop count computation."""
    # 100x100 image, 32x32 crops, stride 32
    num_crops = compute_num_crops(100, 100, 32, 32)
    # (100 - 32) // 32 + 1 = 3 rows
    # (100 - 32) // 32 + 1 = 3 cols
    assert num_crops == 9


def test_compute_num_crops_overlapping():
    """Test crop count with overlapping crops."""
    # 100x100 image, 32x32 crops, stride 16 (50% overlap)
    num_crops = compute_num_crops(100, 100, 32, 16)
    # (100 - 32) // 16 + 1 = 5 rows
    # (100 - 32) // 16 + 1 = 5 cols
    assert num_crops == 25


def test_compute_num_crops_exact_fit():
    """Test crop count when crops fit exactly."""
    # 64x64 image, 32x32 crops, stride 32
    num_crops = compute_num_crops(64, 64, 32, 32)
    # (64 - 32) // 32 + 1 = 2 rows
    # (64 - 32) // 32 + 1 = 2 cols
    assert num_crops == 4


def test_compute_num_crops_rectangular():
    """Test crop count for rectangular images."""
    # 128x64 image, 32x32 crops, stride 32
    num_crops = compute_num_crops(128, 64, 32, 32)
    # (128 - 32) // 32 + 1 = 4 rows
    # (64 - 32) // 32 + 1 = 2 cols
    assert num_crops == 8


def test_make_default_image_parser_grayscale():
    """Test default parser with grayscale conversion."""
    parser = make_default_image_parser(grayscale=True)
    
    # Create test image
    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img_str = tf.io.encode_png(image).numpy()
    
    example = tf.train.Example(
        features=tf.train.Features(
            feature={'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_str]))}
        )
    )
    
    serialized = example.SerializeToString()
    parsed = parser(tf.constant(serialized))
    
    assert 'image' in parsed
    assert isinstance(parsed['image'], tf.Tensor)
    assert parsed['image'].shape[-1] == 1  # Grayscale
    assert parsed['image'].dtype == tf.float32


def test_make_default_image_parser_color():
    """Test default parser with color images."""
    parser = make_default_image_parser(grayscale=False)
    
    # Create test image
    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img_str = tf.io.encode_png(image).numpy()
    
    example = tf.train.Example(
        features=tf.train.Features(
            feature={'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_str]))}
        )
    )
    
    serialized = example.SerializeToString()
    parsed = parser(tf.constant(serialized))
    
    assert 'image' in parsed
    assert parsed['image'].shape[-1] == 3  # RGB
    assert parsed['image'].dtype == tf.float32


def test_make_default_image_parser_normalization():
    """Test that parser normalizes to [0, 1] range."""
    parser = make_default_image_parser(grayscale=True)
    
    # Create test image with known values
    image = np.ones((64, 64, 3), dtype=np.uint8) * 128
    img_str = tf.io.encode_png(image).numpy()
    
    example = tf.train.Example(
        features=tf.train.Features(
            feature={'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_str]))}
        )
    )
    
    serialized = example.SerializeToString()
    parsed = parser(tf.constant(serialized))
    
    # Should be normalized to [0, 1]
    assert tf.reduce_min(parsed['image']).numpy() >= 0.0
    assert tf.reduce_max(parsed['image']).numpy() <= 1.0


def test_create_tfrecord_iterator_basic(sample_tfrecord: str):
    """Test basic iterator creation."""
    iterator, n_batches = create_tfrecord_iterator(
        sample_tfrecord,
        batch_size=2,
        shuffle=False,
        repeat=False,
    )
    
    # Should have 10 samples / 2 batch_size = 5 batches
    assert n_batches == 5
    
    # Get first batch
    batch = next(iterator)
    assert 'image' in batch
    assert batch['image'].shape[0] == 2  # Batch size


def test_create_tfrecord_iterator_with_augmentation(sample_tfrecord: str):
    """Test iterator with augmentation function."""
    def augment_fn(data_dict):
        img = data_dict['image']
        img = tf.image.flip_left_right(img)
        data_dict['image'] = img
        return data_dict
    
    iterator, n_batches = create_tfrecord_iterator(
        sample_tfrecord,
        batch_size=2,
        augment_fn=augment_fn,
        shuffle=False,
        repeat=False,
    )
    
    assert n_batches == 5
    batch = next(iterator)
    assert 'image' in batch


def test_create_iterator_basic(sample_tfrecord: str):
    """Test unified iterator creation."""
    iterator, n_batches = create_iterator(
        sample_tfrecord,
        batch_size=2,
        shuffle=False,
        repeat=False,
    )
    
    assert n_batches == 5
    batch = next(iterator)
    assert 'image' in batch


def test_create_iterator_with_field_configs(sample_tfrecord: str):
    """Test iterator with custom field configs."""
    field_configs = {
        'image': FieldConfig(
            name='image',
            field_type=FieldType.IMAGE,
            standardize=True,
            stats=(0.5, 0.2),  # Precomputed stats
        )
    }
    
    iterator, n_batches = create_iterator(
        sample_tfrecord,
        batch_size=2,
        field_configs=field_configs,
        shuffle=False,
        repeat=False,
    )
    
    assert n_batches == 5
    batch = next(iterator)
    assert 'image' in batch


def test_create_iterator_with_custom_parser(tmp_path: Path):
    """Test iterator with custom parser."""
    # Create TFRecord with custom data
    tfrecord_path = tmp_path / "custom.tfrecord"
    
    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        for i in range(5):
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'value': tf.train.Feature(float_list=tf.train.FloatList(value=[float(i)]))
                    }
                )
            )
            writer.write(example.SerializeToString())
    
    def custom_parser(example_proto):
        parsed = tf.io.parse_single_example(
            example_proto,
            {'value': tf.io.FixedLenFeature([1], tf.float32)}
        )
        return parsed
    
    iterator, n_batches = create_iterator(
        str(tfrecord_path),
        batch_size=2,
        parser=custom_parser,
        field_configs={},  # No standardization
        shuffle=False,
        repeat=False,
    )
    
    assert n_batches == 2  # 5 samples / 2 batch_size = 2 (drops last incomplete batch)
    batch = next(iterator)
    assert 'value' in batch


def test_create_iterator_no_files_raises_error():
    """Test that iterator raises error when no files found."""
    with pytest.raises(ValueError, match="No files found"):
        create_iterator(
            "/nonexistent/path/*.tfrecord",
            batch_size=2,
        )


def test_create_iterator_crop_size_without_stride_raises_error(sample_tfrecord: str):
    """Test that crop_size without stride raises error."""
    with pytest.raises(ValueError, match="stride must be provided"):
        create_iterator(
            sample_tfrecord,
            batch_size=2,
            crop_size=32,
            stride=None,
        )


def test_compute_welford_stats_basic(sample_tfrecord: str):
    """Test Welford's algorithm for computing stats."""
    mean, std = compute_welford_stats([sample_tfrecord])
    
    # Should return reasonable values for normalized images
    assert 0.0 <= mean <= 1.0
    assert std > 0.0


def test_compute_welford_stats_with_custom_parser(tmp_path: Path):
    """Test Welford stats with custom parser and field name."""
    # Create TFRecord with custom data
    tfrecord_path = tmp_path / "custom.tfrecord"
    
    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        for i in range(10):
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'data': tf.train.Feature(float_list=tf.train.FloatList(value=[1.0, 2.0, 3.0]))
                    }
                )
            )
            writer.write(example.SerializeToString())
    
    def custom_parser(example_proto):
        parsed = tf.io.parse_single_example(
            example_proto,
            {'data': tf.io.FixedLenFeature([3], tf.float32)}
        )
        return {'data': tf.reshape(parsed['data'], [1, 1, 3])}
    
    mean, std = compute_welford_stats(
        [str(tfrecord_path)],
        parser=custom_parser,
        field_name='data',
    )
    
    # Mean should be close to 2.0 (mean of 1, 2, 3)
    assert 1.9 < mean < 2.1
    assert std > 0.0

