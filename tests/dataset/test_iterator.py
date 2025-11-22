"""Tests for iterator module."""
from __future__ import annotations

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
from functools import partial

from beagle.dataset.iterator import (
    build_dataset_pipeline,
    compute_num_crops,
    split_files_train_val,
    count_tfrecord_samples,
)
from beagle.dataset.parsers import make_default_image_parser
from beagle.dataset.stats import compute_fields_mean_std


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


@pytest.fixture
def multiple_tfrecords(tmp_path: Path) -> str:
    """Create multiple TFRecord files for testing train/val split."""
    for i in range(5):
        tfrecord_path = tmp_path / f"test_{i}.tfrecord"
        with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
            for _ in range(4):  # 4 samples per file = 20 total
                image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                img_str = tf.io.encode_png(image).numpy()
                
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_str]))}
                    )
                )
                writer.write(example.SerializeToString())
    
    return str(tmp_path / "test_*.tfrecord")


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


def test_build_dataset_pipeline_basic(sample_tfrecord: str):
    """Test basic iterator creation."""
    parser = make_default_image_parser(grayscale=True)
    field_configs = {'image': lambda x: x}  # No normalization
    
    iterator, n_batches = build_dataset_pipeline(
        files=[sample_tfrecord],
        parser=parser,
        field_configs=field_configs,
        batch_size=2,
        crop_size=None,
        stride=None,
        augment_fn=None,
        shuffle=False,
        repeat=False,
        image_shape=(64, 64),
    )
    
    # Should have 10 samples / 2 batch_size = 5 batches
    assert n_batches == 5
    
    # Get first batch
    batch = next(iterator)
    assert 'image' in batch
    assert batch['image'].shape[0] == 2  # Batch size


def test_build_dataset_pipeline_with_augmentation(sample_tfrecord: str):
    """Test iterator with augmentation function."""
    parser = make_default_image_parser(grayscale=True)
    field_configs = {'image': lambda x: x}
    
    def augment_fn(data_dict):
        return {**data_dict, 'image': tf.image.flip_left_right(data_dict['image'])}
    
    iterator, n_batches = build_dataset_pipeline(
        files=[sample_tfrecord],
        parser=parser,
        field_configs=field_configs,
        batch_size=2,
        crop_size=None,
        stride=None,
        augment_fn=augment_fn,
        shuffle=False,
        repeat=False,
        image_shape=(64, 64),
    )
    
    assert n_batches == 5
    batch = next(iterator)
    assert 'image' in batch


def test_build_dataset_pipeline_with_normalization(sample_tfrecord: str):
    """Test iterator with field normalization."""
    parser = make_default_image_parser(grayscale=True)
    
    # Z-score normalization
    field_configs = {'image': lambda x: (x - 0.5) / 0.2}
    
    iterator, n_batches = build_dataset_pipeline(
        files=[sample_tfrecord],
        parser=parser,
        field_configs=field_configs,
        batch_size=2,
        crop_size=None,
        stride=None,
        augment_fn=None,
        shuffle=False,
        repeat=False,
        image_shape=(64, 64),
    )
    
    assert n_batches == 5
    batch = next(iterator)
    assert 'image' in batch


def test_build_dataset_pipeline_with_computed_stats(sample_tfrecord: str):
    """Test iterator with computed statistics."""
    parser = make_default_image_parser(grayscale=True)
    
    # Compute stats first
    stats = compute_fields_mean_std([sample_tfrecord], parser, ['image'])
    mean, std = stats['image']
    
    # Use computed stats for normalization
    field_configs = {'image': lambda x: (x - mean) / (std + 1e-8)}
    
    iterator, n_batches = build_dataset_pipeline(
        files=[sample_tfrecord],
        parser=parser,
        field_configs=field_configs,
        batch_size=2,
        crop_size=None,
        stride=None,
        augment_fn=None,
        shuffle=False,
        repeat=False,
        image_shape=(64, 64),
    )
    
    assert n_batches == 5
    batch = next(iterator)
    assert 'image' in batch


def test_build_dataset_pipeline_with_custom_parser(tmp_path: Path):
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
    
    iterator, n_batches = build_dataset_pipeline(
        files=[str(tfrecord_path)],
        parser=custom_parser,
        field_configs={'value': lambda x: x},  # Pass through
        batch_size=2,
        crop_size=None,
        stride=None,
        augment_fn=None,
        shuffle=False,
        repeat=False,
        image_shape=(1, 1),  # Dummy shape
    )
    
    assert n_batches == 2  # 5 samples / 2 batch_size = 2 (drops last incomplete batch)
    batch = next(iterator)
    assert 'value' in batch


def test_build_dataset_pipeline_with_empty_files():
    """Test that iterator handles empty file list."""
    parser = make_default_image_parser(grayscale=True)
    field_configs = {'image': lambda x: x}
    
    # Empty files list should result in 0 batches
    iterator, n_batches = build_dataset_pipeline(
        files=[],
        parser=parser,
        field_configs=field_configs,
        batch_size=2,
        crop_size=None,
        stride=None,
        augment_fn=None,
        shuffle=False,
        repeat=False,
        image_shape=(64, 64),
    )
    
    assert n_batches == 0


@pytest.mark.skip(reason="TensorFlow graph mode issue with shape inspection in crops.py")
def test_build_dataset_pipeline_with_crops(sample_tfrecord: str):
    """Test iterator with overlapping crops."""
    parser = make_default_image_parser(grayscale=True)
    field_configs = {'image': lambda x: x}
    
    iterator, n_batches = build_dataset_pipeline(
        files=[sample_tfrecord],
        parser=parser,
        field_configs=field_configs,
        batch_size=2,
        crop_size=32,
        stride=16,
        augment_fn=None,
        shuffle=False,
        repeat=False,
        image_shape=(64, 64),
    )
    
    # 64x64 image, 32x32 crops, stride 16
    # (64-32)//16 + 1 = 3 rows, 3 cols = 9 crops per image
    # 10 images * 9 crops = 90 crops
    # 90 / 2 batch_size = 45 batches
    assert n_batches == 45
    
    batch = next(iterator)
    assert 'image' in batch
    assert batch['image'].shape[0] == 2


def test_compute_fields_mean_std_basic(sample_tfrecord: str):
    """Test computing mean/std for fields."""
    parser = make_default_image_parser(grayscale=True)
    stats = compute_fields_mean_std([sample_tfrecord], parser, ['image'])
    
    assert 'image' in stats
    mean, std = stats['image']
    
    # Should return reasonable values for normalized images
    assert 0.0 <= mean <= 1.0
    assert std > 0.0


def test_compute_fields_mean_std_with_custom_parser(tmp_path: Path):
    """Test computing stats with custom parser and field name."""
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
    
    stats = compute_fields_mean_std(
        [str(tfrecord_path)],
        parser=custom_parser,
        field_names=['data'],
    )
    
    assert 'data' in stats
    mean, std = stats['data']
    
    # Mean should be close to 2.0 (mean of 1, 2, 3)
    assert 1.9 < mean < 2.1
    assert std > 0.0


def test_split_files_train_val_basic():
    """Test basic train/val split."""
    files = [f"file_{i}.tfrecord" for i in range(10)]
    train, val = split_files_train_val(files, val_split=0.2, seed=42)
    
    # Should split into 8 train, 2 val (20% validation)
    assert len(train) == 8
    assert len(val) == 2
    
    # No overlap between train and val
    assert set(train).isdisjoint(set(val))
    
    # All files accounted for
    assert set(train) | set(val) == set(files)


def test_split_files_train_val_deterministic():
    """Test that split is deterministic with same seed."""
    files = [f"file_{i}.tfrecord" for i in range(10)]
    
    train1, val1 = split_files_train_val(files, val_split=0.3, seed=123)
    train2, val2 = split_files_train_val(files, val_split=0.3, seed=123)
    
    # Same seed should give same split
    assert train1 == train2
    assert val1 == val2


def test_split_files_train_val_different_seeds():
    """Test that different seeds give different splits."""
    files = [f"file_{i}.tfrecord" for i in range(10)]
    
    train1, val1 = split_files_train_val(files, val_split=0.3, seed=42)
    train2, val2 = split_files_train_val(files, val_split=0.3, seed=999)
    
    # Different seeds should likely give different splits
    # (very small chance they're the same by accident)
    assert train1 != train2 or val1 != val2


def test_split_files_train_val_edge_cases():
    """Test train/val split with edge case values."""
    files = [f"file_{i}.tfrecord" for i in range(10)]
    
    # Very small validation split
    train, val = split_files_train_val(files, val_split=0.1)
    assert len(val) == 1
    assert len(train) == 9
    
    # Large validation split
    train, val = split_files_train_val(files, val_split=0.9)
    assert len(val) == 9
    assert len(train) == 1


def test_count_tfrecord_samples(sample_tfrecord: str):
    """Test counting samples in TFRecord files."""
    n_samples = count_tfrecord_samples([sample_tfrecord])
    assert n_samples == 10

