"""Polymer dataset loading utilities.

Reusable data loading for polymer depth maps with proper augmentations.
Used by both training scripts and data inspection tools.
"""

from __future__ import annotations

from pathlib import Path
from functools import partial

import tensorflow as tf
import glob

from beagle.dataset import (
    load_tfr_dict,
    create_iterator,
    compute_field_stats,
    count_tfrecord_samples,
    FieldConfig,
    FieldType,
)


def make_polymer_parser(
    feature_dict: dict,
    shape_dict: dict
) -> callable:
    """Create parser for polymer TFRecords."""
    def parse(example_proto: tf.Tensor) -> dict[str, tf.Tensor]:
        parsed = tf.io.parse_single_example(example_proto, feature_dict)
        depth = tf.reshape(parsed['surface'], shape_dict['surface'] + [1])
        depth = tf.where(tf.math.is_nan(depth), 0.0, depth)
        return {'depth': depth}
    return parse


def create_polymer_augmentation() -> callable:
    """Create TensorFlow augmentation pipeline for depth maps."""
    def augment(data_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        depth = data_dict['depth']
        
        # Random flips
        depth = tf.image.random_flip_left_right(depth)
        depth = tf.image.random_flip_up_down(depth)
        
        # Random 90-degree rotations
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        depth = tf.image.rot90(depth, k=k)
        
        # Brightness (additive noise for depth maps)
        depth = tf.image.random_brightness(depth, 0.1)
        
        data_dict['depth'] = depth
        return data_dict
    
    return augment


def create_polymer_iterator(
    data_dir: str | Path,
    batch_size: int = 32,
    crop_size: int = 256,
    stride: int = 192,
    shuffle: bool = True,
    augment: bool = True,
    field_stats: tuple[float, float] | None = None,
) -> tuple:
    """Create iterator for polymer depth map dataset.
    
    Args:
        data_dir: Directory containing *.tfrecord and polymer.json
        batch_size: Batch size
        crop_size: Size of crops
        stride: Crop stride/overlap
        shuffle: Whether to shuffle
        augment: Whether to apply augmentations
        field_stats: Optional (mean, std) to skip computation
        
    Returns:
        (iterator, batches_per_epoch, image_shape)
    """
    data_dir = Path(data_dir)
    tfrecord_pattern = str(data_dir / "*.tfrecord")
    json_path = str(data_dir / "polymer.json")
    
    # Load schema
    feature_dict, shape_dict = load_tfr_dict(json_path)
    parser = make_polymer_parser(feature_dict, shape_dict)
    
    # Get image dimensions
    img_height, img_width = shape_dict['surface']
    
    # Configure field
    field_configs = {
        'depth': FieldConfig(
            name='depth',
            field_type=FieldType.IMAGE,
            standardize=True,
            stats=field_stats,  # None = auto-compute
        )
    }
    
    # Optional augmentation
    aug_fn = create_polymer_augmentation() if augment else None
    
    # Create iterator
    iterator, batches_per_epoch = create_iterator(
        tfrecord_pattern=tfrecord_pattern,
        parser=parser,
        crop_size=crop_size,
        stride=stride,
        image_shape=(img_height, img_width),
        batch_size=batch_size,
        field_configs=field_configs,
        augment_fn=aug_fn,
        shuffle=shuffle,
    )
    
    return iterator, batches_per_epoch, (img_height, img_width)


def compute_polymer_stats(data_dir: str | Path) -> tuple[float, float, int]:
    """Compute statistics for polymer dataset.
    
    Args:
        data_dir: Directory containing *.tfrecord and polymer.json
        
    Returns:
        (mean, std, num_images)
    """
    data_dir = Path(data_dir)
    tfrecord_pattern = str(data_dir / "*.tfrecord")
    json_path = str(data_dir / "polymer.json")
    
    feature_dict, shape_dict = load_tfr_dict(json_path)
    parser = make_polymer_parser(feature_dict, shape_dict)
    
    files = sorted(glob.glob(tfrecord_pattern))
    
    mean, std = compute_field_stats(files, parser=parser, field_name='depth')
    n_imgs = count_tfrecord_samples(files)
    
    return mean, std, n_imgs

