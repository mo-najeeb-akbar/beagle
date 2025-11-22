"""Data loader for root tip images with masks."""
from __future__ import annotations

from pathlib import Path
from functools import partial
import glob

import tensorflow as tf

from beagle.dataset import (
    build_dataset_pipeline,
    compute_fields_mean_std,
    save_field_stats,
    load_field_stats,
    load_tfr_dict,
)
from beagle.augmentations import (
    compose,
    random_flip_left_right,
    random_flip_up_down,
    random_rotate_90,
    random_brightness,
    random_contrast,
)


def make_root_tip_parser(
    feature_dict: dict,
    shape_dict: dict
) -> callable:
    """Create parser for root tip TFRecords with image and mask."""
    def parse(example_proto: tf.Tensor) -> dict[str, tf.Tensor]:
        # Override feature dict to expect string (serialized images)
        feature_spec = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'mask': tf.io.FixedLenFeature([], tf.string),
        }
        parsed = tf.io.parse_single_example(example_proto, feature_spec)
        
        # Decode PNG images
        image = tf.io.decode_png(parsed['image'], channels=1)
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
        
        # Decode PNG mask
        mask = tf.io.decode_png(parsed['mask'], channels=1)
        mask = tf.cast(mask, tf.float32)
        
        return {'image': image, 'mask': mask}
    return parse


def create_root_tip_iterator(
    data_dir: str | Path,
    batch_size: int = 8,
    shuffle: bool = True,
    augment: bool = False,
    load_stats: bool = False,
) -> tuple:
    """
    Create iterator for root tip image-mask dataset.
    
    Args:
        data_dir: Directory containing TFRecords and root_tip.json
        batch_size: Batch size
        shuffle: Whether to shuffle data
        augment: Whether to apply augmentations
        
    Returns:
        (iterator, batches_per_epoch)
    """
    data_dir = Path(data_dir)
    tfrecord_pattern = str(data_dir / "*.tfrecord")
    files = sorted(glob.glob(tfrecord_pattern))
    json_path = str(data_dir / "root_tip.json")
    
    if not files:
        raise ValueError(f"No TFRecord files found in {data_dir}")
    
    # Load schema
    feature_dict, shape_dict = load_tfr_dict(json_path)
    parser = make_root_tip_parser(feature_dict, shape_dict)
    
    # Compute statistics for image field (not mask)
    if load_stats:
        stats = load_field_stats(str(data_dir / "image_stats.json"))
    else:
        stats = compute_fields_mean_std(files, parser, ['image'])
        save_field_stats(str(data_dir / "image_stats.json"), stats)
    
    # Get image dimensions
    img_height, img_width, _ = shape_dict['image']
    
    def z_score_norm(tensor: tf.Tensor) -> tf.Tensor:
        """Z-score normalization for image (pure)."""
        return (tensor - stats['image'][0]) / stats['image'][1]
    
    def identity(tensor: tf.Tensor) -> tf.Tensor:
        """Identity function to keep mask unchanged (pure)."""
        return tensor
    
    field_configs = {
        'image': z_score_norm,
        'mask': identity,  # Keep mask in [0, 1]
    }
    
    # Optional augmentation
    aug_fn = None
    if augment:
        # Geometric augmentations applied to BOTH image and mask
        aug_fn = compose(
            random_flip_left_right(fields=['image', 'mask']),
            random_flip_up_down(fields=['image', 'mask']),
            random_rotate_90(fields=['image', 'mask']),
            # Photometric augmentations applied ONLY to image
            random_brightness(max_delta=0.1, field='image'),
            random_contrast(lower=0.9, upper=1.1, field='image'),
        )
    
    # Create iterator
    iterator, batches_per_epoch = build_dataset_pipeline(
        files=files,
        parser=parser,
        field_configs=field_configs,
        batch_size=batch_size,
        crop_size=None,
        stride=None,
        augment_fn=aug_fn,
        shuffle=shuffle,
        repeat=True,
        image_shape=(img_height, img_width),
    )
    
    return iterator, batches_per_epoch

