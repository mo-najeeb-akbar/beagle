"""Data loader for root tip images with masks."""
from __future__ import annotations

from pathlib import Path
from functools import partial
import glob

import tensorflow as tf

from beagle.dataset import (
    create_train_val_iterators,
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


def make_segmentation_parser(
    feature_dict: dict,
    shape_dict: dict
) -> callable:
    """Create parser for segmentation TFRecords with image and mask."""
    def parse(example_proto: tf.Tensor) -> dict[str, tf.Tensor]:
        # Override feature dict to expect string (serialized images)
        feature_spec = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'mask': tf.io.FixedLenFeature([], tf.string),
        }
        parsed = tf.io.parse_single_example(example_proto, feature_spec)
        
        # Decode PNG images
        image = tf.io.decode_png(parsed['image'], channels=1)
        image = (tf.cast(image, tf.float32) - 127.5) / 127.5
        
        # Decode PNG mask
        mask = tf.io.decode_png(parsed['mask'], channels=1)
        mask = tf.cast(mask, tf.float32)
        
        return {'image': image, 'mask': mask}
    return parse


def create_segmentation_iterator(
    data_dir: str | Path,   
    batch_size: int = 8,
    shuffle: bool = True,
    augment: bool = False,
    val_fraction: float | None = None,
    seed: int = 42,
) -> tuple:
    """
    Create iterator for segmentation image-mask dataset.
    
    Args:
        data_dir: Directory containing TFRecords and segmentation.json
        batch_size: Batch size
        shuffle: Whether to shuffle data
        augment: Whether to apply augmentations
        
    Returns:
        (iterator, batches_per_epoch)
    """
    data_dir = Path(data_dir)
    tfrecord_pattern = str(data_dir / "*.tfrecord")
    files = sorted(glob.glob(tfrecord_pattern))
    json_path = str(data_dir / "segmentation.json")
    
    # Load schema
    feature_dict, shape_dict = load_tfr_dict(json_path)
    parser = make_segmentation_parser(feature_dict, shape_dict)
    
    # Get image dimensions
    img_height, img_width, _ = shape_dict['image']
    
    # Optional augmentation
    aug_fn = None
    if augment:
        # Geometric augmentations applied to BOTH image and mask
        aug_fn = compose(
            random_flip_left_right(fields=['image', 'mask']),
            random_flip_up_down(fields=['image', 'mask']),
            random_rotate_90(fields=['image', 'mask']),
            # Photometric augmentations applied ONLY to image
            random_brightness(max_delta=10, field='image'),
            random_contrast(lower=0.9, upper=1.1, field='image'),
        )
    
    # With split - use create_train_val_iterators with augmentation in TF pipeline
    train_iter, val_iter, train_batches, val_batches = create_train_val_iterators(
        files=files,
        parser=parser,
        batch_size=batch_size,
        val_fraction=val_fraction,
        shuffle=shuffle,
        seed=seed,
        repeat=True,
        augment_fn=aug_fn if augment else None,
    )

    return train_iter, val_iter, train_batches, val_batches
