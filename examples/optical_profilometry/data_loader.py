from __future__ import annotations

from pathlib import Path
from functools import partial

import tensorflow as tf
import glob

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


def gaussian_noise(stddev: float, field: str = 'depth') -> callable:
    """
    Add Gaussian noise for adversarial robustness (pure).
   
    Args:
        stddev: Standard deviation of noise (for standardized data, ~0.05-0.1)
        field: Field name to apply noise to
       
    Returns:
        Augmentation function
    """
    def augment(data_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        if field in data_dict:
            noise = tf.random.normal(
                shape=tf.shape(data_dict[field]),
                mean=0.0,
                stddev=stddev,
                dtype=data_dict[field].dtype
            )
            # KEY CHANGE: Return new dict instead of mutating
            return {**data_dict, field: data_dict[field] + noise}
        return data_dict
    return augment


def create_polymer_iterator(
    data_dir: str | Path,
    batch_size: int = 32,
    crop_size: int = 256,
    stride: int = 192,
    shuffle: bool = True,
    augment: bool = False,
    load_stats: bool = False,
) -> tuple:

    data_dir = Path(data_dir)
    tfrecord_pattern = str(data_dir / "*.tfrecord")
    files = sorted(glob.glob(tfrecord_pattern))
    json_path = str(data_dir / "polymer.json")
    
    # Load schema
    feature_dict, shape_dict = load_tfr_dict(json_path)
    parser = make_polymer_parser(feature_dict, shape_dict)
    
    # Compute statistics
    if load_stats:
        stats = load_field_stats(str(data_dir / "depth_stats.json"))
    else:
        stats = compute_fields_mean_std(files, parser, ['depth'])
        save_field_stats(str(data_dir / "depth_stats.json"), stats)
    
    # Get image dimensions
    img_height, img_width = shape_dict['surface']
    
    def z_score_norm(tensor: tf.Tensor) -> tf.Tensor:
        return (tensor - stats['depth'][0]) / stats['depth'][1]

    field_configs = {
        'depth': z_score_norm
    }
    
    # Optional augmentation
    aug_fn = None
    if augment:
        aug_fn = compose(
            random_flip_left_right(fields=['depth']),
            random_flip_up_down(fields=['depth']),
            random_rotate_90(fields=['depth']),
            gaussian_noise(stddev=0.05, field='depth'),  # Add here
            random_brightness(max_delta=0.1, field='depth'),
            random_contrast(lower=0.95, upper=1.05, field='depth'),
        )
    
    # Create iterator with optional train/val split
    iterator, batches_per_epoch = build_dataset_pipeline(
        files=files,
        parser=parser,
        field_configs=field_configs,
        batch_size=batch_size,
        crop_size=crop_size,
        stride=stride,
        augment_fn=aug_fn,
        shuffle=shuffle,
        repeat=True,
        image_shape=(img_height, img_width),
    )
    
    return iterator, batches_per_epoch

