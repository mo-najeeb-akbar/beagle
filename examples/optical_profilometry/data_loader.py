from __future__ import annotations

from pathlib import Path
from functools import partial

import tensorflow as tf
import glob

from beagle.dataset import (
    build_dataset_pipeline,
    create_train_val_iterators,
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
        depth = tf.reshape(parsed['depth'], shape_dict['depth'] + [1])
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
    load_stats: bool = None,
    val_fraction: float | None = None,
    seed: int = 42,
) -> tuple:

    """
    Create polymer data iterator with optional train/val splitting.

    Args:
        data_dir: Directory containing TFRecord files
        batch_size: Batch size
        crop_size: Size of crops to extract
        stride: Stride for overlapping crops
        shuffle: Whether to shuffle data
        augment: Whether to apply augmentation
        load_stats: Whether to load/compute statistics (None=skip, True=load, False=compute)
        val_fraction: If provided, split into train/val at sample level (e.g., 0.2 for 20% val)
        seed: Random seed for splitting

    Returns:
        If val_fraction is None: (iterator, batches_per_epoch)
        If val_fraction is provided: (train_iterator, val_iterator, train_batches, val_batches)
    """
    data_dir = Path(data_dir)
    tfrecord_pattern = str(data_dir / "*.tfrecord")
    files = sorted(glob.glob(tfrecord_pattern))
    json_path = str(data_dir / "polymer.json")

    # Load schema
    feature_dict, shape_dict = load_tfr_dict(json_path)
    parser = make_polymer_parser(feature_dict, shape_dict)

    # Compute statistics
    if load_stats is not None:
        if load_stats:
            stats = load_field_stats(str(data_dir / "depth_stats.json"))
        else:
            stats = compute_fields_mean_std(files, parser, ['depth'])
            save_field_stats(str(data_dir / "depth_stats.json"), stats)
    else:
        stats = None

    # Get image dimensions
    img_height, img_width = shape_dict['depth']

    def z_score_norm(tensor: tf.Tensor) -> tf.Tensor:
        # return (tensor - stats['depth'][0]) / stats['depth'][1]
        return (tensor  - 127.5) / 127.5

    # Create parser with field normalization
    def parser_with_norm(serialized: tf.Tensor) -> dict[str, tf.Tensor]:
        parsed = parser(serialized)
        return {'depth': z_score_norm(parsed['depth'])}

    # Optional augmentation
    aug_fn = None
    if augment:
        aug_fn = compose(
            random_flip_left_right(fields=['depth']),
            random_flip_up_down(fields=['depth']),
            random_rotate_90(fields=['depth']),
            # gaussian_noise(stddev=10.0, field='depth'),  # Add here
            # random_brightness(max_delta=5.0, field='depth'),
            # random_contrast(lower=0.95, upper=1.05, field='depth'),
        )

    # Create iterator with optional train/val split
    if val_fraction is None:
        # No split - use build_dataset_pipeline for backward compatibility
        field_configs = {'depth': z_score_norm}
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

    else:
        # With split - use create_train_val_iterators with augmentation in TF pipeline
        train_iter, val_iter, train_batches, val_batches = create_train_val_iterators(
            files=files,
            parser=parser_with_norm,
            batch_size=batch_size,
            val_fraction=val_fraction,
            shuffle=shuffle,
            seed=seed,
            repeat=True,
            crop_size=crop_size,
            stride=stride,
            image_shape=(img_height, img_width),
            augment_fn=aug_fn if augment else None,
        )

        return train_iter, val_iter, train_batches, val_batches

