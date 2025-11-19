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
    save_field_stats,
    load_field_stats,
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
            data_dict[field] = data_dict[field] + noise
        return data_dict
    return augment


def random_dropout_patches(
    patch_size: int = 32,
    max_patches: int = 3,
    fill_value: float = 0.0,
    field: str = 'depth',
    probability: float = 0.5,
) -> callable:
    """
    Randomly dropout square patches to simulate missing/corrupted data (pure).
    
    Args:
        patch_size: Size of square patches to drop
        max_patches: Maximum number of patches to drop per image
        fill_value: Value to fill dropped patches with
        field: Field name to apply dropout to
        probability: Probability of applying this augmentation
        
    Returns:
        Augmentation function
    """
    def augment(data_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        if field not in data_dict:
            return data_dict
        
        # Apply with probability
        if tf.random.uniform([]) > probability:
            return data_dict
        
        img = data_dict[field]
        shape = tf.shape(img)
        h, w = shape[0], shape[1]
        
        # Random number of patches to drop
        num_patches = tf.random.uniform([], 0, max_patches + 1, dtype=tf.int32)
        
        # Create mask for patches
        mask = tf.ones_like(img)
        
        def drop_patch(i: tf.Tensor, current_mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            # Random position
            y = tf.random.uniform([], 0, h - patch_size + 1, dtype=tf.int32)
            x = tf.random.uniform([], 0, w - patch_size + 1, dtype=tf.int32)
            
            # Create indices for patch
            y_range = tf.range(y, y + patch_size)
            x_range = tf.range(x, x + patch_size)
            yy, xx = tf.meshgrid(y_range, x_range, indexing='ij')
            indices = tf.stack([
                tf.reshape(yy, [-1]),
                tf.reshape(xx, [-1]),
                tf.zeros([patch_size * patch_size], dtype=tf.int32)
            ], axis=1)
            
            # Update mask
            updates = tf.zeros([patch_size * patch_size], dtype=current_mask.dtype)
            new_mask = tf.tensor_scatter_nd_update(current_mask, indices, updates)
            
            return i + 1, new_mask
        
        _, mask = tf.while_loop(
            lambda i, _: i < num_patches,
            drop_patch,
            [0, mask],
            shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None, None])]
        )
        
        # Apply mask
        data_dict[field] = img * mask + (1 - mask) * fill_value
        
        return data_dict
    
    return augment


def create_polymer_augmentation(
    use_dropout: bool = False,
    noise_stddev: float = 0.05,
    brightness_delta: float = 0.15,
    contrast_range: tuple[float, float] = (0.85, 1.15),
) -> callable:
    """
    Create robust augmentation pipeline for depth maps.
    
    Designed for:
    - Autoencoder compression learning
    - Adversarial robustness
    - Physical plausibility (heightmaps)
    
    Args:
        use_dropout: Whether to include random dropout patches
        noise_stddev: Standard deviation for Gaussian noise
        brightness_delta: Max brightness adjustment (additive)
        contrast_range: Contrast adjustment range (multiplicative)
        
    Returns:
        Composed augmentation function
    """
    augmentations = [
        # Geometric augmentations (physically valid)
        random_flip_left_right(fields=['depth']),
        random_flip_up_down(fields=['depth']),
        random_rotate_90(fields=['depth']),
        
        # Additive noise (critical for adversarial robustness)
        gaussian_noise(stddev=noise_stddev, field='depth'),
        
        # Brightness variation (additive constant shift)
        random_brightness(max_delta=brightness_delta, field='depth'),
        
        # Contrast variation (multiplicative scaling)
        random_contrast(lower=contrast_range[0], upper=contrast_range[1], field='depth'),
    ]
    
    # Optional: dropout patches for extra robustness
    if use_dropout:
        augmentations.append(
            random_dropout_patches(
                patch_size=32,
                max_patches=2,
                field='depth',
                probability=0.3
            )
        )
    
    return compose(*augmentations)


def create_polymer_iterator(
    data_dir: str | Path,
    batch_size: int = 32,
    crop_size: int = 256,
    stride: int = 192,
    shuffle: bool = True,
    augment: bool = True,
    field_stats: tuple[float, float] | None = None,
    val_split: float | None = None,
    split_seed: int = 42,
    save_stats: bool = False,
    stats_path: str | Path | None = None,
) -> tuple:
    """Create iterator for polymer depth map dataset.
    
    Args:
        data_dir: Directory containing *.tfrecord and polymer.json
        batch_size: Batch size
        crop_size: Size of crops
        stride: Crop stride/overlap
        shuffle: Whether to shuffle
        augment: Whether to apply augmentations (not applied to validation)
        field_stats: Optional (mean, std) to skip computation
        val_split: Validation fraction (e.g., 0.2 for 20% validation)
        split_seed: Random seed for reproducible train/val split
        save_stats: If True, save computed stats to stats_path
        stats_path: Path to save/load stats (default: data_dir/polymer_stats.json)
        
    Returns:
        If val_split is None:
            (iterator, batches_per_epoch, image_shape)
        If val_split is provided:
            ((train_iter, train_batches), (val_iter, val_batches), image_shape)
    """
    data_dir = Path(data_dir)
    tfrecord_pattern = str(data_dir / "*.tfrecord")
    json_path = str(data_dir / "polymer.json")
    
    # Default stats path
    if stats_path is None:
        stats_path = data_dir / "polymer_stats.json"
    else:
        stats_path = Path(stats_path)
    
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
    
    # Determine whether to save stats
    save_stats_path = str(stats_path) if save_stats else None
    
    # Create iterator with optional train/val split
    result = create_iterator(
        tfrecord_pattern=tfrecord_pattern,
        parser=parser,
        crop_size=crop_size,
        stride=stride,
        image_shape=(img_height, img_width),
        batch_size=batch_size,
        field_configs=field_configs,
        augment_fn=aug_fn,
        shuffle=shuffle,
        val_split=val_split,
        split_seed=split_seed,
        save_stats_path=save_stats_path,
    )
    
    # Append image shape to result
    if val_split is not None:
        # Result is ((train_iter, train_batches), (val_iter, val_batches))
        return result[0], result[1], (img_height, img_width)
    else:
        # Result is (iterator, batches_per_epoch)
        return result[0], result[1], (img_height, img_width)


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


def create_polymer_iterator_from_saved_stats(
    data_dir: str | Path,
    stats_path: str | Path | None = None,
    batch_size: int = 32,
    crop_size: int = 256,
    stride: int = 192,
    shuffle: bool = False,
    augment: bool = False,
    val_split: float | None = None,
    split_seed: int = 42,
) -> tuple:
    """
    Create iterator for inference using saved stats (no augmentation by default).
    
    This is a convenience function for inference that:
    - Loads pre-computed stats from JSON
    - Disables augmentation by default
    - Disables shuffling by default
    
    Args:
        data_dir: Directory containing *.tfrecord and polymer.json
        stats_path: Path to saved stats JSON (default: data_dir/polymer_stats.json)
        batch_size: Batch size
        crop_size: Size of crops
        stride: Crop stride/overlap
        shuffle: Whether to shuffle (default: False for inference)
        augment: Whether to apply augmentations (default: False for inference)
        val_split: Validation fraction (e.g., 0.2 for 20% validation)
        split_seed: Random seed for reproducible train/val split
        
    Returns:
        If val_split is None:
            (iterator, batches_per_epoch, image_shape)
        If val_split is provided:
            ((train_iter, train_batches), (val_iter, val_batches), image_shape)
        
    Example:
        >>> # Training: save stats
        >>> train_iter, _, _ = create_polymer_iterator(
        ...     '/data/train',
        ...     save_stats=True,
        ...     augment=True
        ... )
        >>> 
        >>> # Inference: load stats
        >>> val_iter, _, _ = create_polymer_iterator_from_saved_stats(
        ...     '/data/val',
        ...     stats_path='/data/train/polymer_stats.json'
        ... )
        >>>
        >>> # Or with train/val split
        >>> (train, _), (val, _), _ = create_polymer_iterator_from_saved_stats(
        ...     '/data/all',
        ...     stats_path='/data/train/polymer_stats.json',
        ...     val_split=0.2
        ... )
    """
    data_dir = Path(data_dir)
    
    # Default stats path
    if stats_path is None:
        stats_path = data_dir / "polymer_stats.json"
    else:
        stats_path = Path(stats_path)
    
    # Load saved stats
    loaded_configs = load_field_stats(stats_path)
    
    # Extract stats for backward compatibility
    depth_config = loaded_configs['depth']
    field_stats = depth_config.stats
    
    return create_polymer_iterator(
        data_dir=data_dir,
        batch_size=batch_size,
        crop_size=crop_size,
        stride=stride,
        shuffle=shuffle,
        augment=augment,
        field_stats=field_stats,
        val_split=val_split,
        split_seed=split_seed,
        save_stats=False,  # Don't re-save
        stats_path=stats_path,
    )

