from __future__ import annotations

from typing import Iterator, Callable
from dataclasses import dataclass
import glob

import tensorflow as tf

from .iterator_utils import (
    to_jax,
    compute_num_crops,
    split_files_train_val,
    build_dataset_pipeline,
)
from .parsers import make_default_image_parser
from .stats import compute_welford_stats, count_tfrecord_samples
from .preprocessing import (
    FieldConfig,
    FieldType,
    compute_stats_for_fields,
    save_field_stats,
)


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for dataset loading (immutable)."""
    batch_size: int = 32
    shuffle: bool = True
    repeat: bool = True
    crop_size: int | None = None
    stride: int | None = None
    image_shape: tuple[int, int] | None = None


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for train/val split (immutable)."""
    val_split: float
    seed: int = 42


def _resolve_parser(
    parser: Callable[[tf.Tensor], dict[str, tf.Tensor]] | None,
    grayscale: bool,
) -> Callable[[tf.Tensor], dict[str, tf.Tensor]]:
    """Get parser, using default if none provided (pure)."""
    return parser if parser is not None else make_default_image_parser(grayscale)


def _resolve_field_configs(
    files: list[str],
    parser: Callable[[tf.Tensor], dict[str, tf.Tensor]],
    field_configs: dict[str, FieldConfig] | None,
    legacy_stats: tuple[float, float] | None,
) -> dict[str, FieldConfig]:
    """Resolve field configs with stats computation (has side effect: file I/O)."""
    if field_configs is not None:
        return compute_stats_for_fields(files, parser, field_configs)
    
    # Legacy path: single 'image' field
    if legacy_stats is None:
        mean, std = compute_welford_stats(files, parser=parser, field_name="image")
    else:
        mean, std = legacy_stats
    
    return {
        'image': FieldConfig(
            name='image',
            field_type=FieldType.IMAGE,
            standardize=True,
            stats=(mean, std),
        )
    }


def create_iterator(
    tfrecord_pattern: str,
    batch_size: int = 32,
    parser: Callable[[tf.Tensor], dict[str, tf.Tensor]] | None = None,
    field_configs: dict[str, FieldConfig] | None = None,
    augment_fn: Callable[[dict[str, tf.Tensor]], dict[str, tf.Tensor]] | None = None,
    shuffle: bool = True,
    grayscale: bool = True,
    repeat: bool = True,
    crop_size: int | None = None,
    stride: int | None = None,
    image_shape: tuple[int, int] | None = None,
    val_split: float | None = None,
    split_seed: int = 42,
    save_stats_path: str | None = None,
    precomputed_stats: tuple[float, float] | None = None,
) -> tuple[Iterator[dict], int] | tuple[tuple[Iterator[dict], int], tuple[Iterator[dict], int]]:
    """
    Create JAX iterator from TFRecords (has side effects: file I/O).
    
    Args:
        tfrecord_pattern: Glob pattern for TFRecord files
        batch_size: Batch size
        parser: Custom parser or None for default image parser
        field_configs: Field preprocessing configs (computes stats if needed)
        augment_fn: TF augmentation function (not applied to validation)
        shuffle: Whether to shuffle
        grayscale: Convert to grayscale (only if parser=None)
        repeat: Repeat dataset indefinitely
        crop_size: Generate overlapping crops (requires stride)
        stride: Crop stride (required if crop_size set)
        image_shape: (H, W) for efficient crop counting
        val_split: Validation fraction (returns train/val if set)
        split_seed: Seed for train/val split
        save_stats_path: Save computed stats to JSON
        precomputed_stats: DEPRECATED - use field_configs instead
    
    Returns:
        (iterator, batches) or ((train_iter, train_batches), (val_iter, val_batches))
    """
    if crop_size is not None and stride is None:
        raise ValueError("stride required when crop_size is set")
    
    files = sorted(glob.glob(tfrecord_pattern))
    if not files:
        raise ValueError(f"No files found: {tfrecord_pattern}")
    
    parser = _resolve_parser(parser, grayscale)
    field_configs = _resolve_field_configs(files, parser, field_configs, precomputed_stats)
    
    if save_stats_path is not None:
        save_field_stats(field_configs, save_stats_path)
    
    if val_split is not None:
        train_files, val_files = split_files_train_val(files, val_split, split_seed)
        
        train_iter, train_batches = build_dataset_pipeline(
            train_files, parser, field_configs, batch_size,
            crop_size, stride, augment_fn, shuffle, repeat, image_shape,
        )
        val_iter, val_batches = build_dataset_pipeline(
            val_files, parser, field_configs, batch_size,
            crop_size, stride, None, shuffle, repeat, image_shape,
        )
        return (train_iter, train_batches), (val_iter, val_batches)
    
    iterator, batches = build_dataset_pipeline(
        files, parser, field_configs, batch_size,
        crop_size, stride, augment_fn, shuffle, repeat, image_shape,
    )
    return iterator, batches


def create_tfrecord_iterator(
    tfrecord_pattern: str,
    batch_size: int = 32,
    precomputed_stats: tuple[float, float] | None = None,
    augment_fn: Callable[[dict[str, tf.Tensor]], dict[str, tf.Tensor]] | None = None,
    shuffle: bool = True,
    grayscale: bool = True,
    repeat: bool = True,
) -> tuple[Iterator[dict], int]:
    """
    Create a JAX dataloader from TFRecords (has side effects: file I/O).
    
    Note: This function is maintained for backward compatibility.
    New code should use create_iterator() for more flexibility.
    
    Args:
        tfrecord_pattern: Glob pattern for TFRecord files
        batch_size: Batch size
        precomputed_stats: (mean, std) or None to compute from data
        augment_fn: Optional TensorFlow augmentation function (dict -> dict)
        shuffle: Whether to shuffle data
        grayscale: Whether to convert images to grayscale
        repeat: Whether to repeat dataset indefinitely
    
    Returns:
        (iterator, batches_per_epoch)
    """
    return create_iterator(
        tfrecord_pattern=tfrecord_pattern,
        batch_size=batch_size,
        parser=None,  # Use default image parser
        crop_size=None,  # No crops
        stride=None,
        precomputed_stats=precomputed_stats,
        augment_fn=augment_fn,
        shuffle=shuffle,
        grayscale=grayscale,
        repeat=repeat,
    )
