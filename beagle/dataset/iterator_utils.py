from __future__ import annotations

from typing import Iterator, Callable
from functools import partial
import random

import tensorflow as tf
import jax
import jax.numpy as jnp

from .crops import create_overlapping_crops
from .preprocessing import FieldConfig, create_standardize_fn
from .stats import count_tfrecord_samples
from .parsers import make_default_image_parser


def to_jax(tensor_dict: dict, dtype: jnp.dtype = jnp.float32) -> dict:
    """
    Convert TensorFlow tensors to JAX arrays (pure function).
    
    Args:
        tensor_dict: Dictionary of TensorFlow tensors
        dtype: Target JAX dtype
    
    Returns:
        Dictionary with same structure but JAX arrays
    """
    return jax.tree.map(lambda x: jnp.array(x, dtype=dtype), tensor_dict)


def compute_num_crops(
    height: int,
    width: int,
    crop_size: int,
    stride: int,
) -> int:
    """
    Compute number of crops from image dimensions (pure function).
    
    Uses the same formula as TensorFlow's extract_patches with VALID padding.
    
    Args:
        height: Image height
        width: Image width
        crop_size: Crop size (assumes square crops)
        stride: Step size between crops
    
    Returns:
        Total number of crops
    """
    num_rows = (height - crop_size) // stride + 1
    num_cols = (width - crop_size) // stride + 1
    return num_rows * num_cols


def split_files_train_val(
    files: list[str],
    val_split: float,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Split files into train and validation sets (pure function with deterministic seed).
    
    Args:
        files: List of file paths
        val_split: Fraction of files to use for validation (0.0 to 1.0)
        seed: Random seed for reproducible splits
    
    Returns:
        (train_files, val_files) tuple
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be between 0 and 1, got {val_split}")
    
    # Create deterministic split
    rng = random.Random(seed)
    shuffled = files.copy()
    rng.shuffle(shuffled)
    
    val_size = int(len(shuffled) * val_split)
    val_files = shuffled[:val_size]
    train_files = shuffled[val_size:]
    
    return train_files, val_files


def build_dataset_pipeline(
    files: list[str],
    parser: Callable[[tf.Tensor], dict[str, tf.Tensor]],
    field_configs: dict[str, FieldConfig],
    batch_size: int,
    crop_size: int | None,
    stride: int | None,
    augment_fn: Callable[[dict[str, tf.Tensor]], dict[str, tf.Tensor]] | None,
    shuffle: bool,
    repeat: bool,
    image_shape: tuple[int, int] | None = None,
) -> tuple[Iterator[dict], int]:
    """
    Build a single TF dataset pipeline (has side effects: file I/O).
    
    Args:
        files: List of TFRecord file paths
        parser: Parser function (tf.Tensor -> dict)
        field_configs: Field configuration for preprocessing
        batch_size: Batch size
        crop_size: Optional crop size
        stride: Optional stride (required if crop_size is set)
        augment_fn: Optional augmentation function
        shuffle: Whether to shuffle
        repeat: Whether to repeat dataset indefinitely
        image_shape: Optional (height, width) for efficient crop counting
    
    Returns:
        (iterator, batches_per_epoch) tuple
    """
    n_parallel = 6
    
    # Count samples BEFORE building the pipeline (to avoid consuming it)
    if crop_size is not None:
        if image_shape is not None:
            # Fast: compute mathematically
            n_images = count_tfrecord_samples(files)
            crops_per_image = compute_num_crops(image_shape[0], image_shape[1], crop_size, stride)
            n_samples = n_images * crops_per_image
        else:
            # Slow: iterate dataset to count
            print("Warning: Counting crops by iteration (slow). Provide image_shape for faster startup.")
            crop_fn = partial(create_overlapping_crops, crop_size=crop_size, stride=stride)
            temp_dataset = tf.data.TFRecordDataset(files).map(parser)
            temp_dataset = temp_dataset.map(crop_fn).unbatch()
            n_samples = sum(1 for _ in temp_dataset)
    else:
        n_samples = count_tfrecord_samples(files)
    
    batches_per_epoch = n_samples // batch_size
    
    # Build dataset pipeline
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=n_parallel).map(
        parser, num_parallel_calls=n_parallel
    )
    
    # Generate crops if requested
    if crop_size is not None:
        crop_fn = partial(create_overlapping_crops, crop_size=crop_size, stride=stride)
        dataset = dataset.map(crop_fn, num_parallel_calls=n_parallel)
        dataset = dataset.unbatch()
    
    # Apply preprocessing/standardization
    standardize_fn = create_standardize_fn(field_configs)
    dataset = dataset.map(standardize_fn, num_parallel_calls=n_parallel)
    
    # Cache AFTER preprocessing
    dataset = dataset.cache()
    
    # Apply augmentation AFTER cache
    if augment_fn is not None:
        dataset = dataset.map(augment_fn, num_parallel_calls=n_parallel)
    
    # Shuffle and repeat
    if repeat:
        dataset = dataset.repeat()
    
    if shuffle:
        if crop_size:
            buffer_size = min(10 * batch_size, n_samples // 2)
        else:
            buffer_size = min(16 * batch_size, n_samples)
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size, drop_remainder=True)
    prefetch_size = 8 if crop_size else 4
    dataset = dataset.prefetch(prefetch_size)
    
    # Convert to JAX
    iterator = map(partial(to_jax, dtype=jnp.float32), dataset)
    
    return iterator, batches_per_epoch

