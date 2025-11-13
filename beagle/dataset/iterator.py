from __future__ import annotations

from typing import Iterator, Callable
from functools import partial
import glob

import tensorflow as tf
import jax
import jax.numpy as jnp
import numpy as np

from .crops import create_overlapping_crops
from .preprocessing import (
    FieldConfig,
    FieldType,
    create_standardize_fn,
    compute_stats_for_fields,
)


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


def count_tfrecord_samples(files: list[str]) -> int:
    """
    Count total samples in TFRecord files (has side effect: reads files).
    
    Args:
        files: List of TFRecord file paths
    
    Returns:
        Total number of samples
    """
    return sum(sum(1 for _ in tf.data.TFRecordDataset([f])) for f in files)


def compute_welford_stats(
    files: list[str],
    parser: Callable[[tf.Tensor], dict[str, tf.Tensor]] | None = None,
    field_name: str = "image",
) -> tuple[float, float]:
    """
    Compute global mean/std using Welford's online algorithm (has side effect: reads files).
    
    Uses batched/vectorized updates for efficiency - much faster than per-element updates.
    
    Args:
        files: List of TFRecord file paths
        parser: Optional custom parser function that returns dict with 'image' field
                If None, uses default image parser
        field_name: Name of field to compute stats on (default: 'image')
    
    Returns:
        (mean, std) tuple
    """
    if parser is None:
        def parse_image(example_proto: tf.Tensor) -> dict[str, tf.Tensor]:
            parsed = tf.io.parse_single_example(
                example_proto, {"image": tf.io.FixedLenFeature([], tf.string)}
            )
            img = tf.io.decode_image(parsed["image"], channels=3)
            img = tf.image.rgb_to_grayscale(img)
            img = tf.cast(img, tf.float32) / 255.0
            return {"image": img}
        parser = parse_image
    
    # Parallel map and larger batches for speed
    dataset = (
        tf.data.TFRecordDataset(files)
        .map(parser, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(100)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    # Welford's algorithm for batched updates (vectorized - MUCH faster!)
    count = 0
    mean = 0.0
    m2 = 0.0
    
    for batch in dataset:
        values = tf.reshape(batch[field_name], [-1]).numpy()
        # Remove NaNs and zeros (masked values)
        values = values[~np.isnan(values)]
        values = values[values != 0.0]
        
        if len(values) == 0:
            continue
        
        # Batched Welford update (vectorized!)
        batch_count = len(values)
        batch_mean = np.mean(values)
        batch_m2 = np.sum((values - batch_mean) ** 2)
        
        # Merge batch stats with running stats
        new_count = count + batch_count
        delta = batch_mean - mean
        mean = mean + delta * batch_count / new_count
        m2 = m2 + batch_m2 + delta**2 * count * batch_count / new_count
        count = new_count
    
    std = float(np.sqrt(m2 / (count - 1))) if count > 1 else 0.0
    return float(mean), std


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


def make_default_image_parser(
    grayscale: bool = True,
) -> Callable[[tf.Tensor], dict[str, tf.Tensor]]:
    """
    Create default parser for standard image TFRecords (pure).
    
    Args:
        grayscale: Whether to convert to grayscale
    
    Returns:
        Parser function that returns dict with 'image' field
    """
    def parse(example_proto: tf.Tensor) -> dict[str, tf.Tensor]:
        parsed = tf.io.parse_single_example(
            example_proto, {"image": tf.io.FixedLenFeature([], tf.string)}
        )
        img = tf.io.decode_image(parsed["image"], channels=3)
        
        if grayscale:
            img = tf.image.rgb_to_grayscale(img)
        
        img = tf.cast(img, tf.float32) / 255.0
        
        return {"image": img}
    
    return parse


def create_iterator(
    tfrecord_pattern: str,
    batch_size: int = 32,
    parser: Callable[[tf.Tensor], dict[str, tf.Tensor]] | None = None,
    crop_size: int | None = None,
    stride: int | None = None,
    image_shape: tuple[int, int] | None = None,
    field_configs: dict[str, FieldConfig] | None = None,
    precomputed_stats: tuple[float, float] | None = None,
    augment_fn: Callable[[dict[str, tf.Tensor]], dict[str, tf.Tensor]] | None = None,
    shuffle: bool = True,
    grayscale: bool = True,
    repeat: bool = True,
) -> tuple[Iterator[dict], int]:
    """
    Create JAX iterator from TFRecords with optional crops (has side effects: file I/O).
    
    This is the unified iterator function that handles both standard images
    and custom parsers, with optional crop generation.
    
    Args:
        tfrecord_pattern: Glob pattern for TFRecord files
        batch_size: Batch size
        parser: Custom parser (tf.Tensor -> dict)
                If None, uses default image parser with decode/grayscale
        crop_size: If provided, generate overlapping crops of this size
        stride: Step size between crops (required if crop_size is set)
        image_shape: (height, width) for efficient crop count computation
                     If None and crops are used, counts by iterating dataset (slower)
        field_configs: Dictionary mapping field names to FieldConfig objects
                       Controls preprocessing/standardization per field
                       If None, uses legacy behavior (standardize 'image' field only)
        precomputed_stats: DEPRECATED - use field_configs instead
                          (mean, std) for 'image' field backward compatibility
        augment_fn: Optional TensorFlow augmentation function (dict -> dict)
                    Applied BEFORE batching in the TF pipeline for max performance
        shuffle: Whether to shuffle data/crops
        grayscale: Whether to convert to grayscale (only used if parser=None)
        repeat: Whether to repeat dataset indefinitely
    
    Returns:
        (iterator, batches_per_epoch) tuple where batches_per_epoch is
        computed from total samples (images or crops) divided by batch_size
    
    Examples:
        # Standard images without crops
        >>> iterator, n = create_iterator("data/*.tfrecord", batch_size=32)
        
        # Standard images with crops (fast with image_shape)
        >>> iterator, n = create_iterator(
        ...     "data/*.tfrecord",
        ...     crop_size=256,
        ...     stride=192,
        ...     image_shape=(1024, 1024),  # For efficient crop count
        ...     batch_size=32,
        ... )
        
        # With TensorFlow augmentation (efficient!)
        >>> def augment(data_dict):
        ...     img = data_dict['image']
        ...     img = tf.image.random_flip_left_right(img)
        ...     img = tf.image.random_brightness(img, 0.2)
        ...     data_dict['image'] = img
        ...     return data_dict
        >>> iterator, n = create_iterator(
        ...     "data/*.tfrecord",
        ...     batch_size=32,
        ...     augment_fn=augment,
        ... )
    """
    # Validate inputs
    if crop_size is not None and stride is None:
        raise ValueError("stride must be provided when crop_size is set")
    
    files = sorted(glob.glob(tfrecord_pattern))
    if not files:
        raise ValueError(f"No files found: {tfrecord_pattern}")
    
    # Use default parser if none provided
    if parser is None:
        parser = make_default_image_parser(grayscale=grayscale)
    
    # Handle field configs (new flexible API vs legacy)
    if field_configs is None:
        # Legacy behavior: standardize only 'image' field
        if precomputed_stats is None:
            mean, std = compute_welford_stats(files, parser=parser, field_name="image")
        else:
            mean, std = precomputed_stats
        
        field_configs = {
            'image': FieldConfig(
                name='image',
                field_type=FieldType.IMAGE,
                standardize=True,
                stats=(mean, std),
            )
        }
    else:
        # New flexible API: compute stats for fields that need it
        field_configs = compute_stats_for_fields(files, parser, field_configs)
    
    # Build dataset pipeline with optimized ordering
    # Use moderate parallelism (4-8 threads) for predictable performance
    # AUTOTUNE can be unstable; explicit values give consistent timing
    n_parallel = 6
    
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=n_parallel).map(
        parser, num_parallel_calls=n_parallel
    )
    
    # Generate crops if requested
    if crop_size is not None:
        crop_fn = partial(create_overlapping_crops, crop_size=crop_size, stride=stride)
        dataset = dataset.map(crop_fn, num_parallel_calls=n_parallel)
        dataset = dataset.unbatch()
    
    # Apply preprocessing/standardization (before cache for consistency)
    standardize_fn = create_standardize_fn(field_configs)
    dataset = dataset.map(standardize_fn, num_parallel_calls=n_parallel)
    
    # Cache AFTER preprocessing to avoid redundant computation
    dataset = dataset.cache()
    
    # Apply augmentation AFTER cache (augmentations are random, shouldn't be cached)
    if augment_fn is not None:
        dataset = dataset.map(augment_fn, num_parallel_calls=n_parallel)
    
    # Count samples/crops
    if crop_size is not None:
        if image_shape is not None:
            # Fast: compute mathematically
            n_images = count_tfrecord_samples(files)
            crops_per_image = compute_num_crops(image_shape[0], image_shape[1], crop_size, stride)
            n_samples = n_images * crops_per_image
        else:
            # Slow: iterate dataset to count (only if image_shape not provided)
            print("Warning: Counting crops by iteration (slow). Provide image_shape for faster startup.")
            temp_dataset = tf.data.TFRecordDataset(files).map(parser)
            temp_dataset = temp_dataset.map(crop_fn).unbatch()
            n_samples = sum(1 for _ in temp_dataset)
    else:
        # Count images
        n_samples = count_tfrecord_samples(files)
    
    batches_per_epoch = n_samples // batch_size
    
    # Shuffle and repeat
    if repeat:
        dataset = dataset.repeat()
    
    if shuffle:
        # Optimized shuffle buffer: balance memory vs I/O efficiency
        # For crops: larger buffer (more variety), for images: smaller buffer (faster)
        if crop_size:
            # Use ~10 batches worth of crops for good mixing without excessive memory
            buffer_size = min(10 * batch_size, n_samples // 2)
        else:
            # For images, 8-16 batches is sufficient
            buffer_size = min(16 * batch_size, n_samples)
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    
    # Batch and prefetch with explicit sizing for consistent performance
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # Explicit prefetch of 4-8 batches prevents I/O stalls
    # More aggressive than AUTOTUNE but ensures GPU never starves
    prefetch_size = 8 if crop_size else 4
    dataset = dataset.prefetch(prefetch_size)
    
    # Convert to JAX only at the very end
    iterator = map(partial(to_jax, dtype=jnp.float32), dataset)
    
    return iterator, batches_per_epoch


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
    
    Example:
        >>> def augment(data_dict):
        ...     img = data_dict['image']
        ...     img = tf.image.random_flip_left_right(img)
        ...     img = tf.image.random_brightness(img, 0.2)
        ...     data_dict['image'] = img
        ...     return data_dict
        >>> iterator, n_batches = create_tfrecord_iterator(
        ...     "data/*.tfrecord",
        ...     batch_size=32,
        ...     augment_fn=augment,
        ... )
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

