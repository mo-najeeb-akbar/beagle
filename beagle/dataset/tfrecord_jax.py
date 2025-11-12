"""
TFRecord to JAX iterator utilities.

Provides efficient data loading from TFRecords with optional augmentation
and automatic conversion to JAX arrays.
"""
from __future__ import annotations

from typing import Iterator, Callable
from functools import partial
import glob

import tensorflow as tf
import jax
import jax.numpy as jnp
import numpy as np


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


def compute_welford_stats(files: list[str]) -> tuple[float, float]:
    """
    Compute global mean/std using Welford's online algorithm (has side effect: reads files).
    
    This is numerically stable and memory efficient for large datasets.
    
    Args:
        files: List of TFRecord file paths
    
    Returns:
        (mean, std) tuple
    """
    def parse_image(example_proto: tf.Tensor) -> tf.Tensor:
        parsed = tf.io.parse_single_example(
            example_proto, {"image": tf.io.FixedLenFeature([], tf.string)}
        )
        img = tf.io.decode_image(parsed["image"], channels=3)
        img = tf.image.rgb_to_grayscale(img)
        return tf.cast(img, tf.float32) / 255.0
    
    dataset = tf.data.TFRecordDataset(files).map(parse_image).batch(100)
    
    count, mean, m2 = 0, 0.0, 0.0
    for batch in dataset:
        for val in tf.reshape(batch, [-1]).numpy():
            count += 1
            delta = val - mean
            mean += delta / count
            m2 += delta * (val - mean)
    
    std = float((m2 / (count - 1)) ** 0.5) if count > 1 else 0.0
    return float(mean), std


def count_tfrecord_samples(files: list[str]) -> int:
    """
    Count total samples in TFRecord files (has side effect: reads files).
    
    Args:
        files: List of TFRecord file paths
    
    Returns:
        Total number of samples
    """
    return sum(sum(1 for _ in tf.data.TFRecordDataset([f])) for f in files)


def make_image_parser(
    mean: float,
    std: float,
    augment_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    grayscale: bool = True,
) -> Callable[[tf.Tensor], dict[str, tf.Tensor]]:
    """
    Create parsing function with closure over stats and augmentation (pure).
    
    Args:
        mean: Normalization mean
        std: Normalization std
        augment_fn: Optional augmentation function (numpy -> numpy)
        grayscale: Whether to convert to grayscale
    
    Returns:
        Parser function for TFRecord examples
    """
    def parse(example_proto: tf.Tensor) -> dict[str, tf.Tensor]:
        parsed = tf.io.parse_single_example(
            example_proto, {"image": tf.io.FixedLenFeature([], tf.string)}
        )
        img = tf.io.decode_image(parsed["image"], channels=3)
        
        if grayscale:
            img = tf.image.rgb_to_grayscale(img)
        
        img = tf.cast(img, tf.float32) / 255.0
        
        # Apply augmentation if provided (TF -> numpy -> TF)
        if augment_fn is not None:
            img = tf.py_function(
                lambda x: augment_fn(x.numpy()).astype(np.float32),
                [img],
                tf.float32,
            )
            img.set_shape([None, None, 1] if grayscale else [None, None, 3])
        
        # Normalize
        img = (img - mean) / std
        
        return {"image": img}
    
    return parse


def create_tfrecord_iterator(
    tfrecord_pattern: str,
    batch_size: int = 32,
    precomputed_stats: tuple[float, float] | None = None,
    augment_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    shuffle: bool = True,
    grayscale: bool = True,
    repeat: bool = True,
) -> tuple[Iterator[dict], int]:
    """
    Create a JAX dataloader from TFRecords (has side effects: file I/O).
    
    Args:
        tfrecord_pattern: Glob pattern for TFRecord files
        batch_size: Batch size
        precomputed_stats: (mean, std) or None to compute from data
        augment_fn: Optional augmentation function (numpy array -> numpy array)
        shuffle: Whether to shuffle data
        grayscale: Whether to convert images to grayscale
        repeat: Whether to repeat dataset indefinitely
    
    Returns:
        (iterator, batches_per_epoch)
    
    Example:
        >>> from beagle.augmentations import create_transform, MODERATE_AUGMENT
        >>> transform = create_transform(MODERATE_AUGMENT)
        >>> aug_fn = lambda img: apply_transform(transform, img)['image']
        >>> iterator, n_batches = create_tfrecord_iterator(
        ...     "data/*.tfrecord",
        ...     batch_size=32,
        ...     augment_fn=aug_fn,
        ... )
    """
    files = sorted(glob.glob(tfrecord_pattern))
    if not files:
        raise ValueError(f"No files found: {tfrecord_pattern}")
    
    # Compute or use precomputed stats
    mean, std = precomputed_stats if precomputed_stats else compute_welford_stats(files)
    
    # Count samples
    n_samples = count_tfrecord_samples(files)
    batches_per_epoch = n_samples // batch_size
    
    # Build pipeline
    parser = make_image_parser(mean, std, augment_fn, grayscale)
    dataset = tf.data.TFRecordDataset(files).map(
        parser, num_parallel_calls=tf.data.AUTOTUNE
    )
    
    if repeat:
        dataset = dataset.repeat()
    
    if shuffle:
        dataset = dataset.shuffle(4 * batch_size)
    
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    # Create JAX iterator
    iterator = map(partial(to_jax, dtype=jnp.float32), dataset)
    
    return iterator, batches_per_epoch

