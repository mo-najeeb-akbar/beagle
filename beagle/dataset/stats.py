from __future__ import annotations

import tensorflow as tf
import numpy as np
from typing import Callable


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

