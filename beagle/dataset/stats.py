from __future__ import annotations

from typing import Callable

import tensorflow as tf


def compute_fields_min_max(
    files: list[str],
    parser: Callable[[tf.Tensor], dict[str, tf.Tensor]],
    field_names: list[str],
) -> dict[str, tuple[float, float]]:
    """
    Compute min and max for multiple fields (has side effect: file I/O).
    
    Automatically filters out NaN and zero values (common masking approach).
    More efficient than calling compute_field_min_max multiple times.
    
    Args:
        files: List of TFRecord file paths
        parser: Parser function that returns dict with the fields
        field_names: Names of fields to compute stats for
    
    Returns:
        Dict mapping field name to (min, max) tuple
    """
    import numpy as np
    
    dataset = (
        tf.data.TFRecordDataset(files)
        .map(parser, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(100)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    stats = {
        field_name: (float('inf'), float('-inf'))
        for field_name in field_names
    }
    
    for batch in dataset:
        for field_name in field_names:
            if field_name not in batch:
                continue
            
            values = tf.reshape(batch[field_name], [-1]).numpy()
            values = values[~np.isnan(values)]
            values = values[values != 0.0]
            
            if len(values) == 0:
                continue
            
            current_min, current_max = stats[field_name]
            stats[field_name] = (
                min(current_min, float(np.min(values))),
                max(current_max, float(np.max(values)))
            )
    
    return stats


def compute_fields_mean_std(
    files: list[str],
    parser: Callable[[tf.Tensor], dict[str, tf.Tensor]],
    field_names: list[str],
) -> dict[str, tuple[float, float]]:
    """
    Compute mean and std for multiple fields using Welford's algorithm (has side effect: file I/O).
    
    Automatically filters out NaN and zero values (common masking approach).
    More efficient than calling compute_field_mean_std multiple times.
    
    Args:
        files: List of TFRecord file paths
        parser: Parser function that returns dict with the fields
        field_names: Names of fields to compute stats for
    
    Returns:
        Dict mapping field name to (mean, std) tuple
    """
    import numpy as np
    
    dataset = (
        tf.data.TFRecordDataset(files)
        .map(parser, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(100)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    stats = {
        field_name: {'count': 0, 'mean': 0.0, 'm2': 0.0}
        for field_name in field_names
    }
    
    for batch in dataset:
        for field_name in field_names:
            if field_name not in batch:
                continue
            
            values = tf.reshape(batch[field_name], [-1]).numpy()
            values = values[~np.isnan(values)]
            values = values[values != 0.0]
            
            if len(values) == 0:
                continue
            
            batch_count = len(values)
            batch_mean = np.mean(values)
            batch_m2 = np.sum((values - batch_mean) ** 2)
            
            field_stats = stats[field_name]
            count = field_stats['count']
            mean = field_stats['mean']
            m2 = field_stats['m2']
            
            new_count = count + batch_count
            delta = batch_mean - mean
            new_mean = mean + delta * batch_count / new_count
            new_m2 = m2 + batch_m2 + delta**2 * count * batch_count / new_count
            
            stats[field_name] = {
                'count': new_count,
                'mean': new_mean,
                'm2': new_m2
            }
    
    return {
        field_name: (
            float(field_stats['mean']),
            float(np.sqrt(field_stats['m2'] / (field_stats['count'] - 1)))
            if field_stats['count'] > 1 else 0.0
        )
        for field_name, field_stats in stats.items()
    }



def save_field_stats(
    path: str,
    stats: dict[str, dict[str, float]]
) -> None:
    """
    Save statistics for multiple fields to a JSON file (has side effect: file I/O).
    
    Args:
        path: Path to save JSON file
        stats: Dict mapping field names to their stats.
               Example: {"field1": {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.2}}
    """
    import json
    with open(path, 'w') as f:
        json.dump(stats, f, indent=2)


def load_field_stats(path: str) -> dict[str, dict[str, float]]:
    """
    Load statistics for multiple fields from a JSON file (has side effect: file I/O).
    
    Args:
        path: Path to JSON file
        
    Returns:
        Dict mapping field names to their stats
    """
    import json
    with open(path, 'r') as f:
        return json.load(f)