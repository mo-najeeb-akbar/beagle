"""
Flexible preprocessing for different data types.

Provides configurable preprocessing for images, segmentation maps, labels, etc.
Each field can have its own preprocessing pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal
from enum import Enum
from pathlib import Path
import json

import tensorflow as tf


class FieldType(Enum):
    """Data field types for preprocessing."""
    IMAGE = "image"  # Float data that should be standardized
    MASK = "mask"  # Integer segmentation mask (no standardization)
    LABEL = "label"  # Integer or categorical label
    VECTOR = "vector"  # Float vector (e.g., bounding box, regression target)
    RAW = "raw"  # No preprocessing


@dataclass(frozen=True)
class FieldConfig:
    """Configuration for a single data field (immutable)."""
    name: str
    field_type: FieldType
    standardize: bool = True  # Whether to standardize (only for IMAGE/VECTOR types)
    stats: tuple[float, float] | None = None  # (mean, std) for standardization
    dtype: tf.DType = tf.float32  # Output dtype


def create_standardize_fn(
    field_configs: dict[str, FieldConfig],
) -> Callable[[dict[str, tf.Tensor]], dict[str, tf.Tensor]]:
    """
    Create standardization function from field configs (pure).
    
    Args:
        field_configs: Dictionary mapping field names to their configurations
    
    Returns:
        Function that standardizes specified fields
    
    Examples:
        >>> configs = {
        ...     'image': FieldConfig('image', FieldType.IMAGE, stats=(0.5, 0.2)),
        ...     'mask': FieldConfig('mask', FieldType.MASK, standardize=False),
        ...     'label': FieldConfig('label', FieldType.LABEL),
        ... }
        >>> standardize = create_standardize_fn(configs)
        >>> data = {'image': img_tensor, 'mask': mask_tensor, 'label': label_tensor}
        >>> standardized = standardize(data)
    """
    def standardize(data_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        result = {}
        for name, tensor in data_dict.items():
            if name not in field_configs:
                # Pass through unknown fields unchanged
                result[name] = tensor
                continue
            
            config = field_configs[name]
            
            # Apply standardization if requested and stats available
            if config.standardize and config.stats is not None:
                mean, std = config.stats
                result[name] = (tensor - mean) / (std + 1e-8)
            else:
                result[name] = tensor
            
            # Cast to output dtype if different
            if tensor.dtype != config.dtype:
                result[name] = tf.cast(result[name], config.dtype)
        
        return result
    
    return standardize


def compute_field_stats(
    files: list[str],
    parser: Callable[[tf.Tensor], dict[str, tf.Tensor]],
    field_name: str,
) -> tuple[float, float]:
    """
    Compute mean/std for a specific field using Welford's algorithm (has side effect: file I/O).
    
    Works with any numeric field type (images, depth maps, regression targets, etc.).
    Automatically filters out NaN and zero values (common masking approach).
    
    Args:
        files: List of TFRecord file paths
        parser: Parser function that returns dict with the field
        field_name: Name of field to compute stats for
    
    Returns:
        (mean, std) tuple
    """
    import numpy as np
    
    dataset = (
        tf.data.TFRecordDataset(files)
        .map(parser, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(100)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    # Welford's algorithm (vectorized for batches)
    count = 0
    mean = 0.0
    m2 = 0.0
    
    for batch in dataset:
        if field_name not in batch:
            continue
        
        values = tf.reshape(batch[field_name], [-1]).numpy()
        
        # Filter invalid values
        values = values[~np.isnan(values)]
        values = values[values != 0.0]
        
        if len(values) == 0:
            continue
        
        # Batched Welford update
        batch_count = len(values)
        batch_mean = np.mean(values)
        batch_m2 = np.sum((values - batch_mean) ** 2)
        
        new_count = count + batch_count
        delta = batch_mean - mean
        mean = mean + delta * batch_count / new_count
        m2 = m2 + batch_m2 + delta**2 * count * batch_count / new_count
        count = new_count
    
    std = float(np.sqrt(m2 / (count - 1))) if count > 1 else 0.0
    return float(mean), std


def compute_stats_for_fields(
    files: list[str],
    parser: Callable[[tf.Tensor], dict[str, tf.Tensor]],
    field_configs: dict[str, FieldConfig],
) -> dict[str, FieldConfig]:
    """
    Compute stats for all fields that need standardization (has side effect: file I/O).
    
    Args:
        files: List of TFRecord file paths
        parser: Parser function
        field_configs: Field configurations (will be updated with computed stats)
    
    Returns:
        Updated field configs with computed stats
    """
    updated_configs = {}
    
    for name, config in field_configs.items():
        if config.standardize and config.stats is None:
            # Need to compute stats
            if config.field_type in (FieldType.IMAGE, FieldType.VECTOR):
                print(f"Computing stats for field '{name}'...")
                mean, std = compute_field_stats(files, parser, name)
                print(f"  mean={mean:.6f}, std={std:.6f}")
                # Create new config with stats (dataclass is frozen)
                updated_configs[name] = FieldConfig(
                    name=config.name,
                    field_type=config.field_type,
                    standardize=config.standardize,
                    stats=(mean, std),
                    dtype=config.dtype,
                )
            else:
                # Can't standardize non-numeric types
                updated_configs[name] = config
        else:
            updated_configs[name] = config
    
    return updated_configs


def save_field_stats(
    field_configs: dict[str, FieldConfig],
    output_path: str | Path,
) -> None:
    """
    Save field statistics to JSON file for later use during inference (has side effect: file I/O).
    
    Args:
        field_configs: Field configurations with computed stats
        output_path: Path to save JSON file (e.g., 'dataset_stats.json')
    
    Example:
        >>> # During training
        >>> configs = compute_stats_for_fields(files, parser, field_configs)
        >>> save_field_stats(configs, 'stats.json')
        >>> 
        >>> # During inference
        >>> configs = load_field_stats('stats.json')
        >>> iterator = create_iterator(..., field_configs=configs)
    """
    output_path = Path(output_path)
    
    stats_dict = {}
    for name, config in field_configs.items():
        if config.stats is not None:
            mean, std = config.stats
            stats_dict[name] = {
                'mean': float(mean),
                'std': float(std),
                'field_type': config.field_type.value,
                'standardize': config.standardize,
            }
    
    with open(output_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    
    print(f"Saved stats for {len(stats_dict)} field(s) to {output_path}")


def load_field_stats(
    stats_path: str | Path,
    field_configs: dict[str, FieldConfig] | None = None,
) -> dict[str, FieldConfig]:
    """
    Load field statistics from JSON file (has side effect: file I/O).
    
    Args:
        stats_path: Path to JSON file with saved stats
        field_configs: Optional existing configs to update with loaded stats
                      If None, creates new configs from saved stats
    
    Returns:
        Field configurations with loaded stats
    
    Example:
        >>> # Load stats and create configs
        >>> configs = load_field_stats('stats.json')
        >>> iterator = create_iterator(..., field_configs=configs)
        >>> 
        >>> # Or update existing configs
        >>> configs = {'depth': FieldConfig('depth', FieldType.IMAGE)}
        >>> configs = load_field_stats('stats.json', field_configs=configs)
    """
    stats_path = Path(stats_path)
    
    with open(stats_path, 'r') as f:
        stats_dict = json.load(f)
    
    loaded_configs = {}
    
    for name, stats_data in stats_dict.items():
        mean = stats_data['mean']
        std = stats_data['std']
        field_type = FieldType(stats_data['field_type'])
        standardize = stats_data.get('standardize', True)
        
        # If existing configs provided, update them
        if field_configs and name in field_configs:
            existing = field_configs[name]
            loaded_configs[name] = FieldConfig(
                name=name,
                field_type=existing.field_type,
                standardize=existing.standardize,
                stats=(mean, std),
                dtype=existing.dtype,
            )
        else:
            # Create new config from saved data
            loaded_configs[name] = FieldConfig(
                name=name,
                field_type=field_type,
                standardize=standardize,
                stats=(mean, std),
            )
    
    # Include any configs not in stats file
    if field_configs:
        for name, config in field_configs.items():
            if name not in loaded_configs:
                loaded_configs[name] = config
    
    print(f"Loaded stats for {len(stats_dict)} field(s) from {stats_path}")
    return loaded_configs

