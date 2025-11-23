"""Utilities for converting hierarchical JAX/Flax models to flat target models.

Handles cases where JAX params are nested (e.g., backbone + heads) but the
target framework (TensorFlow/Keras) has a flat layer structure.
"""
from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from beagle.conversions.dispatch import ConversionRegistry
from beagle.conversions.types import ParamDict


def _to_numpy(arr: Any) -> np.ndarray:
    """Convert JAX array to NumPy (pure function)."""
    return np.array(jnp.asarray(arr))


def transfer_hierarchical_params(
    target_model: Any,
    source_params: ParamDict,
    batch_stats: ParamDict | None = None,
    hierarchy_keys: list[str] | None = None,
    layer_type_patterns: dict[str, str] | None = None,
) -> dict[str, int]:
    """Transfer weights from hierarchical JAX params to flat Keras model.
    
    Args:
        target_model: Keras model with flat layer structure
        source_params: JAX params dict (may be nested with hierarchy_keys)
        batch_stats: Optional JAX batch stats (for BatchNorm layers)
        hierarchy_keys: Keys defining hierarchy (e.g., ['backbone'])
        layer_type_patterns: Dict mapping layer type prefixes to Keras patterns
                           (e.g., {'Conv_': 'conv2d', 'BatchNorm_': 'batch_normalization'})
    
    Returns:
        Dict with counts per layer type transferred
    """
    if hierarchy_keys is None:
        hierarchy_keys = ['backbone']
    
    if layer_type_patterns is None:
        layer_type_patterns = {
            'Conv_': 'conv2d',
            'Dense_': 'dense',
            'BatchNorm_': 'batch_normalization',
        }
    
    # Get all Keras layers with weights
    keras_layers = [layer for layer in target_model.layers if len(layer.weights) > 0]
    
    # Build layer name mappings by type
    layer_maps: dict[str, dict[int, Any]] = {}
    for pattern in layer_type_patterns.values():
        layer_maps[pattern] = {}
    
    for layer in keras_layers:
        layer_name = layer.name
        for pattern in layer_type_patterns.values():
            if pattern in layer_name:
                # Extract index from layer name
                parts = layer_name.split('_')
                idx = 0 if len(parts) == 1 or not parts[-1].isdigit() else int(parts[-1])
                layer_maps[pattern][idx] = layer
                break
    
    # Count layers in each hierarchy level
    hierarchy_counts: dict[str, dict[str, int]] = {}
    for hier_key in hierarchy_keys:
        hierarchy_counts[hier_key] = {}
        if hier_key in source_params:
            for jax_prefix in layer_type_patterns.keys():
                count = sum(1 for k in source_params[hier_key].keys() if k.startswith(jax_prefix))
                hierarchy_counts[hier_key][jax_prefix] = count
    
    # Also count top-level params (prediction heads)
    hierarchy_counts['heads'] = {}
    for jax_prefix in layer_type_patterns.keys():
        count = sum(1 for k in source_params.keys() if k.startswith(jax_prefix))
        hierarchy_counts['heads'][jax_prefix] = count
    
    # Transfer each layer type
    stats: dict[str, int] = {}
    
    for jax_prefix, keras_pattern in layer_type_patterns.items():
        if keras_pattern not in layer_maps or not layer_maps[keras_pattern]:
            continue
        
        keras_layers_of_type = layer_maps[keras_pattern]
        num_keras_layers = len(keras_layers_of_type)
        
        # Determine how many are in backbone vs heads
        num_backbone = 0
        for hier_key in hierarchy_keys:
            if hier_key in hierarchy_counts:
                num_backbone += hierarchy_counts[hier_key].get(jax_prefix, 0)
        
        num_heads = hierarchy_counts['heads'].get(jax_prefix, 0)
        
        # Transfer each layer
        for keras_idx in range(num_keras_layers):
            if keras_idx not in keras_layers_of_type:
                continue
            
            layer = keras_layers_of_type[keras_idx]
            
            # Determine if this is a backbone or head layer
            if keras_idx < num_backbone:
                # Backbone layer - find in hierarchy
                jax_idx = keras_idx
                params_dict = None
                
                for hier_key in hierarchy_keys:
                    jax_key = f'{jax_prefix}{jax_idx}'
                    if hier_key in source_params and jax_key in source_params[hier_key]:
                        params_dict = source_params[hier_key][jax_key]
                        break
                
                if params_dict is None:
                    continue
                
                # Transfer weights based on layer type
                if keras_pattern in ['conv2d', 'dense']:
                    _transfer_conv_or_dense(layer, params_dict)
                elif keras_pattern == 'batch_normalization':
                    # Get batch stats from hierarchy
                    batch_stats_dict = None
                    if batch_stats:
                        for hier_key in hierarchy_keys:
                            if hier_key in batch_stats and jax_key in batch_stats[hier_key]:
                                batch_stats_dict = batch_stats[hier_key][jax_key]
                                break
                    _transfer_batch_norm(layer, params_dict, batch_stats_dict)
            else:
                # Head layer
                jax_idx = keras_idx - num_backbone
                jax_key = f'{jax_prefix}{jax_idx}'
                
                if jax_key not in source_params:
                    continue
                
                params_dict = source_params[jax_key]
                
                # Transfer weights based on layer type
                if keras_pattern in ['conv2d', 'dense']:
                    _transfer_conv_or_dense(layer, params_dict)
                elif keras_pattern == 'batch_normalization':
                    # Get batch stats from top level
                    batch_stats_dict = None
                    if batch_stats and jax_key in batch_stats:
                        batch_stats_dict = batch_stats[jax_key]
                    _transfer_batch_norm(layer, params_dict, batch_stats_dict)
        
        # Update stats
        transferred_count = min(num_keras_layers, num_backbone + num_heads)
        stats[keras_pattern] = stats.get(keras_pattern, 0) + transferred_count
    
    return stats


def _transfer_conv_or_dense(layer: Any, params: ParamDict) -> None:
    """Transfer Conv2D or Dense layer weights (pure data transformation)."""
    kernel = _to_numpy(params['kernel'])
    
    if 'bias' in params:
        bias = _to_numpy(params['bias'])
        layer.set_weights([kernel, bias])
    else:
        layer.set_weights([kernel])


def _transfer_batch_norm(
    layer: Any,
    params: ParamDict,
    batch_stats: ParamDict | None = None
) -> None:
    """Transfer BatchNormalization layer weights (pure data transformation)."""
    gamma = _to_numpy(params['scale'])
    beta = _to_numpy(params['bias'])
    
    if batch_stats and 'mean' in batch_stats and 'var' in batch_stats:
        mean = _to_numpy(batch_stats['mean'])
        var = _to_numpy(batch_stats['var'])
    else:
        # Fallback to zeros and ones
        mean = np.zeros_like(gamma)
        var = np.ones_like(gamma)
    
    layer.set_weights([gamma, beta, mean, var])

