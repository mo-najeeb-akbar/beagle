"""Utilities for converting JAX/Flax models to TensorFlow.js format.

Uses the same hierarchical transfer approach as Keras conversion, but adapted
for TFJS weight format.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from beagle.conversions.types import ParamDict


DTYPE_MAP = {
    'float32': np.float32,
    'float16': np.float16,
    'int32': np.int32,
    'int16': np.int16,
    'int8': np.int8,
    'uint8': np.uint8,
    'bool': np.bool_,
}

DTYPE_BYTES = {
    'float32': 4,
    'float16': 2,
    'int32': 4,
    'int16': 2,
    'int8': 1,
    'uint8': 1,
    'bool': 1,
}


def _to_numpy(arr: Any) -> np.ndarray:
    """Convert JAX array to NumPy float32 (pure function)."""
    result = np.array(jnp.asarray(arr))
    if result.dtype == np.float64:
        result = result.astype(np.float32)
    return result


class TFJSWeightAdapter:
    """Adapter to make TFJS weights look like Keras layers.
    
    This allows us to use the same transfer infrastructure.
    """
    
    def __init__(
        self,
        name: str,
        weights_dict: dict[str, np.ndarray],
        layer_type: str = 'conv2d',
        debug: bool = False,
    ):
        """Initialize adapter with TFJS weight references.

        Args:
            name: Layer name pattern (e.g., 'conv2d_1', 'batch_normalization_0')
            weights_dict: Reference to full TFJS weights dict (will be modified)
            layer_type: Type of layer ('conv2d', 'dense', 'batch_normalization')
            debug: If True, print debug info about weight matching
        """
        self.name = name
        self.layer_type = layer_type
        self._weights_dict = weights_dict
        self._debug = debug
        self._weight_names = self._find_weight_names()
    
    def _find_weight_names(self) -> list[str]:
        """Find all TFJS weight names matching this layer.

        For BatchNorm, we need to find gamma, beta, mean, var separately.
        TFJS naming patterns from Keras:
        - Conv/Dense: */layer_name/*/kernel or bias
        - BatchNorm: */layer_name/*/gamma, beta, moving_mean, moving_variance

        We match by checking if the layer name appears as a path component.
        """
        matches = []

        if self._debug:
            print(f"\n[DEBUG] Finding weights for {self.name} ({self.layer_type})")
            print(f"[DEBUG] Searching in {len(self._weights_dict)} total weights")
            # Show a few matching candidates
            candidates = [k for k in self._weights_dict.keys() if self.name in k]
            if candidates:
                print(f"[DEBUG] Found {len(candidates)} candidates containing '{self.name}':")
                for c in candidates[:5]:
                    print(f"[DEBUG]   - {c}")

        if self.layer_type == 'batch_normalization':
            # BatchNorm - match all weights containing layer name
            # Standard names: gamma, beta, moving_mean, moving_variance
            # TF optimized: batchnorm/mul, batchnorm/sub, etc.
            param_patterns = [
                ('gamma', 'mul'),
                ('beta', 'sub'),
                ('moving_mean', 'mean'),
                ('moving_variance', 'variance', 'var'),
            ]

            for key in self._weights_dict.keys():
                if self.name in key.split('/'):
                    matches.append(key)
        else:
            # Conv/Dense - match all weights containing layer name
            # Standard names: kernel, bias
            # TF optimized: convolution/merged_input, ReadVariableOp, etc.
            for key in self._weights_dict.keys():
                # Check if layer name is in the path (as a complete path component)
                if self.name in key.split('/'):
                    matches.append(key)

        if self._debug:
            if matches:
                print(f"[DEBUG] Found {len(matches)} weights:")
                for m in matches:
                    shape = self._weights_dict[m].shape
                    print(f"[DEBUG]   - {m} (shape: {shape})")
            else:
                print(f"[DEBUG] WARNING: No weights found for {self.name}")

        return matches
    
    def get_weights(self) -> list[np.ndarray]:
        """Get current weights."""
        return [self._weights_dict[name] for name in self._weight_names]
    
    def set_weights(self, weights: list[np.ndarray]) -> None:
        """Set weights - updates the underlying dict."""
        if len(weights) != len(self._weight_names):
            raise ValueError(
                f"Expected {len(self._weight_names)} weights for {self.name} "
                f"({self.layer_type}), got {len(weights)}. "
                f"Weight names: {self._weight_names}"
            )
        for name, weight in zip(self._weight_names, weights):
            self._weights_dict[name] = weight


def load_tfjs_weights(model_path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load TFJS model weights and metadata.
    
    Args:
        model_path: Path to model.json or directory containing it
    
    Returns:
        Tuple of (weights_dict, model_info)
    """
    model_path = Path(model_path)
    
    if model_path.is_dir():
        model_dir = model_path
        model_path = model_path / 'model.json'
    else:
        model_dir = model_path.parent
    
    with open(model_path, 'r') as f:
        model_info = json.load(f)
    
    weights = {}
    
    for manifest in model_info.get('weightsManifest', []):
        bin_data = b''
        for bin_path in manifest.get('paths', []):
            with open(model_dir / bin_path, 'rb') as f:
                bin_data += f.read()
        
        offset = 0
        for spec in manifest.get('weights', []):
            name = spec['name']
            shape = spec['shape']
            dtype = spec['dtype']
            
            num_elements = int(np.prod(shape)) if shape else 1
            num_bytes = num_elements * DTYPE_BYTES.get(dtype, 4)
            
            weight_bytes = bin_data[offset:offset + num_bytes]
            offset += num_bytes
            
            arr = np.frombuffer(weight_bytes, dtype=DTYPE_MAP[dtype])
            weights[name] = arr.reshape(shape) if shape else arr
    
    return weights, model_info


def save_tfjs_weights(
    weights: dict[str, np.ndarray],
    model_info: dict,
    output_dir: str | Path,
    model_name: str = "model",
    validate: bool = True,
) -> None:
    """Save weights to TFJS format.

    Args:
        weights: Weight arrays by name
        model_info: Original model metadata
        output_dir: Output directory
        model_name: Base name for output files
        validate: If True, check for shape mismatches before saving
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build binary data in manifest order
    manifest = model_info['weightsManifest'][0]
    bin_data = b''

    # Validate shapes first if requested
    if validate:
        mismatches = []
        for spec in manifest['weights']:
            name = spec['name']
            if name not in weights:
                continue

            expected_shape = tuple(spec['shape'])
            actual_shape = weights[name].shape

            if expected_shape != actual_shape:
                mismatches.append({
                    'name': name,
                    'expected': expected_shape,
                    'actual': actual_shape,
                })

        if mismatches:
            print("\n" + "=" * 80)
            print("WARNING: Shape mismatches detected!")
            print("=" * 80)
            print(f"\nFound {len(mismatches)} shape mismatches:")
            for i, mm in enumerate(mismatches[:10]):  # Show first 10
                print(f"  {i+1}. {mm['name']}")
                print(f"     Expected: {mm['expected']}, Got: {mm['actual']}")
            if len(mismatches) > 10:
                print(f"  ... and {len(mismatches) - 10} more")
            print("\n" + "=" * 80)
            print("This indicates the TFJS template model has a different architecture")
            print("than the Flax checkpoint. Please verify you're using the correct template.")
            print("=" * 80 + "\n")

    for spec in manifest['weights']:
        name = spec['name']
        if name not in weights:
            raise KeyError(f"Missing weight: {name}")
        
        arr = weights[name]
        
        # Ensure correct dtype
        expected_dtype = DTYPE_MAP[spec['dtype']]
        if arr.dtype != expected_dtype:
            arr = arr.astype(expected_dtype)
        
        # Handle shape matching
        expected_shape = tuple(spec['shape'])
        
        if expected_shape == () and arr.shape == (1,):
            arr = arr.squeeze()
        elif arr.shape != expected_shape:
            if arr.size == (np.prod(expected_shape) if expected_shape else 1):
                arr = arr.reshape(expected_shape) if expected_shape else arr.item()
            else:
                raise ValueError(
                    f"Shape mismatch for {name}: "
                    f"expected {expected_shape}, got {arr.shape}"
                )
        
        bin_data += np.asarray(arr).tobytes()
    
    # Use original bin filename or default
    original_paths = manifest.get('paths', [])
    bin_filename = original_paths[0] if original_paths else f"{model_name}.weights.bin"
    
    with open(output_path / bin_filename, 'wb') as f:
        f.write(bin_data)
    
    # Save model.json
    new_model_info = model_info.copy()
    new_model_info['weightsManifest'] = [{
        'paths': [bin_filename],
        'weights': manifest['weights']
    }]
    
    with open(output_path / f"{model_name}.json", 'w') as f:
        json.dump(new_model_info, f, indent=2)
    
    print(f"Saved to {output_path}")
    print(f"  Weights: {len(weights)}")
    print(f"  Binary size: {len(bin_data):,} bytes ({len(bin_data)/1024/1024:.2f} MB)")


def fuse_batchnorm_into_conv(
    conv_kernel: np.ndarray,
    conv_bias: np.ndarray | None,
    bn_gamma: np.ndarray,
    bn_beta: np.ndarray,
    bn_mean: np.ndarray,
    bn_var: np.ndarray,
    epsilon: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse BatchNorm parameters into Conv layer (pure function).
    
    Given: y = BN(Conv(x))
    Returns: fused kernel and bias such that y = Conv_fused(x)
    
    BN formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
    """
    # Compute scale factor
    scale = bn_gamma / np.sqrt(bn_var + epsilon)
    
    # Fuse into kernel: multiply each output channel by scale
    if len(conv_kernel.shape) == 4:  # Conv2D: (H, W, in_ch, out_ch)
        fused_kernel = conv_kernel * scale[None, None, None, :]
    elif len(conv_kernel.shape) == 2:  # Dense: (in_features, out_features)
        fused_kernel = conv_kernel * scale[None, :]
    else:
        raise ValueError(f"Unsupported kernel shape: {conv_kernel.shape}")
    
    # Fuse into bias
    if conv_bias is None:
        conv_bias = np.zeros_like(bn_beta)
    
    fused_bias = scale * (conv_bias - bn_mean) + bn_beta
    
    return fused_kernel, fused_bias


def transfer_flax_to_tfjs(
    tfjs_weights: dict[str, np.ndarray],
    source_params: ParamDict,
    batch_stats: ParamDict | None = None,
    hierarchy_keys: list[str] | None = None,
    layer_type_patterns: dict[str, str] | None = None,
    fuse_batchnorm: bool = True,
    verbose: bool = False,
) -> dict[str, int]:
    """Transfer weights from hierarchical Flax params to TFJS weights.
    
    This uses the same logic as transfer_hierarchical_params but for TFJS format.
    BatchNorm layers are fused into preceding Conv/Dense layers if fuse_batchnorm=True.
    
    Args:
        tfjs_weights: TFJS weights dict (modified in-place)
        source_params: Flax params dict
        batch_stats: Optional Flax batch stats
        hierarchy_keys: Keys defining hierarchy (e.g., ['backbone'])
        layer_type_patterns: Dict mapping Flax prefixes to TFJS patterns
        fuse_batchnorm: If True, fuse BatchNorm into Conv/Dense layers
    
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
    
    # Build TFJS layer adapters by type
    # Extract layer names from TFJS weight paths and create ordered list
    layer_maps: dict[str, dict[int, TFJSWeightAdapter]] = {}
    for pattern in layer_type_patterns.values():
        layer_maps[pattern] = {}

    # Find all unique layer names by type in TFJS weights
    # Store as set first, then sort by numeric suffix to match Flax ordering
    layers_by_type: dict[str, list[str]] = {p: [] for p in layer_type_patterns.values()}
    layers_by_type_set: dict[str, set[str]] = {p: set() for p in layer_type_patterns.values()}

    for weight_name in tfjs_weights.keys():
        parts = weight_name.split('/')
        for part in parts:
            # Check if this part contains a layer type pattern
            for pattern in layer_type_patterns.values():
                if pattern in part:
                    layers_by_type_set[pattern].add(part)
                    break

    # Sort layers by numeric index extracted from name to match Flax ordering
    def extract_layer_index(layer_name: str) -> int:
        """Extract numeric index from layer name like 'conv2d_25_1' -> 25."""
        # Split by underscore and find the first numeric part
        parts = layer_name.split('_')
        for part in parts:
            if part.isdigit():
                return int(part)
        return 0

    for pattern in layer_type_patterns.values():
        layers_by_type[pattern] = sorted(layers_by_type_set[pattern], key=extract_layer_index)
        if verbose and layers_by_type[pattern]:
            print(f"\n  Sorted {pattern} layers (first 5): {layers_by_type[pattern][:5]}")

    # Create adapters with sequential indices (0, 1, 2, ...) based on sorted order
    # Enable debug on first adapter of each type to diagnose issues
    for pattern, layer_names in layers_by_type.items():
        for sequential_idx, layer_name in enumerate(layer_names):
            # Enable debug for first layer of each type when verbose is on
            debug = verbose and sequential_idx == 0
            adapter = TFJSWeightAdapter(layer_name, tfjs_weights, layer_type=pattern, debug=debug)
            layer_maps[pattern][sequential_idx] = adapter
            # Only show details for first few layers to reduce clutter
            if verbose and sequential_idx < 3:
                print(f"    TFJS {pattern}[{sequential_idx}] = {layer_name} "
                      f"({len(adapter._weight_names)} weights)")
        # Show summary
        if verbose and len(layer_names) > 3:
            print(f"    ... and {len(layer_names) - 3} more {pattern} layers")
    
    # Count layers in each hierarchy level
    hierarchy_counts: dict[str, dict[str, int]] = {}
    for hier_key in hierarchy_keys:
        hierarchy_counts[hier_key] = {}
        if hier_key in source_params:
            for jax_prefix in layer_type_patterns.keys():
                count = sum(
                    1 for k in source_params[hier_key].keys()
                    if k.startswith(jax_prefix)
                )
                hierarchy_counts[hier_key][jax_prefix] = count
                if verbose and count > 0:
                    print(f"    Flax {hier_key}/{jax_prefix}: {count} layers")

    # Count top-level params (heads)
    hierarchy_counts['heads'] = {}
    for jax_prefix in layer_type_patterns.keys():
        count = sum(1 for k in source_params.keys() if k.startswith(jax_prefix))
        hierarchy_counts['heads'][jax_prefix] = count
        if verbose and count > 0:
            print(f"    Flax heads/{jax_prefix}: {count} layers")
    
    # Build map of Flax Conv/Dense -> BatchNorm pairs for fusion
    # We match by shape - BatchNorm's channel count must match Conv's output channels
    # For TFJS optimized format, we also need to track the TFJS BN adapter indices
    conv_bn_pairs: dict[str, tuple[ParamDict, ParamDict, ParamDict]] = {}
    conv_to_bn_tfjs_idx: dict[int, int] = {}  # Maps conv TFJS idx -> bn TFJS idx
    
    if fuse_batchnorm and batch_stats:
        # Helper to get output channels from kernel
        def get_output_channels(kernel: Any) -> int:
            arr = _to_numpy(kernel)
            if len(arr.shape) == 4:  # Conv2D: (H, W, in_ch, out_ch)
                return arr.shape[3]
            elif len(arr.shape) == 2:  # Dense: (in_features, out_features)
                return arr.shape[1]
            return 0
        
        # Scan Flax params to find Conv+BatchNorm pairs by matching channel dimensions
        for hier_key in hierarchy_keys:
            if hier_key not in source_params:
                continue
            
            # Build dict of Conv/Dense layers with their output channels
            conv_layers = {}
            for key in source_params[hier_key].keys():
                if key.startswith('Conv_') or key.startswith('Dense_'):
                    if 'kernel' in source_params[hier_key][key]:
                        out_ch = get_output_channels(source_params[hier_key][key]['kernel'])
                        conv_layers[key] = out_ch
            
            # Match with BatchNorm by channel count and sequential ordering
            conv_keys_sorted = sorted(conv_layers.keys(), key=lambda k: int(k.split('_')[1]))
            bn_keys_sorted = sorted(
                [k for k in source_params[hier_key].keys() if k.startswith('BatchNorm_')],
                key=lambda k: int(k.split('_')[1])
            )
            
            # Try to pair Conv_i with the next BatchNorm that matches channels
            for conv_key in conv_keys_sorted:
                conv_idx = int(conv_key.split('_')[1])
                conv_out_ch = conv_layers[conv_key]
                
                # Find the next BatchNorm with matching channels
                for bn_key in bn_keys_sorted:
                    bn_idx = int(bn_key.split('_')[1])
                    if bn_idx >= conv_idx:  # BatchNorm should come after Conv
                        if bn_key in source_params[hier_key]:
                            bn_params = source_params[hier_key][bn_key]
                            if 'scale' in bn_params:
                                bn_ch = _to_numpy(bn_params['scale']).shape[0]
                                if bn_ch == conv_out_ch:
                                    # Found a match!
                                    if hier_key in batch_stats and bn_key in batch_stats[hier_key]:
                                        conv_params = source_params[hier_key][conv_key]
                                        bn_stats = batch_stats[hier_key][bn_key]
                                        conv_bn_pairs[f'{hier_key}/{conv_key}'] = (
                                            conv_params, bn_params, bn_stats
                                        )
                                        break
        
        # Also check top-level (heads) - simpler since less nesting
        conv_layers_head = {}
        for key in source_params.keys():
            if (key.startswith('Conv_') or key.startswith('Dense_')) and isinstance(source_params[key], dict):
                if 'kernel' in source_params[key]:
                    out_ch = get_output_channels(source_params[key]['kernel'])
                    conv_layers_head[key] = out_ch
        
        for conv_key, conv_out_ch in conv_layers_head.items():
            conv_idx = int(conv_key.split('_')[1])
            for bn_key in source_params.keys():
                if bn_key.startswith('BatchNorm_'):
                    bn_idx = int(bn_key.split('_')[1])
                    if bn_idx >= conv_idx and isinstance(source_params[bn_key], dict):
                        bn_params = source_params[bn_key]
                        if 'scale' in bn_params:
                            bn_ch = _to_numpy(bn_params['scale']).shape[0]
                            if bn_ch == conv_out_ch and bn_key in batch_stats:
                                conv_params = source_params[conv_key]
                                bn_stats = batch_stats[bn_key]
                                conv_bn_pairs[conv_key] = (conv_params, bn_params, bn_stats)
                                break
    
    # Transfer Conv/Dense layers (with optional BatchNorm fusion)
    stats: dict[str, int] = {}
    
    for jax_prefix in ['Conv_', 'Dense_']:
        tfjs_pattern = layer_type_patterns.get(jax_prefix)
        if not tfjs_pattern or tfjs_pattern not in layer_maps:
            continue
        
        tfjs_layers_of_type = layer_maps[tfjs_pattern]
        num_tfjs_layers = len(tfjs_layers_of_type)
        
        # Count layers in backbone vs heads
        num_backbone = 0
        for hier_key in hierarchy_keys:
            if hier_key in hierarchy_counts:
                num_backbone += hierarchy_counts[hier_key].get(jax_prefix, 0)
        
        num_heads = hierarchy_counts['heads'].get(jax_prefix, 0)
        
        # Transfer each layer
        for tfjs_idx in range(num_tfjs_layers):
            if tfjs_idx not in tfjs_layers_of_type:
                continue
            
            adapter = tfjs_layers_of_type[tfjs_idx]
            
            # Find the Flax params
            if tfjs_idx < num_backbone:
                # Backbone layer
                jax_idx = tfjs_idx
                params_dict = None
                flax_key = None
                
                for hier_key in hierarchy_keys:
                    jax_key = f'{jax_prefix}{jax_idx}'
                    if hier_key in source_params and jax_key in source_params[hier_key]:
                        params_dict = source_params[hier_key][jax_key]
                        flax_key = f'{hier_key}/{jax_key}'
                        break
            else:
                # Head layer
                jax_idx = tfjs_idx - num_backbone
                jax_key = f'{jax_prefix}{jax_idx}'
                
                if jax_key in source_params:
                    params_dict = source_params[jax_key]
                    flax_key = jax_key
                else:
                    params_dict = None
                    flax_key = None
            
            if params_dict is None:
                if verbose:
                    print(f"    Skipping {tfjs_pattern}[{tfjs_idx}]: no Flax params found")
                continue

            # Check if we should fuse BatchNorm
            if fuse_batchnorm and flax_key in conv_bn_pairs:
                conv_params, bn_params, bn_stats = conv_bn_pairs[flax_key]

                # Get the corresponding BatchNorm adapter
                # In optimized TFJS format, BN follows Conv with same index
                bn_pattern = layer_type_patterns.get('BatchNorm_', 'batch_normalization')
                if bn_pattern in layer_maps and tfjs_idx in layer_maps[bn_pattern]:
                    bn_adapter = layer_maps[bn_pattern][tfjs_idx]

                    if verbose:
                        print(f"    Transferring {tfjs_pattern}[{tfjs_idx}] + {bn_pattern}[{tfjs_idx}] "
                              f"<- Flax {flax_key} (fused)")

                    _transfer_conv_with_fused_bn_tfjs(
                        adapter, bn_adapter, conv_params, bn_params, bn_stats
                    )

                    # Mark this BN as processed so we don't try to transfer it again
                    conv_to_bn_tfjs_idx[tfjs_idx] = tfjs_idx
                else:
                    # No matching BN adapter, just transfer conv without fusion
                    if verbose:
                        print(f"    Transferring {tfjs_pattern}[{tfjs_idx}] <- Flax {flax_key} "
                              f"(BN adapter not found)")
                    _transfer_conv_or_dense_tfjs(adapter, params_dict)
            else:
                if verbose:
                    print(f"    Transferring {tfjs_pattern}[{tfjs_idx}] <- Flax {flax_key}")
                _transfer_conv_or_dense_tfjs(adapter, params_dict)
        
        transferred_count = min(num_tfjs_layers, num_backbone + num_heads)
        stats[tfjs_pattern] = stats.get(tfjs_pattern, 0) + transferred_count
    
    # Transfer BatchNorm layers (only those not already fused with Conv)
    if 'BatchNorm_' in layer_type_patterns:
        bn_tfjs_pattern = layer_type_patterns['BatchNorm_']
        if bn_tfjs_pattern in layer_maps:
            tfjs_layers_of_type = layer_maps[bn_tfjs_pattern]
            num_tfjs_layers = len(tfjs_layers_of_type)

            # Count layers in backbone vs heads
            num_backbone = 0
            for hier_key in hierarchy_keys:
                if hier_key in hierarchy_counts:
                    num_backbone += hierarchy_counts[hier_key].get('BatchNorm_', 0)

            num_heads = hierarchy_counts['heads'].get('BatchNorm_', 0)

            # Transfer each layer (skip those already fused)
            for tfjs_idx in range(num_tfjs_layers):
                # Skip if already processed during Conv fusion
                if tfjs_idx in conv_to_bn_tfjs_idx:
                    if verbose:
                        print(f"    Skipping {bn_tfjs_pattern}[{tfjs_idx}]: already fused with conv")
                    continue

                if tfjs_idx not in tfjs_layers_of_type:
                    continue

                adapter = tfjs_layers_of_type[tfjs_idx]

                # Find the Flax params (same logic as Conv)
                if tfjs_idx < num_backbone:
                    # Backbone layer
                    jax_idx = tfjs_idx
                    params_dict = None
                    batch_stats_dict = None

                    for hier_key in hierarchy_keys:
                        jax_key = f'BatchNorm_{jax_idx}'
                        if hier_key in source_params and jax_key in source_params[hier_key]:
                            params_dict = source_params[hier_key][jax_key]
                            if batch_stats and hier_key in batch_stats and jax_key in batch_stats[hier_key]:
                                batch_stats_dict = batch_stats[hier_key][jax_key]
                            break
                else:
                    # Head layer
                    jax_idx = tfjs_idx - num_backbone
                    jax_key = f'BatchNorm_{jax_idx}'

                    if jax_key in source_params:
                        params_dict = source_params[jax_key]
                        if batch_stats and jax_key in batch_stats:
                            batch_stats_dict = batch_stats[jax_key]
                    else:
                        params_dict = None
                        batch_stats_dict = None

                if params_dict is None:
                    if verbose:
                        print(f"    Skipping {bn_tfjs_pattern}[{tfjs_idx}]: no Flax params found")
                    continue

                if verbose:
                    print(f"    Transferring {bn_tfjs_pattern}[{tfjs_idx}] <- Flax BatchNorm_{jax_idx}")
                _transfer_batch_norm_tfjs(adapter, params_dict, batch_stats_dict)

            transferred_count = min(num_tfjs_layers, num_backbone + num_heads)
            stats[bn_tfjs_pattern] = stats.get(bn_tfjs_pattern, 0) + transferred_count

    return stats


def _transfer_conv_or_dense_tfjs(
    adapter: TFJSWeightAdapter,
    params: ParamDict
) -> None:
    """Transfer Conv/Dense weights to TFJS format.

    Handles both standard format (kernel + bias) and optimized format (just kernel).
    """
    kernel = _to_numpy(params['kernel'])
    num_weights = len(adapter._weight_names)

    if num_weights == 1:
        # Optimized format: just kernel
        adapter.set_weights([kernel])
    elif num_weights == 2:
        # Standard format: kernel + bias
        if 'bias' in params:
            bias = _to_numpy(params['bias'])
            adapter.set_weights([kernel, bias])
        else:
            # No bias in params, use zeros
            bias = np.zeros(kernel.shape[-1], dtype=kernel.dtype)
            adapter.set_weights([kernel, bias])
    else:
        # Fallback: try to set what we have
        if 'bias' in params:
            bias = _to_numpy(params['bias'])
            adapter.set_weights([kernel, bias][:num_weights])
        else:
            adapter.set_weights([kernel])


def _transfer_conv_with_fused_bn_tfjs(
    conv_adapter: TFJSWeightAdapter,
    bn_adapter: TFJSWeightAdapter,
    conv_params: ParamDict,
    bn_params: ParamDict,
    bn_stats: ParamDict,
    epsilon: float = 1e-5,
) -> None:
    """Transfer Conv/Dense with fused BatchNorm to TFJS optimized format.

    Handles two cases:
    1. Conv has 1 weight (kernel): Write fused_kernel to conv, fused_bias to bn
    2. Conv has 2 weights (kernel+bias): Write both fused weights to conv
    """
    kernel = _to_numpy(conv_params['kernel'])
    conv_bias = _to_numpy(conv_params['bias']) if 'bias' in conv_params else None

    gamma = _to_numpy(bn_params['scale'])
    beta = _to_numpy(bn_params['bias'])
    mean = _to_numpy(bn_stats['mean'])
    var = _to_numpy(bn_stats['var'])

    # Fuse BatchNorm into Conv
    fused_kernel, fused_bias = fuse_batchnorm_into_conv(
        kernel, conv_bias, gamma, beta, mean, var, epsilon
    )

    num_conv_weights = len(conv_adapter._weight_names)
    num_bn_weights = len(bn_adapter._weight_names)

    if num_conv_weights == 1:
        # Conv has only kernel - write fused kernel there, fused bias to BN
        conv_adapter.set_weights([fused_kernel])
        if num_bn_weights == 1:
            bn_adapter.set_weights([fused_bias])
    elif num_conv_weights == 2:
        # Conv has kernel + bias - write both fused weights to conv
        conv_adapter.set_weights([fused_kernel, fused_bias])
    else:
        # Fallback: try to match what we have
        conv_adapter.set_weights([fused_kernel])
        if num_bn_weights == 1:
            bn_adapter.set_weights([fused_bias])


def _transfer_batch_norm_tfjs(
    adapter: TFJSWeightAdapter,
    params: ParamDict,
    batch_stats: ParamDict | None = None
) -> None:
    """Transfer BatchNorm weights to TFJS format (for unfused case).

    In TensorFlow's optimized format, BatchNorm often has just 1 weight (batchnorm/sub)
    which represents the bias term. The scale is baked into the preceding conv layer.
    """
    # In optimized format, we typically only have the bias term
    # Standard format would have [gamma, beta, mean, var]
    beta = _to_numpy(params['bias'])

    # Check how many weights the adapter expects
    num_weights = len(adapter._weight_names)

    if num_weights == 1:
        # Optimized format: just the bias
        adapter.set_weights([beta])
    elif num_weights == 4:
        # Standard format: gamma, beta, mean, var
        gamma = _to_numpy(params['scale'])

        if batch_stats and 'mean' in batch_stats and 'var' in batch_stats:
            mean = _to_numpy(batch_stats['mean'])
            var = _to_numpy(batch_stats['var'])
        else:
            mean = np.zeros_like(gamma)
            var = np.ones_like(gamma)

        adapter.set_weights([gamma, beta, mean, var])
    else:
        # Try to match whatever format we have
        weights_to_set = [beta]
        if num_weights > 1 and 'scale' in params:
            weights_to_set.insert(0, _to_numpy(params['scale']))
        adapter.set_weights(weights_to_set[:num_weights])
