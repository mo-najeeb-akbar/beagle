"""Visualization callbacks for training loops (orchestrates pure + side effects)."""

from __future__ import annotations

from typing import Callable, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from beagle.visualization.plotting import plot_comparison, save_figure


# Type alias for callbacks
VizCallback = Callable[[Any, Any, Any, int], None]


@dataclass(frozen=True)
class VizConfig:
    """Configuration for visualization callbacks.
    
    Args:
        plot_every: Visualize every N epochs (None = never)
        num_samples: Number of samples to visualize
        output_dir: Directory to save plots (None = show only)
        format: Output format (png, jpg, pdf, etc.)
    """
    plot_every: int | None = 5
    num_samples: int = 4
    output_dir: str | None = None
    format: str = 'png'


def to_numpy(x: Any) -> np.ndarray:
    """Convert JAX array or numpy array to numpy (pure function)."""
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def create_viz_callback(
    viz_fn: Callable[[Any, Any, Any], dict[str, jnp.ndarray]],
    config: VizConfig,
    extract_inputs_fn: Callable[[Any], Any] | None = None
) -> VizCallback:
    """Create a visualization callback for training loops.
    
    This is a higher-order function that creates a callback combining:
    - A pure inference/processing function (jitted)
    - Matplotlib plotting (side effect)
    - File saving (side effect)
    
    Args:
        viz_fn: Pure function (state, batch, rng_key) -> dict[str, arrays]
            Should return dict with keys like 'input', 'prediction', 'target'
        config: Visualization configuration
        extract_inputs_fn: Optional function to extract inputs from batch
            
    Returns:
        Callback with signature: (state, batch, rng_key, epoch) -> None
        
    Example:
        >>> @jax.jit
        >>> def inference(state, batch, rng_key):
        ...     preds = state.apply_fn({'params': state.params}, batch['image'])
        ...     return {'input': batch['image'], 'prediction': preds}
        >>> 
        >>> viz_config = VizConfig(plot_every=5, output_dir="/data/viz")
        >>> callback = create_viz_callback(inference, viz_config)
        >>> 
        >>> # In training loop:
        >>> callback(state, batch, rng_key, epoch)
    """
    def callback(state: Any, batch: Any, rng_key: Any, epoch: int) -> None:
        """Visualization callback (orchestrates pure + side effects)."""
        # Check if we should visualize
        if config.plot_every is None:
            return
        
        if (epoch + 1) % config.plot_every != 0:
            return
        
        # Extract inputs if needed
        if extract_inputs_fn is not None:
            inputs = extract_inputs_fn(batch)
        else:
            inputs = batch
        
        # Run visualization function (pure computation)
        outputs = viz_fn(state, inputs, rng_key)
        
        # Convert to numpy and limit samples (pure transformation)
        outputs_np = {
            k: to_numpy(v)[:config.num_samples] 
            for k, v in outputs.items()
        }
        
        # Create plot (side effect)
        fig = plot_comparison(
            outputs_np,
            num_samples=config.num_samples,
            suptitle=f"Epoch {epoch + 1}"
        )
        
        # Save or show (side effect)
        if config.output_dir is not None:
            output_path = Path(config.output_dir) / f"epoch_{epoch+1:04d}.{config.format}"
            save_figure(fig, output_path, close=True)
        else:
            import matplotlib.pyplot as plt
            plt.show()
            plt.close(fig)
    
    return callback


def create_simple_reconstruction_callback(
    apply_fn: Callable,
    config: VizConfig,
    input_key: str = 'image',
    training: bool = False
) -> VizCallback:
    """Create a callback for simple reconstruction models (convenience wrapper).
    
    Args:
        apply_fn: Model's apply function
        config: Visualization configuration
        input_key: Key to extract inputs from batch dict
        training: Whether to run in training mode
        
    Returns:
        Visualization callback
        
    Example:
        >>> config = VizConfig(plot_every=10, output_dir="/data/viz")
        >>> callback = create_simple_reconstruction_callback(
        ...     model.apply, config, input_key='image'
        ... )
    """
    import jax
    
    @jax.jit
    def viz_fn(state, batch, rng_key):
        """Run inference and package outputs (pure, jitted)."""
        inputs = batch[input_key] if isinstance(batch, dict) else batch
        
        # Run model
        if hasattr(state, 'batch_stats') and state.batch_stats is not None:
            outputs = apply_fn(
                {'params': state.params, 'batch_stats': state.batch_stats},
                inputs,
                training=training,
                key=rng_key
            )
        else:
            outputs = apply_fn(
                {'params': state.params},
                inputs,
                training=training,
                key=rng_key
            )
        
        # Extract reconstruction (handle different output formats)
        if isinstance(outputs, tuple):
            reconstruction = outputs[0]
        else:
            reconstruction = outputs
        
        return {
            'Input': inputs,
            'Reconstruction': reconstruction
        }
    
    return create_viz_callback(viz_fn, config)

