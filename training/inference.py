"""Pure inference utilities for model evaluation (separate from training)."""

from __future__ import annotations

from typing import Callable, Any
import jax
import jax.numpy as jnp

from beagle.training.types import TrainState


def create_inference_fn(
    apply_fn: Callable,
    use_batch_stats: bool = True,
    training: bool = False
) -> Callable[[TrainState, Any, Any], Any]:
    """Create a jitted inference function.
    
    Higher-order function that returns a pure, jitted inference function.
    
    Args:
        apply_fn: Model's apply function
        use_batch_stats: Whether model uses batch normalization
        training: Whether to run in training mode
        
    Returns:
        Jitted inference function with signature:
            (state, inputs, rng_key) -> outputs
    
    Example:
        >>> inference_fn = create_inference_fn(model.apply, use_batch_stats=True)
        >>> outputs = inference_fn(state, batch, key)
    """
    @jax.jit
    def inference(state: TrainState, inputs: Any, rng_key: Any) -> Any:
        """Pure inference (jitted)."""
        if use_batch_stats and state.batch_stats is not None:
            return apply_fn(
                {'params': state.params, 'batch_stats': state.batch_stats},
                inputs,
                training=training,
                key=rng_key
            )
        else:
            return apply_fn(
                {'params': state.params},
                inputs,
                training=training,
                key=rng_key
            )
    
    return inference


def batch_inference(
    state: TrainState,
    inference_fn: Callable[[TrainState, Any, Any], Any],
    data_iterator: Any,
    num_batches: int,
    rng_key: Any
) -> list[Any]:
    """Run inference on multiple batches (orchestrates pure inference).
    
    Args:
        state: Training state
        inference_fn: Jitted inference function
        data_iterator: Iterator yielding batches
        num_batches: Number of batches to process
        rng_key: JAX random key
        
    Returns:
        List of outputs for each batch
    """
    import jax.random as random
    
    outputs = []
    
    for _ in range(num_batches):
        rng_key, step_key = random.split(rng_key)
        batch = next(data_iterator)
        
        output = inference_fn(state, batch, step_key)
        outputs.append(output)
    
    return outputs

