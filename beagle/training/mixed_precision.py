"""Mixed precision training utilities for JAX/Flax.

Provides safe mixed precision training with:
- Compute in bfloat16 for speed (2x memory bandwidth)
- Master weights stay in float32
- Loss computations in float32 for stability
- Outputs in float32

Key principle: Only cast at boundaries (input/output), never change weight dtypes.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar
from functools import wraps

import jax
import jax.numpy as jnp
from flax import struct

T = TypeVar("T")


@struct.dataclass
class MixedPrecisionPolicy:
    """Immutable policy for mixed precision training.
    
    Attributes:
        compute_dtype: dtype for internal computations (bfloat16/float16)
        param_dtype: dtype for parameters (always float32)
        output_dtype: dtype for outputs and loss (always float32)
    """
    
    compute_dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32
    output_dtype: jnp.dtype = jnp.float32


def create_mixed_precision_policy(
    compute_dtype: str = "bfloat16"
) -> MixedPrecisionPolicy:
    """Create mixed precision policy (pure function).
    
    Args:
        compute_dtype: "bfloat16" or "float16" for internal compute
        
    Returns:
        Immutable policy object
    """
    dtype_map = {
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
        "float32": jnp.float32,
    }
    
    if compute_dtype not in dtype_map:
        raise ValueError(f"compute_dtype must be one of {list(dtype_map.keys())}")
    
    return MixedPrecisionPolicy(
        compute_dtype=dtype_map[compute_dtype],
        param_dtype=jnp.float32,
        output_dtype=jnp.float32,
    )


def cast_inputs_to_compute(
    inputs: Any,
    policy: MixedPrecisionPolicy
) -> Any:
    """Cast input tensors to compute dtype (pure function).
    
    Args:
        inputs: Input arrays or nested structure
        policy: Mixed precision policy
        
    Returns:
        Inputs casted to compute dtype
    """
    def cast_fn(x):
        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(policy.compute_dtype)
        return x
    
    return jax.tree.map(cast_fn, inputs)


def cast_outputs_to_float32(outputs: Any) -> Any:
    """Cast output tensors to float32 (pure function).
    
    Always cast outputs back to float32 for:
    - Loss computation stability
    - Visualization
    - Metrics
        
    Args:
        outputs: Output arrays or nested structure
        
    Returns:
        Outputs casted to float32
    """
    def cast_fn(x):
        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(jnp.float32)
        return x
    
    return jax.tree.map(cast_fn, outputs)


def wrap_forward_with_mixed_precision(
    forward_fn: Callable,
    policy: MixedPrecisionPolicy
) -> Callable:
    """Wrap model forward pass with mixed precision casts (pure).
    
    Pattern:
        input (float32) → cast → compute (bfloat16) → cast → output (float32)
    
    Args:
        forward_fn: Original forward function
        policy: Mixed precision policy
        
    Returns:
        Wrapped forward function with automatic casting
        
    Example:
        >>> policy = create_mixed_precision_policy("bfloat16")
        >>> def forward(params, x):
        ...     return model.apply(params, x)
        >>> mixed_forward = wrap_forward_with_mixed_precision(forward, policy)
        >>> # Inputs/outputs stay float32, compute is bfloat16
        >>> output = mixed_forward(params, x_float32)  # output is float32
    """
    @wraps(forward_fn)
    def wrapped(*args, **kwargs):
        # Cast inputs to compute dtype
        args_compute = cast_inputs_to_compute(args, policy)
        kwargs_compute = cast_inputs_to_compute(kwargs, policy)
        
        # Run computation in low precision
        outputs = forward_fn(*args_compute, **kwargs_compute)
        
        # Cast outputs back to float32
        return cast_outputs_to_float32(outputs)
    
    return wrapped


def create_mixed_precision_train_step(
    train_step_fn: Callable,
    policy: MixedPrecisionPolicy | None = None
) -> Callable:
    """Wrap training step with mixed precision (pure function).
    
    Automatically handles:
    - Casting inputs to compute dtype
    - Computing in low precision
    - Casting loss/gradients to float32
    - Keeping optimizer state in float32
    
    Args:
        train_step_fn: Original training step function
        policy: Mixed precision policy (None = default bfloat16)
        
    Returns:
        Wrapped training step with mixed precision
        
    Example:
        >>> @jax.jit
        ... def train_step(state, batch, rng_key):
        ...     def loss_fn(params):
        ...         pred = state.apply_fn({'params': params}, batch['x'])
        ...         return jnp.mean((pred - batch['y'])**2)
        ...     loss, grads = jax.value_and_grad(loss_fn)(state.params)
        ...     state = state.apply_gradients(grads=grads)
        ...     return state, {'loss': loss}
        ...
        >>> # Wrap with mixed precision
        >>> policy = create_mixed_precision_policy("bfloat16")
        >>> mp_train_step = create_mixed_precision_train_step(train_step, policy)
        >>> # Use exactly like original train_step!
        >>> state, metrics = mp_train_step(state, batch, rng_key)
    """
    if policy is None:
        policy = create_mixed_precision_policy("bfloat16")
    
    @wraps(train_step_fn)
    def wrapped_train_step(state, batch, *args, **kwargs):
        # Cast batch inputs to compute dtype
        batch_compute = cast_inputs_to_compute(batch, policy)
        
        # Run training step (internally casts to compute dtype)
        # Gradients automatically computed in compute dtype
        state, metrics = train_step_fn(state, batch_compute, *args, **kwargs)
        
        # Cast metrics to float32 for logging
        metrics_float32 = cast_outputs_to_float32(metrics)
        
        return state, metrics_float32
    
    return wrapped_train_step


def get_recommended_policy() -> MixedPrecisionPolicy:
    """Get recommended mixed precision policy based on hardware (pure).
    
    Returns:
        Policy optimized for current hardware
        
    Rules:
        - Ampere+ (A100, RTX 30x0): bfloat16 (native support, 2x faster)
        - Volta/Turing (V100, T4): float16 (no bfloat16, but float16 works)
        - TPU: bfloat16 (native support)
        - CPU: float32 (no speedup from low precision)
    """
    # Check device type
    devices = jax.devices()
    if not devices:
        return MixedPrecisionPolicy()  # Default bfloat16
    
    device_kind = devices[0].device_kind.lower()
    
    # TPU: always use bfloat16 (native support)
    if "tpu" in device_kind:
        return create_mixed_precision_policy("bfloat16")
    
    # CPU: no benefit from low precision
    if "cpu" in device_kind:
        return create_mixed_precision_policy("float32")
    
    # GPU: bfloat16 is safe default (works on Ampere+, degrades gracefully on older)
    # Ampere (A100, 3090, 4090): native bfloat16, 2x faster
    # Older GPUs: bfloat16 emulated but still safe
    return create_mixed_precision_policy("bfloat16")


# Convenience function for quick setup
def enable_mixed_precision(
    train_step_fn: Callable,
    policy: str | MixedPrecisionPolicy | None = None
) -> Callable:
    """Convenience function to enable mixed precision (pure).
    
    Args:
        train_step_fn: Training step function to wrap
        policy: "auto", "bfloat16", "float16", or MixedPrecisionPolicy object
                "auto" = use hardware recommendation
                
    Returns:
        Mixed precision training step
        
    Example:
        >>> # Simplest usage: auto-detect best dtype
        >>> mp_train_step = enable_mixed_precision(train_step, policy="auto")
        >>> 
        >>> # Explicit dtype
        >>> mp_train_step = enable_mixed_precision(train_step, policy="bfloat16")
    """
    if policy is None or policy == "auto":
        policy_obj = get_recommended_policy()
    elif isinstance(policy, str):
        policy_obj = create_mixed_precision_policy(policy)
    else:
        policy_obj = policy
    
    return create_mixed_precision_train_step(train_step_fn, policy_obj)

