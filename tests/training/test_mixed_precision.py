"""Tests for mixed precision training utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import flax.linen as nn

from beagle.training import (
    TrainState,
    MixedPrecisionPolicy,
    create_mixed_precision_policy,
    enable_mixed_precision,
    get_recommended_policy,
)
from beagle.training.mixed_precision import (
    cast_inputs_to_compute,
    cast_outputs_to_float32,
    wrap_forward_with_mixed_precision,
)


def test_create_mixed_precision_policy_bfloat16():
    """Test creating bfloat16 policy."""
    policy = create_mixed_precision_policy("bfloat16")
    
    assert policy.compute_dtype == jnp.bfloat16
    assert policy.param_dtype == jnp.float32
    assert policy.output_dtype == jnp.float32


def test_create_mixed_precision_policy_float16():
    """Test creating float16 policy."""
    policy = create_mixed_precision_policy("float16")
    
    assert policy.compute_dtype == jnp.float16
    assert policy.param_dtype == jnp.float32
    assert policy.output_dtype == jnp.float32


def test_create_mixed_precision_policy_float32():
    """Test creating float32 policy (no mixed precision)."""
    policy = create_mixed_precision_policy("float32")
    
    assert policy.compute_dtype == jnp.float32
    assert policy.param_dtype == jnp.float32
    assert policy.output_dtype == jnp.float32


def test_create_mixed_precision_policy_invalid():
    """Test invalid dtype raises error."""
    try:
        create_mixed_precision_policy("invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "compute_dtype must be one of" in str(e)


def test_cast_inputs_to_compute():
    """Test casting inputs to compute dtype."""
    policy = create_mixed_precision_policy("bfloat16")
    
    # Single array
    x_f32 = jnp.ones((2, 3), dtype=jnp.float32)
    x_bf16 = cast_inputs_to_compute(x_f32, policy)
    
    assert x_bf16.dtype == jnp.bfloat16
    assert x_bf16.shape == x_f32.shape
    
    # Nested structure
    inputs = {
        'x': jnp.ones((2, 3), dtype=jnp.float32),
        'y': jnp.zeros((2, 3), dtype=jnp.float32),
    }
    outputs = cast_inputs_to_compute(inputs, policy)
    
    assert outputs['x'].dtype == jnp.bfloat16
    assert outputs['y'].dtype == jnp.bfloat16


def test_cast_inputs_preserves_non_float():
    """Test that non-float types are preserved."""
    policy = create_mixed_precision_policy("bfloat16")
    
    inputs = {
        'x': jnp.ones((2, 3), dtype=jnp.float32),
        'idx': jnp.array([0, 1, 2], dtype=jnp.int32),
    }
    outputs = cast_inputs_to_compute(inputs, policy)
    
    assert outputs['x'].dtype == jnp.bfloat16
    assert outputs['idx'].dtype == jnp.int32  # Preserved


def test_cast_outputs_to_float32():
    """Test casting outputs to float32."""
    # Single array
    x_bf16 = jnp.ones((2, 3), dtype=jnp.bfloat16)
    x_f32 = cast_outputs_to_float32(x_bf16)
    
    assert x_f32.dtype == jnp.float32
    assert x_f32.shape == x_bf16.shape
    
    # Nested structure
    outputs_bf16 = {
        'pred': jnp.ones((2, 3), dtype=jnp.bfloat16),
        'loss': jnp.array(0.5, dtype=jnp.bfloat16),
    }
    outputs_f32 = cast_outputs_to_float32(outputs_bf16)
    
    assert outputs_f32['pred'].dtype == jnp.float32
    assert outputs_f32['loss'].dtype == jnp.float32


def test_wrap_forward_with_mixed_precision():
    """Test wrapping forward function with mixed precision."""
    policy = create_mixed_precision_policy("bfloat16")
    
    def forward(x):
        # This would compute in bfloat16
        return x * 2.0
    
    wrapped = wrap_forward_with_mixed_precision(forward, policy)
    
    # Input float32
    x_f32 = jnp.ones((2, 3), dtype=jnp.float32)
    y = wrapped(x_f32)
    
    # Output should be float32
    assert y.dtype == jnp.float32
    assert jnp.allclose(y, 2.0)


def test_enable_mixed_precision_with_simple_train_step():
    """Test mixed precision with simple training step."""
    # Simple linear model
    class SimpleModel(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(1)(x)
    
    # Create model and state
    model = SimpleModel()
    key = random.key(0)
    x_dummy = jnp.ones((1, 10))
    variables = model.init(key, x_dummy)
    
    tx = optax.sgd(0.01)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )
    
    # Verify params are float32
    param_dtypes = jax.tree.map(lambda x: x.dtype, state.params)
    assert all(d == jnp.float32 for d in jax.tree.leaves(param_dtypes))
    
    # Define training step
    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            pred = state.apply_fn({'params': params}, batch['x'])
            return jnp.mean((pred - batch['y'])**2)
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, {'loss': loss}
    
    # Enable mixed precision
    mp_train_step = enable_mixed_precision(train_step, policy="bfloat16")
    
    # Create batch
    key, batch_key = random.split(key)
    batch = {
        'x': random.normal(batch_key, (4, 10)),
        'y': random.normal(batch_key, (4, 1)),
    }
    
    # Run training step
    state, metrics = mp_train_step(state, batch)
    
    # Verify loss is float32
    assert metrics['loss'].dtype == jnp.float32
    
    # Verify params are still float32
    param_dtypes = jax.tree.map(lambda x: x.dtype, state.params)
    assert all(d == jnp.float32 for d in jax.tree.leaves(param_dtypes))


def test_enable_mixed_precision_auto_policy():
    """Test auto policy selection."""
    # Simple model
    class SimpleModel(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(1)(x)
    
    model = SimpleModel()
    key = random.key(0)
    x_dummy = jnp.ones((1, 3))
    variables = model.init(key, x_dummy)
    
    tx = optax.sgd(0.01)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )
    
    # Simple training step
    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            pred = state.apply_fn({'params': params}, batch['x'])
            return jnp.mean(pred**2)
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, {'loss': loss}
    
    # Enable with auto policy
    mp_train_step = enable_mixed_precision(train_step, policy="auto")
    
    batch = {'x': jnp.ones((2, 3))}
    result_state, metrics = mp_train_step(state, batch)
    
    assert metrics['loss'].dtype == jnp.float32


def test_get_recommended_policy():
    """Test getting recommended policy."""
    policy = get_recommended_policy()
    
    # Should return a valid policy
    assert isinstance(policy, MixedPrecisionPolicy)
    assert policy.param_dtype == jnp.float32
    assert policy.output_dtype == jnp.float32
    
    # Compute dtype depends on hardware
    assert policy.compute_dtype in [jnp.float32, jnp.bfloat16, jnp.float16]


def test_mixed_precision_preserves_gradients():
    """Test that mixed precision preserves gradient flow."""
    # Simple model
    class SimpleModel(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(1)(x)
    
    model = SimpleModel()
    key = random.key(42)
    x_dummy = jnp.ones((1, 5))
    variables = model.init(key, x_dummy)
    
    # Loss function
    def loss_fn(params, x, y):
        pred = model.apply({'params': params}, x)
        return jnp.mean((pred - y)**2)
    
    # Compute gradients without mixed precision
    x = jnp.ones((2, 5))
    y = jnp.ones((2, 1))
    grads_f32 = jax.grad(loss_fn)(variables['params'], x, y)
    
    # Compute gradients with mixed precision
    policy = create_mixed_precision_policy("bfloat16")
    
    def loss_fn_mp(params, x, y):
        # Cast inputs
        x_bf16 = x.astype(jnp.bfloat16)
        # Compute
        pred_bf16 = model.apply({'params': params}, x_bf16)
        # Cast output
        pred_f32 = pred_bf16.astype(jnp.float32)
        return jnp.mean((pred_f32 - y)**2)
    
    grads_mp = jax.grad(loss_fn_mp)(variables['params'], x, y)
    
    # Gradients should be close (within bfloat16 precision)
    for g_f32, g_mp in zip(
        jax.tree.leaves(grads_f32), jax.tree.leaves(grads_mp)
    ):
        # Allow 1% relative error due to bfloat16 precision
        assert jnp.allclose(g_f32, g_mp, rtol=0.01)


def test_mixed_precision_convergence():
    """Test that mixed precision trains a simple model successfully."""
    # Simple linear regression: y = 2x + 1
    class LinearModel(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(1, use_bias=True)(x)
    
    model = LinearModel()
    key = random.key(123)
    x_dummy = jnp.ones((1, 1))
    variables = model.init(key, x_dummy)
    
    tx = optax.sgd(0.1)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )
    
    # Define training step
    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            pred = state.apply_fn({'params': params}, batch['x'])
            return jnp.mean((pred - batch['y'])**2)
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, {'loss': loss}
    
    # Enable mixed precision
    mp_train_step = enable_mixed_precision(train_step, policy="bfloat16")
    
    # Generate data: y = 2x + 1
    key, data_key = random.split(key)
    x = random.uniform(data_key, (100, 1))
    y = 2.0 * x + 1.0
    
    # Train for a few steps
    initial_loss = None
    final_loss = None
    
    for i in range(20):
        batch = {'x': x, 'y': y}
        state, metrics = mp_train_step(state, batch)
        
        if i == 0:
            initial_loss = float(metrics['loss'])
        if i == 19:
            final_loss = float(metrics['loss'])
    
    # Loss should decrease significantly
    assert final_loss < initial_loss
    
    # Should converge reasonably (bfloat16 precision may prevent perfect convergence)
    assert final_loss < 0.15  # Within bfloat16 tolerance

