"""Tests for training module."""

from __future__ import annotations

import tempfile
import os
import json

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import optax
from flax import linen as nn

from beagle.training import (
    TrainState,
    TrainingConfig,
    Metrics,
    create_metrics,
    average_metrics,
    combine_metrics,
    format_metrics,
    accumulate_history,
    save_checkpoint,
    load_checkpoint,
    close_checkpointer,
    save_config,
    save_metrics_history,
    train_epoch,
)


# ============================================================================
# Test Types
# ============================================================================

def test_training_config_immutable():
    """TrainingConfig should be immutable."""
    config = TrainingConfig(
        num_epochs=10,
        batch_size=32,
        learning_rate=0.001
    )
    
    with pytest.raises(AttributeError):
        config.num_epochs = 20


def test_metrics_immutable():
    """Metrics should be immutable."""
    metrics = Metrics(values={"loss": 1.0, "acc": 0.9})
    
    with pytest.raises(AttributeError):
        metrics.values = {"new": 1.0}


def test_create_metrics_converts_jax_arrays():
    """create_metrics should convert JAX arrays to scalars."""
    raw = {
        "loss": jnp.array(1.5),
        "acc": jnp.array([0.9]),
        "scalar": 0.8
    }
    
    metrics = create_metrics(raw)
    
    assert metrics["loss"] == pytest.approx(1.5)
    assert metrics["acc"] == pytest.approx(0.9)
    assert metrics["scalar"] == pytest.approx(0.8)
    assert all(isinstance(v, float) for v in metrics.values.values())


# ============================================================================
# Test Metrics Functions
# ============================================================================

def test_average_metrics():
    """average_metrics should compute mean across metrics."""
    metrics_list = [
        Metrics(values={"loss": 1.0, "acc": 0.8}),
        Metrics(values={"loss": 2.0, "acc": 0.9}),
        Metrics(values={"loss": 3.0, "acc": 1.0})
    ]
    
    avg = average_metrics(metrics_list)
    
    assert avg["loss"] == pytest.approx(2.0)
    assert avg["acc"] == pytest.approx(0.9)


def test_average_metrics_empty():
    """average_metrics should handle empty list."""
    avg = average_metrics([])
    assert avg.values == {}


def test_combine_metrics_with_max():
    """combine_metrics should support custom aggregation."""
    metrics_list = [
        Metrics(values={"loss": 1.0}),
        Metrics(values={"loss": 3.0}),
        Metrics(values={"loss": 2.0})
    ]
    
    combined = combine_metrics(metrics_list, combine_fn=max)
    assert combined["loss"] == pytest.approx(3.0)


def test_format_metrics():
    """format_metrics should create readable string."""
    metrics = Metrics(values={"loss": 1.2345, "acc": 0.9876})
    formatted = format_metrics(metrics, precision=2)
    
    # Check both possible orderings (dict iteration order)
    assert "loss: 1.23" in formatted
    assert "acc: 0.99" in formatted
    assert "|" in formatted


def test_accumulate_history():
    """accumulate_history should append metrics to history."""
    history = {"loss": [1.0, 2.0], "acc": [0.8, 0.85]}
    metrics = Metrics(values={"loss": 3.0, "acc": 0.9})
    
    new_history = accumulate_history(history, metrics)
    
    # Check immutability - original unchanged
    assert len(history["loss"]) == 2
    
    # Check new history
    assert new_history["loss"] == [1.0, 2.0, 3.0]
    assert new_history["acc"] == [0.8, 0.85, 0.9]


def test_accumulate_history_new_metric():
    """accumulate_history should handle new metric keys."""
    history = {"loss": [1.0]}
    metrics = Metrics(values={"loss": 2.0, "new_metric": 0.5})
    
    new_history = accumulate_history(history, metrics)
    
    assert new_history["loss"] == [1.0, 2.0]
    assert new_history["new_metric"] == [0.5]


# ============================================================================
# Test Checkpoint Functions
# ============================================================================

def test_save_and_load_checkpoint():
    """save_checkpoint and load_checkpoint should roundtrip."""
    # Create simple model and state
    class SimpleModel(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(10)(x)
    
    model = SimpleModel()
    key = random.PRNGKey(0)
    x = jnp.ones((1, 5))
    variables = model.init(key, x)
    
    tx = optax.adam(0.001)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save checkpoint
        save_checkpoint(state, tmpdir, step=10)

        # Close checkpointer to finalize non-blocking save
        close_checkpointer()

        # Check directory exists
        ckpt_path = os.path.join(tmpdir, "checkpoint_10")
        assert os.path.exists(ckpt_path)

        # Load checkpoint
        restored = load_checkpoint(ckpt_path)
        
        # Check params match original
        param_tree_original = jax.tree_util.tree_leaves(state.params)
        param_tree_restored = jax.tree_util.tree_leaves(restored['params'])
        
        for orig, restored_param in zip(param_tree_original, param_tree_restored):
            assert jnp.allclose(orig, restored_param)
        
        # Check opt_state exists
        assert 'opt_state' in restored


def test_save_checkpoint_with_batch_stats():
    """save_checkpoint should handle batch_stats."""
    class BatchNormModel(nn.Module):
        @nn.compact
        def __call__(self, x, train: bool = True):
            x = nn.Dense(10)(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            return x
    
    model = BatchNormModel()
    key = random.PRNGKey(0)
    x = jnp.ones((2, 5))
    variables = model.init(key, x, train=True)
    
    tx = optax.adam(0.001)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        batch_stats=variables['batch_stats'],
        tx=tx
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(state, tmpdir)

        # Close checkpointer to finalize non-blocking save
        close_checkpointer()

        # Check final checkpoint exists
        ckpt_path = os.path.join(tmpdir, "checkpoint_final")
        assert os.path.exists(ckpt_path)

        # Load and verify batch_stats
        restored = load_checkpoint(ckpt_path)
        assert 'batch_stats' in restored
        assert restored['batch_stats'] is not None
        
        # Verify batch_stats values match original
        batch_stats_original = jax.tree_util.tree_leaves(state.batch_stats)
        batch_stats_restored = jax.tree_util.tree_leaves(restored['batch_stats'])
        
        for orig, restored_stat in zip(batch_stats_original, batch_stats_restored):
            assert jnp.allclose(orig, restored_stat)


def test_save_config():
    """save_config should write JSON file."""
    config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "model": "test"
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_config(config, tmpdir)
        
        config_path = os.path.join(tmpdir, "config.json")
        assert os.path.exists(config_path)
        
        with open(config_path) as f:
            loaded = json.load(f)
        
        assert loaded == config


def test_save_metrics_history():
    """save_metrics_history should write JSON file."""
    history = {
        "loss": [1.0, 0.8, 0.6],
        "acc": [0.7, 0.8, 0.9]
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_metrics_history(history, tmpdir)
        
        metrics_path = os.path.join(tmpdir, "metrics.json")
        assert os.path.exists(metrics_path)
        
        with open(metrics_path) as f:
            loaded = json.load(f)
        
        assert loaded == history


# ============================================================================
# Test Training Loop
# ============================================================================

def test_train_epoch():
    """train_epoch should train for one epoch."""
    # Simple model
    class SimpleModel(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(1)(x)
    
    model = SimpleModel()
    key = random.PRNGKey(0)
    x = jnp.ones((1, 5))
    variables = model.init(key, x)
    
    tx = optax.sgd(0.01)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )
    
    # Simple train step
    @jax.jit
    def train_step(state, batch, rng_key):
        def loss_fn(params):
            preds = state.apply_fn({'params': params}, batch['x'])
            loss = jnp.mean((preds - batch['y']) ** 2)
            return loss
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        
        return state, {"loss": loss}
    
    # Create dummy data iterator
    def data_iter():
        for _ in range(5):
            yield {
                'x': random.normal(random.PRNGKey(0), (4, 5)),
                'y': random.normal(random.PRNGKey(1), (4, 1))
            }
    
    # Train one epoch
    key = random.PRNGKey(42)
    new_state, metrics, new_key = train_epoch(
        state=state,
        train_step_fn=train_step,
        data_iterator=data_iter(),
        num_batches=5,
        rng_key=key
    )
    
    # Check state updated
    assert new_state is not state
    assert new_state.step == 5  # Should have taken 5 steps
    
    # Check metrics
    assert "loss" in metrics.values
    assert isinstance(metrics["loss"], float)


def test_train_epoch_with_logging(capsys):
    """train_epoch should log batch metrics when log_every is set."""
    # Minimal setup
    class Model(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(1)(x)
    
    model = Model()
    variables = model.init(random.PRNGKey(0), jnp.ones((1, 5)))
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optax.sgd(0.01)
    )
    
    @jax.jit
    def train_step(state, batch, rng_key):
        return state, {"loss": 1.0}
    
    def data_iter():
        for _ in range(3):
            yield {'x': jnp.ones((1, 5))}
    
    train_epoch(
        state=state,
        train_step_fn=train_step,
        data_iterator=data_iter(),
        num_batches=3,
        rng_key=random.PRNGKey(0),
        log_every=2
    )
    
    captured = capsys.readouterr()
    assert "Batch 2/3" in captured.out

