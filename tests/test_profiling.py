"""Tests for profiling utilities."""

from __future__ import annotations

import time
from typing import Iterator

import jax
import jax.numpy as jnp
import pytest

from beagle.profiling import (
    StepProfile,
    ProfileSummary,
    time_jax_function,
    profile_iterator,
    create_step_profiler,
    format_step_metrics,
    compute_epoch_summary,
    format_epoch_summary,
)


# Module-level functions for picklability


def simple_jax_add(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Simple JAX function for testing."""
    return x + y


def slow_jax_matmul(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Slightly slower JAX operation for timing tests."""
    result = jnp.matmul(x, y)
    # Multiple operations to ensure measurable time
    for _ in range(10):
        result = jnp.matmul(result, jnp.eye(result.shape[0]))
    return result


class TestStepProfile:
    """Test StepProfile dataclass."""
    
    def test_creation(self):
        profile = StepProfile(
            step_num=5,
            total_time=0.1,
            data_time=0.02,
            compute_time=0.08
        )
        assert profile.step_num == 5
        assert profile.total_time == 0.1
        assert profile.data_time == 0.02
        assert profile.compute_time == 0.08
    
    def test_immutability(self):
        profile = StepProfile(
            step_num=1,
            total_time=0.1,
            data_time=0.02,
            compute_time=0.08
        )
        with pytest.raises(Exception):  # dataclass frozen
            profile.step_num = 2
    
    def test_throughput_property(self):
        profile = StepProfile(
            step_num=0,
            total_time=0.1,
            data_time=0.02,
            compute_time=0.08
        )
        assert abs(profile.throughput - 10.0) < 0.01
    
    def test_throughput_zero_time(self):
        profile = StepProfile(
            step_num=0,
            total_time=0.0,
            data_time=0.0,
            compute_time=0.0
        )
        assert profile.throughput == 0.0


class TestProfileSummary:
    """Test ProfileSummary dataclass."""
    
    def test_creation(self):
        summary = ProfileSummary(
            name="test",
            count=10,
            mean_time=0.05,
            min_time=0.03,
            max_time=0.08,
            total_time=0.5
        )
        assert summary.name == "test"
        assert summary.count == 10
        assert summary.mean_time == 0.05
    
    def test_string_formatting(self):
        summary = ProfileSummary(
            name="test_op",
            count=5,
            mean_time=0.1,
            min_time=0.08,
            max_time=0.12,
            total_time=0.5
        )
        result = str(summary)
        assert "test_op" in result
        assert "Count:" in result
        assert "Mean:" in result
        assert "100.00 ms" in result  # 0.1 * 1000


class TestTimeJaxFunction:
    """Test JAX function timing."""
    
    def test_basic_timing(self):
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])
        
        result, elapsed = time_jax_function(simple_jax_add, x, y)
        
        assert jnp.allclose(result, jnp.array([5.0, 7.0, 9.0]))
        assert elapsed >= 0.0
        assert elapsed < 1.0  # Should be very fast
    
    def test_timing_with_kwargs(self):
        x = jnp.ones((10, 10))
        y = jnp.ones((10, 10))
        
        def add_with_scale(a, b, scale=1.0):
            return (a + b) * scale
        
        result, elapsed = time_jax_function(add_with_scale, x, y, scale=2.0)
        
        assert jnp.allclose(result, jnp.ones((10, 10)) * 4.0)
        assert elapsed >= 0.0
    
    def test_timing_with_device_sync(self):
        """Ensure device synchronization happens."""
        x = jnp.ones((100, 100))
        y = jnp.ones((100, 100))
        
        result, elapsed = time_jax_function(slow_jax_matmul, x, y)
        
        # Result should be computed
        assert result.shape == (100, 100)
        # Time should be measurable (not just dispatch time)
        assert elapsed > 0.0
    
    def test_timing_tuple_result(self):
        """Test timing with tuple result (common in training steps)."""
        x = jnp.array([1.0, 2.0])
        
        def multi_return(a):
            return a * 2, a * 3
        
        result, elapsed = time_jax_function(multi_return, x)
        
        assert len(result) == 2
        assert jnp.allclose(result[0], jnp.array([2.0, 4.0]))
        assert jnp.allclose(result[1], jnp.array([3.0, 6.0]))
        assert elapsed >= 0.0
    
    def test_timing_dict_result(self):
        """Test timing with dict result."""
        x = jnp.array([1.0, 2.0])
        
        def dict_return(a):
            return {'double': a * 2, 'triple': a * 3}
        
        result, elapsed = time_jax_function(dict_return, x)
        
        assert 'double' in result
        assert 'triple' in result
        assert jnp.allclose(result['double'], jnp.array([2.0, 4.0]))
        assert elapsed >= 0.0


class TestProfileIterator:
    """Test iterator profiling."""
    
    def test_basic_profiling(self):
        data = [1, 2, 3, 4, 5]
        profiled = profile_iterator(iter(data))
        
        results = list(profiled)
        
        assert len(results) == 5
        for i, (item, load_time) in enumerate(results):
            assert item == i + 1
            assert load_time >= 0.0
    
    def test_empty_iterator(self):
        profiled = profile_iterator(iter([]))
        results = list(profiled)
        assert results == []
    
    def test_custom_name(self):
        """Name parameter doesn't affect output, just for logging."""
        data = [1, 2, 3]
        profiled = profile_iterator(iter(data), name="custom_loader")
        
        results = list(profiled)
        assert len(results) == 3
    
    def test_timing_slow_iterator(self):
        """Test that timing captures iterator delays."""
        def slow_iterator():
            for i in range(3):
                time.sleep(0.01)  # Simulate slow data loading
                yield i
        
        profiled = profile_iterator(slow_iterator())
        results = list(profiled)
        
        # Each load should take at least 0.01 seconds
        for item, load_time in results:
            assert load_time >= 0.009  # Allow small margin


class TestCreateStepProfiler:
    """Test step profiler creation."""
    
    def test_basic_profiling(self):
        profile_step, _ = create_step_profiler(batch_size=32)
        
        def dummy_step(x):
            return x * 2
        
        result, profile = profile_step(
            step_num=0,
            step_fn=dummy_step,
            args=(5,),
            kwargs={}
        )
        
        assert result == 10
        assert profile.step_num == 0
        assert profile.total_time >= 0.0
        assert profile.compute_time >= 0.0
    
    def test_with_data_time(self):
        profile_step, _ = create_step_profiler(batch_size=16)
        
        def dummy_step(x):
            return x + 1
        
        result, profile = profile_step(
            step_num=5,
            step_fn=dummy_step,
            args=(10,),
            kwargs={},
            data_time=0.05
        )
        
        assert result == 11
        assert profile.step_num == 5
        assert profile.data_time == 0.05
    
    def test_summarize_empty(self):
        _, summarize = create_step_profiler()
        
        summary = summarize([])
        
        assert summary.name == "empty"
        assert summary.count == 0
    
    def test_summarize_multiple_profiles(self):
        _, summarize = create_step_profiler()
        
        profiles = [
            StepProfile(0, 0.1, 0.02, 0.08),
            StepProfile(1, 0.12, 0.03, 0.09),
            StepProfile(2, 0.11, 0.02, 0.09),
        ]
        
        summary = summarize(profiles)
        
        assert summary.name == "training_step"
        assert summary.count == 3
        assert abs(summary.mean_time - 0.0867) < 0.001
        assert summary.min_time == 0.08
        assert summary.max_time == 0.09


class TestFormatStepMetrics:
    """Test step metrics formatting."""
    
    def test_basic_formatting(self):
        profile = StepProfile(
            step_num=10,
            total_time=0.1,
            data_time=0.02,
            compute_time=0.08
        )
        metrics = {'loss': 1.2345, 'accuracy': 0.8765}
        
        result = format_step_metrics(10, profile, metrics)
        
        assert "Step 0010" in result
        assert "loss: 1.2345" in result
        assert "accuracy: 0.8765" in result
        assert "data: 20.0ms" in result
        assert "compute: 80.0ms" in result
        assert "total: 100.0ms" in result
    
    def test_empty_metrics(self):
        profile = StepProfile(0, 0.05, 0.01, 0.04)
        metrics = {}
        
        result = format_step_metrics(0, profile, metrics)
        
        assert "Step 0000" in result
        assert "data:" in result


class TestComputeEpochSummary:
    """Test epoch summary computation."""
    
    def test_empty_profiles(self):
        summary = compute_epoch_summary([])
        assert summary == {}
    
    def test_single_profile(self):
        profiles = [StepProfile(0, 0.1, 0.02, 0.08)]
        
        summary = compute_epoch_summary(profiles)
        
        assert summary['steps'] == 1
        assert summary['mean_data_time'] == 0.02
        assert summary['mean_compute_time'] == 0.08
        assert summary['mean_total_time'] == 0.1
        assert summary['total_time'] == 0.1
    
    def test_multiple_profiles(self):
        profiles = [
            StepProfile(0, 0.1, 0.02, 0.08),
            StepProfile(1, 0.12, 0.03, 0.09),
            StepProfile(2, 0.11, 0.025, 0.085),
        ]
        
        summary = compute_epoch_summary(profiles)
        
        assert summary['steps'] == 3
        assert abs(summary['mean_data_time'] - 0.025) < 0.001
        assert abs(summary['mean_compute_time'] - 0.085) < 0.001
        assert abs(summary['total_time'] - 0.33) < 0.001
    
    def test_percentage_computation(self):
        """Test that percentages sum to ~100%."""
        profiles = [
            StepProfile(0, 0.1, 0.03, 0.07),
            StepProfile(1, 0.1, 0.03, 0.07),
        ]
        
        summary = compute_epoch_summary(profiles)
        
        assert abs(summary['data_percent'] - 30.0) < 0.1
        assert abs(summary['compute_percent'] - 70.0) < 0.1


class TestFormatEpochSummary:
    """Test epoch summary formatting."""
    
    def test_basic_formatting(self):
        summary = {
            'steps': 100,
            'mean_data_time': 0.02,
            'mean_compute_time': 0.08,
            'mean_total_time': 0.1,
            'total_time': 10.0,
            'data_percent': 20.0,
            'compute_percent': 80.0,
        }
        
        result = format_epoch_summary(5, summary)
        
        assert "Epoch 5" in result
        assert "Steps:" in result
        assert "100" in result
        assert "20.0 ms" in result
        assert "80.0 ms" in result
        assert "10.00 s" in result
        assert "20.0%" in result
        assert "80.0%" in result


class TestEndToEnd:
    """End-to-end profiling integration tests."""
    
    def test_complete_training_step_profile(self):
        """Simulate a complete training step with profiling."""
        profile_step, summarize = create_step_profiler(batch_size=32)
        
        # Simulate training step
        @jax.jit
        def train_step(x, y):
            pred = x @ y
            loss = jnp.mean(pred ** 2)
            return pred, loss
        
        x = jnp.ones((32, 10))
        y = jnp.ones((10, 5))
        
        result, profile = profile_step(
            step_num=0,
            step_fn=train_step,
            args=(x, y),
            kwargs={},
            data_time=0.01
        )
        
        pred, loss = result
        assert pred.shape == (32, 5)
        assert profile.step_num == 0
        assert profile.data_time == 0.01
        assert profile.compute_time >= 0.0
    
    def test_profiled_epoch(self):
        """Simulate a full epoch with profiling."""
        profile_step, summarize = create_step_profiler(batch_size=8)
        
        def simple_step(state, batch):
            return state + batch, batch
        
        profiles = []
        # Don't pass data_time - let it be measured naturally
        for i in range(10):
            result, profile = profile_step(
                step_num=i,
                step_fn=simple_step,
                args=(i, i * 2),
                kwargs={}
            )
            profiles.append(profile)
        
        summary = compute_epoch_summary(profiles)
        
        assert summary['steps'] == 10
        assert summary['total_time'] >= 0.0
        # Just check the summary contains expected keys
        assert 'mean_data_time' in summary
        assert 'mean_compute_time' in summary
        assert 'data_percent' in summary
        assert 'compute_percent' in summary

