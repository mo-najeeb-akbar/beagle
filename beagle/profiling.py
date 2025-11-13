"""Performance profiling utilities for training and data loading.

Functional profiling tools to measure and track performance bottlenecks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar
from collections.abc import Iterator

import jax


T = TypeVar("T")


@dataclass(frozen=True)
class StepProfile:
    """Immutable profiling data for a single step."""
    
    step_num: int
    total_time: float
    data_time: float
    compute_time: float
    
    @property
    def throughput(self) -> float:
        """Samples per second (assumes batch size of 1 if not provided)."""
        return 1.0 / self.total_time if self.total_time > 0 else 0.0


@dataclass(frozen=True)
class ProfileSummary:
    """Statistical summary of profiling data."""
    
    name: str
    count: int
    mean_time: float
    min_time: float
    max_time: float
    total_time: float
    
    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Count:     {self.count}\n"
            f"  Mean:      {self.mean_time*1000:.2f} ms\n"
            f"  Min:       {self.min_time*1000:.2f} ms\n"
            f"  Max:       {self.max_time*1000:.2f} ms\n"
            f"  Total:     {self.total_time:.2f} s"
        )


def time_jax_function(fn: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, float]:
    """Time a JAX function including device synchronization.
    
    Pure function that executes and times a JAX computation.
    
    Args:
        fn: JAX function to time
        *args: Positional arguments to fn
        **kwargs: Keyword arguments to fn
        
    Returns:
        Tuple of (result, elapsed_time_seconds)
    """
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    
    # Force synchronization for accurate timing
    if hasattr(result, 'block_until_ready'):
        result.block_until_ready()
    elif isinstance(result, (tuple, list)):
        for item in result:
            if hasattr(item, 'block_until_ready'):
                item.block_until_ready()
    elif isinstance(result, dict):
        for item in result.values():
            if hasattr(item, 'block_until_ready'):
                item.block_until_ready()
    
    elapsed = time.perf_counter() - start
    return result, elapsed


def profile_iterator(
    iterator: Iterator[T],
    name: str = "data_loading"
) -> Iterator[tuple[T, float]]:
    """Wrap an iterator to profile time per item.
    
    Pure generator that yields items with timing information.
    
    Args:
        iterator: Source iterator to profile
        name: Name for logging
        
    Yields:
        Tuple of (item, load_time_seconds)
    """
    while True:
        try:
            start = time.perf_counter()
            item = next(iterator)
            elapsed = time.perf_counter() - start
            yield item, elapsed
        except StopIteration:
            break


def create_step_profiler(batch_size: int = 1) -> tuple[
    Callable[[int, Callable, tuple, dict], tuple[Any, StepProfile]],
    Callable[[list[StepProfile]], ProfileSummary]
]:
    """Create profiling functions for training steps.
    
    Returns tuple of (profile_step_fn, summarize_fn).
    
    Args:
        batch_size: Batch size for throughput calculation
        
    Returns:
        Tuple of profiling function and summary function
    """
    def profile_step(
        step_num: int,
        step_fn: Callable,
        args: tuple,
        kwargs: dict,
        data_time: float = 0.0
    ) -> tuple[Any, StepProfile]:
        """Profile a single training step.
        
        Args:
            step_num: Current step number
            step_fn: Training step function
            args: Arguments to step_fn
            kwargs: Keyword arguments to step_fn
            data_time: Time spent loading data (optional)
            
        Returns:
            Tuple of (step_result, profile_data)
        """
        start = time.perf_counter()
        result, compute_time = time_jax_function(step_fn, *args, **kwargs)
        total_time = time.perf_counter() - start
        
        profile = StepProfile(
            step_num=step_num,
            total_time=total_time,
            data_time=data_time,
            compute_time=compute_time
        )
        
        return result, profile
    
    def summarize(profiles: list[StepProfile]) -> ProfileSummary:
        """Create summary statistics from profile list.
        
        Args:
            profiles: List of step profiles
            
        Returns:
            Summary statistics
        """
        if not profiles:
            return ProfileSummary("empty", 0, 0.0, 0.0, 0.0, 0.0)
        
        times = [p.compute_time for p in profiles]
        return ProfileSummary(
            name="training_step",
            count=len(profiles),
            mean_time=sum(times) / len(times),
            min_time=min(times),
            max_time=max(times),
            total_time=sum(times)
        )
    
    return profile_step, summarize


def format_step_metrics(
    step_num: int,
    profile: StepProfile,
    metrics: dict[str, float],
    window_size: int = 10
) -> str:
    """Format step metrics for logging.
    
    Pure function to create formatted string.
    
    Args:
        step_num: Current step number
        profile: Profile data
        metrics: Training metrics (loss, etc.)
        window_size: Number of steps for moving average
        
    Returns:
        Formatted string
    """
    metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    
    return (
        f"Step {step_num:04d} | "
        f"{metric_str} | "
        f"data: {profile.data_time*1000:.1f}ms | "
        f"compute: {profile.compute_time*1000:.1f}ms | "
        f"total: {profile.total_time*1000:.1f}ms"
    )


def compute_epoch_summary(profiles: list[StepProfile]) -> dict[str, float]:
    """Compute summary statistics for an epoch.
    
    Pure function to aggregate profiling data.
    
    Args:
        profiles: List of step profiles from epoch
        
    Returns:
        Dictionary of summary statistics
    """
    if not profiles:
        return {}
    
    data_times = [p.data_time for p in profiles]
    compute_times = [p.compute_time for p in profiles]
    total_times = [p.total_time for p in profiles]
    
    return {
        "steps": len(profiles),
        "mean_data_time": sum(data_times) / len(data_times),
        "mean_compute_time": sum(compute_times) / len(compute_times),
        "mean_total_time": sum(total_times) / len(total_times),
        "total_time": sum(total_times),
        "data_percent": 100.0 * sum(data_times) / sum(total_times) if sum(total_times) > 0 else 0.0,
        "compute_percent": 100.0 * sum(compute_times) / sum(total_times) if sum(total_times) > 0 else 0.0,
    }


def format_epoch_summary(epoch: int, summary: dict[str, float]) -> str:
    """Format epoch summary for logging.
    
    Args:
        epoch: Epoch number
        summary: Summary statistics from compute_epoch_summary
        
    Returns:
        Formatted string
    """
    return (
        f"\n{'='*70}\n"
        f"Epoch {epoch} Summary:\n"
        f"  Steps:              {summary['steps']}\n"
        f"  Mean data time:     {summary['mean_data_time']*1000:.1f} ms ({summary['data_percent']:.1f}%)\n"
        f"  Mean compute time:  {summary['mean_compute_time']*1000:.1f} ms ({summary['compute_percent']:.1f}%)\n"
        f"  Mean total time:    {summary['mean_total_time']*1000:.1f} ms\n"
        f"  Total epoch time:   {summary['total_time']:.2f} s\n"
        f"{'='*70}\n"
    )

