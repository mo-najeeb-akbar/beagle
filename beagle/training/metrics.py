"""Metrics tracking and aggregation (pure functions)."""

from __future__ import annotations

from typing import Callable
from functools import reduce

from beagle.training.types import Metrics


def average_metrics(metrics_list: list[Metrics]) -> Metrics:
    """Average metrics across multiple batches/epochs (pure function).
    
    Args:
        metrics_list: List of Metrics objects to average
        
    Returns:
        New Metrics object with averaged values
    """
    if not metrics_list:
        return Metrics(values={})
    
    # Get all keys from first metrics
    keys = metrics_list[0].keys()
    
    # Average each metric
    averaged = {}
    for key in keys:
        values = [m[key] for m in metrics_list]
        averaged[key] = sum(values) / len(values)
    
    return Metrics(values=averaged)


def combine_metrics(
    metrics_list: list[Metrics],
    combine_fn: Callable[[list[float]], float] = lambda x: sum(x) / len(x)
) -> Metrics:
    """Combine metrics using custom aggregation function (pure function).
    
    Args:
        metrics_list: List of Metrics to combine
        combine_fn: Function to aggregate values (default: mean)
        
    Returns:
        New Metrics with combined values
    """
    if not metrics_list:
        return Metrics(values={})
    
    keys = metrics_list[0].keys()
    combined = {}
    
    for key in keys:
        values = [m[key] for m in metrics_list]
        combined[key] = combine_fn(values)
    
    return Metrics(values=combined)


def format_metrics(metrics: Metrics, precision: int = 4) -> str:
    """Format metrics as string for logging (pure function)."""
    parts = [f"{k}: {v:.{precision}f}" for k, v in metrics.values.items()]
    return " | ".join(parts)


def accumulate_history(
    history: dict[str, list[float]],
    metrics: Metrics
) -> dict[str, list[float]]:
    """Accumulate metrics into history dict (pure function - returns new dict).
    
    Args:
        history: Existing history dictionary
        metrics: New metrics to add
        
    Returns:
        New history dict with metrics appended
    """
    new_history = {k: v.copy() for k, v in history.items()}
    
    for key in metrics.keys():
        if key not in new_history:
            new_history[key] = []
        new_history[key].append(metrics[key])
    
    return new_history

