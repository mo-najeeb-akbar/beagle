"""Training state and configuration types (pure data structures)."""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass

import jax.numpy as jnp
from flax.training import train_state


class TrainState(train_state.TrainState):
    """Extended training state with optional batch statistics.
    
    Pure data container for model training state.
    """
    batch_stats: dict[str, Any] | None = None


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable training configuration.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        seed: Random seed for reproducibility
        checkpoint_dir: Directory to save checkpoints (None = no checkpoints)
        checkpoint_every: Save checkpoint every N epochs (None = only final)
        log_every: Log metrics every N batches (None = only epoch summary)
    """
    num_epochs: int
    batch_size: int
    learning_rate: float
    seed: int = 42
    checkpoint_dir: str | None = None
    checkpoint_every: int | None = None
    log_every: int | None = None


@dataclass(frozen=True)
class Metrics:
    """Immutable metrics container.
    
    Stores metrics as a dictionary for flexibility.
    All values should be Python scalars (not JAX arrays).
    """
    values: dict[str, float]
    
    def __getitem__(self, key: str) -> float:
        return self.values[key]
    
    def keys(self) -> list[str]:
        return list(self.values.keys())


def to_scalar(x: Any) -> float:
    """Convert JAX array or scalar to Python float (pure function)."""
    if isinstance(x, (int, float)):
        return float(x)
    return float(jnp.asarray(x).item())


def create_metrics(raw_metrics: dict[str, Any]) -> Metrics:
    """Convert raw metrics dict to immutable Metrics (pure function)."""
    scalar_metrics = {k: to_scalar(v) for k, v in raw_metrics.items()}
    return Metrics(values=scalar_metrics)

