"""Beagle: Functional ML utilities for JAX/Flax.

Public API exports for dataset handling, network architectures, and training.
"""

# Training utilities
from beagle.training import (
    TrainState,
    TrainingConfig,
    Metrics,
    train_loop,
    train_epoch,
    save_checkpoint,
    load_checkpoint,
)

__version__ = "0.1.0"

__all__ = [
    # Training
    "TrainState",
    "TrainingConfig",
    "Metrics",
    "train_loop",
    "train_epoch",
    "save_checkpoint",
    "load_checkpoint",
]

