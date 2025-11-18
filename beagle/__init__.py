"""Beagle: Functional ML utilities for JAX/Flax.

Public API exports for dataset handling, network architectures, training,
and visualization.
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
    load_params,
    create_inference_fn,
)

# Profiling utilities
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
    "create_inference_fn",
    # Profiling
    "StepProfile",
    "ProfileSummary",
    "time_jax_function",
    "profile_iterator",
    "create_step_profiler",
    "format_step_metrics",
    "compute_epoch_summary",
    "format_epoch_summary",
]

