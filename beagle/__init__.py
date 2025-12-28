"""Beagle: Functional ML utilities for JAX/Flax.

Public API exports for dataset handling, network architectures, training,
and visualization.
"""

# Graph API (new composable architecture)
from beagle.graph import (
    ComputeNode,
    ComputeGraph,
    GraphState,
    create_train_step,
    create_eval_step,
    train_epoch as graph_train_epoch,
    evaluate,
    simple_training_loop,
)

# Checkpoint utilities for graphs
from beagle.checkpoint import (
    save_graph_state,
    save_node,
    load_graph_state,
    load_node,
)

# Training utilities (legacy)
from beagle.training import (
    TrainState,
    TrainingConfig,
    Metrics,
    train_loop,
    train_epoch,
    save_checkpoint,
    load_checkpoint,
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

# Dataset preprocessing utilities
from beagle.dataset import (
    compute_fields_mean_std,
    compute_fields_min_max,
    save_field_stats,
    load_field_stats,
)

__version__ = "0.1.0"

__all__ = [
    # Graph API (new)
    "ComputeNode",
    "ComputeGraph",
    "GraphState",
    "create_train_step",
    "create_eval_step",
    "graph_train_epoch",
    "evaluate",
    "simple_training_loop",
    "save_graph_state",
    "save_node",
    "load_graph_state",
    "load_node",

    # Training (legacy)
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

    # Stats
    "compute_fields_mean_std",
    "compute_fields_min_max",
    "save_field_stats",
    "load_field_stats",
]

