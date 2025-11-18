"""Training utilities for JAX/Flax models.

Functional training library with:
- Immutable training state and configuration
- Pure metric functions
- Composable training loops
- Checkpoint management
- Inference utilities
"""

from beagle.training.types import (
    TrainState,
    TrainingConfig,
    Metrics,
    create_metrics,
)

from beagle.training.metrics import (
    average_metrics,
    combine_metrics,
    format_metrics,
    accumulate_history,
)

from beagle.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_params,
    save_config,
    save_metrics_history,
)

from beagle.training.loop import (
    train_epoch,
    train_loop,
    VizCallback,
)

from beagle.training.inference import (
    create_inference_fn,
    batch_inference,
)

from beagle.training.mixed_precision import (
    MixedPrecisionPolicy,
    create_mixed_precision_policy,
    enable_mixed_precision,
    get_recommended_policy,
)

__all__ = [
    # Types
    "TrainState",
    "TrainingConfig",
    "Metrics",
    "create_metrics",
    # Metrics
    "average_metrics",
    "combine_metrics",
    "format_metrics",
    "accumulate_history",
    # Checkpoints
    "save_checkpoint",
    "load_checkpoint",
    "load_params",
    "save_config",
    "save_metrics_history",
    # Training loops
    "train_epoch",
    "train_loop",
    "VizCallback",
    # Inference
    "create_inference_fn",
    "batch_inference",
    # Mixed precision
    "MixedPrecisionPolicy",
    "create_mixed_precision_policy",
    "enable_mixed_precision",
    "get_recommended_policy",
]

