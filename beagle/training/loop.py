"""Training loop implementation (orchestrates pure and impure code)."""

from __future__ import annotations

from typing import Callable, Iterator, Any
from functools import partial

import jax.random as random

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from beagle.training.types import TrainState, Metrics, create_metrics
from beagle.training.metrics import average_metrics, format_metrics, accumulate_history
from beagle.training.checkpoint import save_checkpoint


StepFn = Callable[[TrainState, Any, Any], tuple[TrainState, dict[str, Any]]]
DataIterator = Iterator[Any] | Callable[[], Iterator[Any]]
VizCallback = Callable[[TrainState, Any, Any, int], None]


def train_epoch(
    state: TrainState,
    train_step_fn: StepFn,
    data_iterator: Iterator[Any],
    num_batches: int,
    rng_key: Any,
    log_every: int | None = None
) -> tuple[TrainState, Metrics, Any]:
    """Train for one epoch (combines pure step_fn with iteration).
    
    Args:
        state: Current training state
        train_step_fn: JIT-compiled training step function
            Should have signature: (state, batch, rng) -> (new_state, metrics_dict)
        data_iterator: Iterator yielding batches
        num_batches: Number of batches per epoch
        rng_key: JAX random key
        log_every: Log metrics every N batches (None = no batch logging)
        
    Returns:
        Tuple of (new_state, epoch_metrics, new_rng_key)
    """
    batch_metrics = []
    
    for batch_idx in range(num_batches):
        # Split RNG key
        rng_key, step_key = random.split(rng_key)
        
        # Get batch and run step
        batch = next(data_iterator)
        state, raw_metrics = train_step_fn(state, batch, step_key)
        
        # Convert to scalar metrics
        metrics = create_metrics(raw_metrics)
        batch_metrics.append(metrics)
        
        # Optional batch-level logging
        if log_every is not None and (batch_idx + 1) % log_every == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches} | {format_metrics(metrics)}")
    
    # Average metrics across epoch
    epoch_metrics = average_metrics(batch_metrics)
    
    return state, epoch_metrics, rng_key


def train_loop(
    state: TrainState,
    train_step_fn: StepFn,
    data_iterator_fn: Callable[[], Iterator[Any]],
    num_epochs: int,
    num_batches: int,
    rng_key: Any,
    checkpoint_dir: str | None = None,
    checkpoint_every: int | None = None,
    log_every: int | None = None,
    val_step_fn: StepFn | None = None,
    val_data_iterator: Iterator[Any] | None = None,
    val_num_batches: int | None = None,
    viz_callback: VizCallback | None = None,
    viz_batch: Any | None = None
) -> tuple[TrainState, dict[str, list[float]]]:
    """Complete training loop with checkpointing, logging, and visualization.
    
    Args:
        state: Initial training state
        train_step_fn: Training step function (state, batch, rng) -> (state, metrics)
        data_iterator_fn: Function that returns fresh data iterator each epoch
        num_epochs: Number of epochs to train
        num_batches: Batches per epoch
        rng_key: Initial JAX random key
        checkpoint_dir: Directory for checkpoints (None = no checkpoints)
        checkpoint_every: Save checkpoint every N epochs (None = only final)
        log_every: Log batch metrics every N batches (None = epoch only)
        val_step_fn: Optional validation step function
        val_data_iterator: Optional validation data iterator
        val_num_batches: Number of validation batches
        viz_callback: Optional visualization callback (state, batch, rng, epoch) -> None
        viz_batch: Optional cached batch for visualization (None = use last training batch)
        
    Returns:
        Tuple of (final_state, metrics_history)
    """
    history: dict[str, list[float]] = {}
    cached_viz_batch = viz_batch
    
    # Use tqdm if available, otherwise simple range
    epoch_iter = tqdm(range(num_epochs), desc="Training") if HAS_TQDM else range(num_epochs)
    
    for epoch in epoch_iter:
        # Create fresh data iterator for this epoch
        data_iter = data_iterator_fn()
        
        # Train one epoch
        state, train_metrics, rng_key = train_epoch(
            state=state,
            train_step_fn=train_step_fn,
            data_iterator=data_iter,
            num_batches=num_batches,
            rng_key=rng_key,
            log_every=log_every
        )
        
        # Cache batch for visualization if not provided
        if viz_callback is not None and cached_viz_batch is None:
            # Get one batch for visualization
            viz_iter = data_iterator_fn()
            cached_viz_batch = next(viz_iter)
        
        # Accumulate metrics with train_ prefix
        train_metrics_prefixed = Metrics(
            values={f"train_{k}": v for k, v in train_metrics.values.items()}
        )
        history = accumulate_history(history, train_metrics_prefixed)
        
        # Optional validation
        if val_step_fn is not None and val_data_iterator is not None:
            rng_key, val_key = random.split(rng_key)
            _, val_metrics, _ = train_epoch(
                state=state,
                train_step_fn=val_step_fn,
                data_iterator=val_data_iterator,
                num_batches=val_num_batches or 1,
                rng_key=val_key,
                log_every=None
            )
            val_metrics_prefixed = Metrics(
                values={f"val_{k}": v for k, v in val_metrics.values.items()}
            )
            history = accumulate_history(history, val_metrics_prefixed)
        
        # Log epoch summary
        print(f"Epoch {epoch + 1}/{num_epochs} | {format_metrics(train_metrics)}")
        
        # Visualization callback
        if viz_callback is not None and cached_viz_batch is not None:
            rng_key, viz_key = random.split(rng_key)
            viz_callback(state, cached_viz_batch, viz_key, epoch)
        
        # Checkpoint saving
        if checkpoint_dir is not None:
            if checkpoint_every is not None and (epoch + 1) % checkpoint_every == 0:
                save_checkpoint(state, checkpoint_dir, step=epoch + 1)
    
    # Save final checkpoint
    if checkpoint_dir is not None:
        save_checkpoint(state, checkpoint_dir, step=None)
    
    return state, history

