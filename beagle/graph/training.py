"""Generic training utilities for ComputeGraph and GraphState.

Provides:
- Generic train_step factory
- Training loop with callbacks
- Automatic gradient computation and state updates
"""

from typing import Callable, Any
import jax
import jax.numpy as jnp
from .state import GraphState


def create_train_step(
    loss_fn: Callable[[dict[str, Any], dict[str, Any]], Any],
    aux_metrics_fn: Callable[[dict[str, Any], dict[str, Any]], dict] | None = None
) -> Callable:
    """Create a JIT-compiled training step for a GraphState.

    Args:
        loss_fn: Function(outputs, batch) -> scalar loss
        aux_metrics_fn: Optional function(outputs, batch) -> metrics dict

    Returns:
        train_step: Function(state, batch) -> (new_state, metrics)

    Example:
        >>> def loss_fn(outputs, batch):
        ...     return jnp.mean((outputs['logits'] - batch['labels'])**2)
        >>>
        >>> train_step = create_train_step(loss_fn)
        >>> state, metrics = train_step(state, batch)
    """

    @jax.jit
    def train_step(
        state: GraphState,
        batch: dict[str, Any]
    ) -> tuple[GraphState, dict[str, Any]]:
        """Single training step.

        Args:
            state: Current GraphState
            batch: Batch of data

        Returns:
            new_state: Updated GraphState
            metrics: Dictionary of metrics including loss
        """

        def forward_and_loss(trainable_params):
            """Forward pass through graph and compute loss."""
            # Merge trainable params into full variables
            full_vars = {}
            for node_name, node_vars in state.variables.items():
                if node_name in trainable_params:
                    # This node is trainable - use provided params
                    full_vars[node_name] = {
                        **node_vars,
                        'params': trainable_params[node_name]
                    }
                else:
                    # This node is frozen - use existing vars
                    full_vars[node_name] = node_vars

            # Forward pass through graph
            outputs, updates = state.graph(full_vars, batch, train=True)

            # Compute loss
            loss = loss_fn(outputs, batch)

            return loss, (outputs, updates)

        # Get trainable parameters
        trainable_params = state.trainable_params()

        # Compute gradients
        (loss, (outputs, updates)), grads = jax.value_and_grad(
            forward_and_loss,
            has_aux=True
        )(trainable_params)

        # Wrap grads in dict structure expected by apply_gradients
        grads_dict = {name: {'params': grad} for name, grad in grads.items()}

        # Update state
        new_state = state.apply_gradients(grads_dict)
        new_state = new_state.merge_updates(updates)

        # Collect metrics
        metrics = {'loss': loss}
        if aux_metrics_fn is not None:
            aux_metrics = aux_metrics_fn(outputs, batch)
            metrics.update(aux_metrics)

        return new_state, metrics

    return train_step


def create_eval_step(
    loss_fn: Callable[[dict[str, Any], dict[str, Any]], Any],
    metrics_fn: Callable[[dict[str, Any], dict[str, Any]], dict] | None = None
) -> Callable:
    """Create a JIT-compiled evaluation step.

    Args:
        loss_fn: Function(outputs, batch) -> scalar loss
        metrics_fn: Optional function(outputs, batch) -> metrics dict

    Returns:
        eval_step: Function(state, batch) -> metrics

    Example:
        >>> def metrics_fn(outputs, batch):
        ...     preds = jnp.argmax(outputs['logits'], axis=-1)
        ...     acc = jnp.mean(preds == batch['labels'])
        ...     return {'accuracy': acc}
        >>>
        >>> eval_step = create_eval_step(loss_fn, metrics_fn)
        >>> metrics = eval_step(state, batch)
    """

    @jax.jit
    def eval_step(
        state: GraphState,
        batch: dict[str, Any]
    ) -> dict[str, Any]:
        """Single evaluation step.

        Args:
            state: Current GraphState
            batch: Batch of data

        Returns:
            metrics: Dictionary of metrics
        """
        # Forward pass (no gradients, no mutable updates needed)
        outputs, _ = state.graph(state.variables, batch, train=False)

        # Compute loss
        loss = loss_fn(outputs, batch)

        # Collect metrics
        metrics = {'loss': loss}
        if metrics_fn is not None:
            aux_metrics = metrics_fn(outputs, batch)
            metrics.update(aux_metrics)

        return metrics

    return eval_step


def train_epoch(
    state: GraphState,
    train_iter,
    train_step: Callable,
    num_batches: int | None = None
) -> tuple[GraphState, dict[str, float]]:
    """Train for one epoch.

    Args:
        state: Current GraphState
        train_iter: Iterator yielding batches
        train_step: Training step function
        num_batches: Optional number of batches (for progress tracking)

    Returns:
        new_state: Updated GraphState
        metrics: Averaged metrics over the epoch
    """
    metrics_accum = {}
    num_steps = 0

    for batch in train_iter:
        state, metrics = train_step(state, batch)

        # Accumulate metrics
        for k, v in metrics.items():
            if k not in metrics_accum:
                metrics_accum[k] = 0.0
            metrics_accum[k] += float(v)

        num_steps += 1
        if num_batches is not None and num_steps >= num_batches:
            break

    # Average metrics
    avg_metrics = {k: v / num_steps for k, v in metrics_accum.items()}

    return state, avg_metrics


def evaluate(
    state: GraphState,
    eval_iter,
    eval_step: Callable,
    num_batches: int | None = None
) -> dict[str, float]:
    """Evaluate on a dataset.

    Args:
        state: Current GraphState
        eval_iter: Iterator yielding batches
        eval_step: Evaluation step function
        num_batches: Optional number of batches to evaluate

    Returns:
        metrics: Averaged metrics over the dataset
    """
    metrics_accum = {}
    num_steps = 0

    for batch in eval_iter:
        metrics = eval_step(state, batch)

        # Accumulate metrics
        for k, v in metrics.items():
            if k not in metrics_accum:
                metrics_accum[k] = 0.0
            metrics_accum[k] += float(v)

        num_steps += 1
        if num_batches is not None and num_steps >= num_batches:
            break

    # Average metrics
    avg_metrics = {k: v / num_steps for k, v in metrics_accum.items()}

    return avg_metrics


def simple_training_loop(
    state: GraphState,
    train_iter,
    num_epochs: int,
    train_step: Callable,
    eval_iter=None,
    eval_step: Callable | None = None,
    checkpoint_fn: Callable[[GraphState, int], None] | None = None,
    log_every: int = 1
) -> GraphState:
    """Simple training loop with optional evaluation and checkpointing.

    Args:
        state: Initial GraphState
        train_iter: Training data iterator
        num_epochs: Number of epochs to train
        train_step: Training step function
        eval_iter: Optional validation data iterator
        eval_step: Optional evaluation step function
        checkpoint_fn: Optional function(state, epoch) to save checkpoints
        log_every: Log metrics every N epochs

    Returns:
        final_state: Final GraphState after training
    """
    for epoch in range(num_epochs):
        # Training
        state, train_metrics = train_epoch(state, train_iter, train_step)

        # Logging
        if epoch % log_every == 0:
            log_msg = f"Epoch {epoch}/{num_epochs}"
            log_msg += f" | train_loss: {train_metrics['loss']:.4f}"

            # Evaluation
            if eval_iter is not None and eval_step is not None:
                eval_metrics = evaluate(state, eval_iter, eval_step)
                log_msg += f" | val_loss: {eval_metrics['loss']:.4f}"

            print(log_msg)

        # Checkpointing
        if checkpoint_fn is not None:
            checkpoint_fn(state, epoch)

    return state
