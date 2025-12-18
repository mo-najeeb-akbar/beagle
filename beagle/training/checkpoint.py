"""Checkpoint saving and loading utilities (side effects isolated)."""

from __future__ import annotations

import os
import json
from typing import Any

import orbax.checkpoint as ocp

from beagle.training.types import TrainState


def save_checkpoint(
    state: TrainState,
    checkpoint_dir: str,
    step: int | None = None
) -> None:
    """Save training state to disk (side effect, BLOCKING).

    This function performs synchronous (blocking) saves to avoid
    race conditions and permission errors from async operations.

    Args:
        state: Training state to save
        checkpoint_dir: Directory to save checkpoint
        step: Optional step number for checkpoint name
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create checkpoint data
    ckpt = {
        'params': state.params,
        'opt_state': state.opt_state
    }

    if state.batch_stats is not None:
        ckpt['batch_stats'] = state.batch_stats

    # Generate checkpoint path
    if step is not None:
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
    else:
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint_final")

    # Create a new synchronous checkpointer for each save
    # This avoids threading issues and ensures saves complete immediately
    checkpointer = ocp.StandardCheckpointer()
    
    # Save and wait for completion (blocking)
    checkpointer.save(os.path.abspath(ckpt_path), ckpt, force=True)
    
    # Wait for all pending operations to complete
    checkpointer.wait_until_finished()
    
    # Close to release resources immediately
    checkpointer.close()


def load_checkpoint(
    checkpoint_path: str,
) -> dict[str, Any]:
    """Load checkpoint and return parameters.

    Args:
        checkpoint_path: Path to checkpoint directory
    """
    # Create a fresh checkpointer for loading (no shared state)
    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(os.path.abspath(checkpoint_path))
    checkpointer.close()
    return restored


def save_config(config: dict[str, Any], output_dir: str) -> None:
    """Save configuration as JSON (side effect).
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save config
    """
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def load_config(checkpoint_dir: str) -> dict[str, Any]:
    """Load configuration from JSON.
    
    Args:
        checkpoint_dir: Directory containing config.json
        
    Returns:
        Configuration dictionary
    """
    config_path = os.path.join(checkpoint_dir, "config.json")
    
    with open(config_path, "r") as f:
        return json.load(f)


def save_metrics_history(
    history: dict[str, list[float]],
    output_dir: str
) -> None:
    """Save metrics history as JSON (side effect).
    
    Args:
        history: Metrics history dictionary
        output_dir: Directory to save history
    """
    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, "metrics.json")
    
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

