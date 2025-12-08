"""Model registry for versioning and loading trained models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orbax.checkpoint as ocp


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata for a registered model."""
    model_name: str
    experiment_id: str
    config_hash: str
    checkpoint_path: str
    metrics: dict[str, float]
    timestamp: str
    model_hash: str  # Hash of checkpoint for integrity


class ModelRegistry:
    """
    Registry for managing trained models.

    Provides easy loading of models by experiment ID or performance metrics.
    """

    def __init__(self, registry_dir: str | Path):
        """
        Initialize model registry.

        Args:
            registry_dir: Directory to store registry metadata
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / 'registry.json'

        # Load existing registry
        self._registry = self._load_registry()

    def _load_registry(self) -> dict:
        """Load registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_registry(self) -> None:
        """Save registry to disk."""
        with open(self.registry_file, 'w') as f:
            json.dump(self._registry, f, indent=2, default=str)

    def _compute_checkpoint_hash(self, checkpoint_dir: Path) -> str:
        """
        Compute hash of checkpoint directory for integrity checking.

        Args:
            checkpoint_dir: Path to checkpoint directory

        Returns:
            SHA256 hash (first 8 characters)
        """
        # Hash all files in checkpoint directory
        hasher = hashlib.sha256()

        for file_path in sorted(checkpoint_dir.rglob('*')):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())

        return hasher.hexdigest()[:8]

    def register_model(
        self,
        model_name: str,
        experiment_id: str,
        config_hash: str,
        checkpoint_path: str | Path,
        metrics: dict[str, float],
        timestamp: str
    ) -> str:
        """
        Register a trained model.

        Args:
            model_name: Human-readable model name
            experiment_id: Experiment ID that produced this model
            config_hash: Configuration hash
            checkpoint_path: Path to model checkpoint
            metrics: Final metrics for this model
            timestamp: Timestamp of registration

        Returns:
            Model ID (experiment_id for now)
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

        # Compute checkpoint hash
        model_hash = self._compute_checkpoint_hash(checkpoint_path)

        # Create metadata
        metadata = {
            'model_name': model_name,
            'experiment_id': experiment_id,
            'config_hash': config_hash,
            'checkpoint_path': str(checkpoint_path.absolute()),
            'metrics': metrics,
            'timestamp': timestamp,
            'model_hash': model_hash
        }

        # Register
        self._registry[experiment_id] = metadata
        self._save_registry()

        return experiment_id

    def get_model_metadata(self, experiment_id: str) -> dict | None:
        """
        Get metadata for a registered model.

        Args:
            experiment_id: Experiment ID

        Returns:
            Model metadata or None
        """
        return self._registry.get(experiment_id)

    def load_model_checkpoint(self, experiment_id: str) -> Any:
        """
        Load model checkpoint by experiment ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            Loaded checkpoint (TrainState with params)
        """
        metadata = self.get_model_metadata(experiment_id)

        if metadata is None:
            raise ValueError(f"Model {experiment_id} not found in registry")

        checkpoint_path = Path(metadata['checkpoint_path'])

        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")

        # Verify integrity
        current_hash = self._compute_checkpoint_hash(checkpoint_path)
        if current_hash != metadata['model_hash']:
            raise ValueError(
                f"Checkpoint integrity check failed for {experiment_id}. "
                f"Expected {metadata['model_hash']}, got {current_hash}"
            )

        # Load checkpoint using Orbax
        checkpointer = ocp.StandardCheckpointer()
        state = checkpointer.restore(checkpoint_path)

        return state

    def list_models(self) -> list[dict]:
        """
        List all registered models.

        Returns:
            List of model metadata dictionaries
        """
        return list(self._registry.values())

    def find_best_model(self, metric_name: str, mode: str = 'min') -> dict | None:
        """
        Find the best model by a metric.

        Args:
            metric_name: Metric to optimize
            mode: 'min' or 'max'

        Returns:
            Best model metadata or None
        """
        models = self.list_models()

        # Filter models that have this metric
        valid_models = [m for m in models if metric_name in m.get('metrics', {})]

        if not valid_models:
            return None

        # Sort by metric
        reverse = (mode == 'max')
        valid_models.sort(
            key=lambda m: m['metrics'][metric_name],
            reverse=reverse
        )

        return valid_models[0]

    def delete_model(self, experiment_id: str, delete_checkpoint: bool = False) -> None:
        """
        Remove model from registry.

        Args:
            experiment_id: Experiment ID
            delete_checkpoint: If True, also delete checkpoint files
        """
        metadata = self.get_model_metadata(experiment_id)

        if metadata is None:
            raise ValueError(f"Model {experiment_id} not found")

        if delete_checkpoint:
            checkpoint_path = Path(metadata['checkpoint_path'])
            if checkpoint_path.exists():
                import shutil
                shutil.rmtree(checkpoint_path)

        del self._registry[experiment_id]
        self._save_registry()

