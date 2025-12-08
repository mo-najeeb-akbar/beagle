"""Single experiment run tracking."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentMetadata:
    """Metadata for a single experiment."""
    experiment_id: str
    config_hash: str
    timestamp: str
    name: str
    config: dict
    status: str = 'running'  # running, completed, failed


@dataclass
class ExperimentRun:
    """
    Tracks a single experiment run.

    Provides methods to log metrics, save checkpoints, and update status.
    """
    experiment_id: str
    output_dir: Path
    config: Any
    metadata: ExperimentMetadata
    _metrics: dict[str, list] = field(default_factory=dict)
    _start_time: float = field(default_factory=time.time)

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """
        Log a metric value.

        Args:
            name: Metric name (e.g., 'train_loss', 'val_accuracy')
            value: Metric value
            step: Optional step/epoch number
        """
        if name not in self._metrics:
            self._metrics[name] = []

        entry = {'value': float(value), 'timestamp': time.time()}
        if step is not None:
            entry['step'] = step

        self._metrics[name].append(entry)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step/epoch number
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def get_metric_history(self, name: str) -> list[dict]:
        """Get full history for a metric."""
        return self._metrics.get(name, [])

    def get_latest_metric(self, name: str) -> float | None:
        """Get the latest value for a metric."""
        history = self.get_metric_history(name)
        return history[-1]['value'] if history else None

    def get_best_metric(self, name: str, mode: str = 'min') -> float | None:
        """
        Get the best value for a metric.

        Args:
            name: Metric name
            mode: 'min' or 'max'

        Returns:
            Best metric value
        """
        history = self.get_metric_history(name)
        if not history:
            return None

        values = [entry['value'] for entry in history]
        return min(values) if mode == 'min' else max(values)

    def save_metrics(self) -> None:
        """Save metrics to JSON file."""
        metrics_path = self.output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self._metrics, f, indent=2)

    def save_metadata(self, status: str = 'completed') -> None:
        """
        Save experiment metadata.

        Args:
            status: Experiment status ('running', 'completed', 'failed')
        """
        metadata_dict = {
            'experiment_id': self.metadata.experiment_id,
            'config_hash': self.metadata.config_hash,
            'timestamp': self.metadata.timestamp,
            'name': self.metadata.name,
            'status': status,
            'duration_seconds': time.time() - self._start_time,
            'config': self.metadata.config,
        }

        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2, default=str)

    def finish(self, status: str = 'completed') -> None:
        """
        Mark experiment as finished and save all data.

        Args:
            status: Final status ('completed' or 'failed')
        """
        self.save_metrics()
        self.save_metadata(status)

    def summary(self) -> str:
        """Get a summary of the experiment."""
        lines = [
            f"Experiment: {self.metadata.name}",
            f"ID: {self.experiment_id}",
            f"Config Hash: {self.metadata.config_hash}",
            f"Output: {self.output_dir}",
            "",
            "Latest Metrics:",
        ]

        for name in sorted(self._metrics.keys()):
            latest = self.get_latest_metric(name)
            lines.append(f"  {name}: {latest:.4f}")

        return "\n".join(lines)
