"""Experiment tracker for managing and comparing experiments."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import config_hash, config_to_dict
from .run import ExperimentMetadata, ExperimentRun


class ExperimentTracker:
    """
    Manages multiple experiment runs.

    Provides unified interface for creating experiments, loading past runs,
    and comparing results.
    """

    def __init__(self, experiments_dir: str | Path):
        """
        Initialize experiment tracker.

        Args:
            experiments_dir: Root directory for all experiments
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment(
        self,
        name: str,
        config: Any,
        experiment_id: str | None = None
    ) -> ExperimentRun:
        """
        Create a new experiment run.

        Args:
            name: Experiment name
            config: Configuration object (dataclass)
            experiment_id: Optional custom experiment ID (default: timestamp_hash)

        Returns:
            ExperimentRun object
        """
        # Generate experiment ID
        if experiment_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cfg_hash = config_hash(config)
            experiment_id = f"{timestamp}_{cfg_hash}"

        # Create output directory
        output_dir = self.experiments_dir / experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata
        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            config_hash=config_hash(config),
            timestamp=datetime.now().isoformat(),
            name=name,
            config=config_to_dict(config),
            status='running'
        )

        # Create run
        run = ExperimentRun(
            experiment_id=experiment_id,
            output_dir=output_dir,
            config=config,
            metadata=metadata
        )

        # Save initial metadata
        run.save_metadata(status='running')

        return run

    def load_experiment(self, experiment_id: str) -> dict:
        """
        Load experiment data.

        Args:
            experiment_id: Experiment ID to load

        Returns:
            Dictionary with metadata and metrics
        """
        exp_dir = self.experiments_dir / experiment_id

        if not exp_dir.exists():
            raise ValueError(f"Experiment {experiment_id} not found")

        # Load metadata
        with open(exp_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        # Load metrics
        metrics_path = exp_dir / 'metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {}

        return {
            'metadata': metadata,
            'metrics': metrics,
            'output_dir': exp_dir
        }

    def list_experiments(self) -> list[dict]:
        """
        List all experiments.

        Returns:
            List of experiment metadata dictionaries
        """
        experiments = []

        for exp_dir in sorted(self.experiments_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            metadata_path = exp_dir / 'metadata.json'
            if not metadata_path.exists():
                continue

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                experiments.append(metadata)

        return experiments

    def compare_experiments(
        self,
        metric_name: str,
        mode: str = 'min',
        top_k: int = 10
    ) -> list[dict]:
        """
        Compare experiments by a metric.

        Args:
            metric_name: Metric to compare (e.g., 'val_loss')
            mode: 'min' or 'max'
            top_k: Number of top experiments to return

        Returns:
            List of experiment summaries sorted by metric
        """
        experiments = self.list_experiments()
        results = []

        for exp_metadata in experiments:
            exp_id = exp_metadata['experiment_id']
            exp_data = self.load_experiment(exp_id)
            metrics = exp_data['metrics']

            if metric_name not in metrics:
                continue

            # Get best value for this metric
            values = [entry['value'] for entry in metrics[metric_name]]
            best_value = min(values) if mode == 'min' else max(values)

            results.append({
                'experiment_id': exp_id,
                'name': exp_metadata['name'],
                'config_hash': exp_metadata['config_hash'],
                metric_name: best_value,
                'output_dir': exp_data['output_dir']
            })

        # Sort by metric
        reverse = (mode == 'max')
        results.sort(key=lambda x: x[metric_name], reverse=reverse)

        return results[:top_k]

    def get_best_experiment(self, metric_name: str, mode: str = 'min') -> dict | None:
        """
        Get the best experiment by a metric.

        Args:
            metric_name: Metric to optimize
            mode: 'min' or 'max'

        Returns:
            Best experiment data or None
        """
        results = self.compare_experiments(metric_name, mode, top_k=1)
        return results[0] if results else None
