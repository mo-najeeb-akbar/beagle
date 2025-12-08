"""Hyperparameter sweep utilities."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator


@dataclass(frozen=True)
class ParamSpec:
    """
    Specification for a hyperparameter.

    Supports discrete values or continuous ranges.
    """
    name: str
    values: list | None = None  # Discrete values
    min_val: float | None = None  # Continuous range min
    max_val: float | None = None  # Continuous range max
    log_scale: bool = False  # Use log scale for continuous ranges

    def __post_init__(self):
        """Validate parameter specification."""
        if self.values is None:
            if self.min_val is None or self.max_val is None:
                raise ValueError(
                    f"Parameter {self.name} must have either 'values' or 'min_val'/'max_val'"
                )
        else:
            if self.min_val is not None or self.max_val is not None:
                raise ValueError(
                    f"Parameter {self.name} cannot have both 'values' and 'min_val'/'max_val'"
                )


def grid_search(param_specs: list[ParamSpec]) -> Iterator[dict[str, Any]]:
    """
    Generate grid search configurations.

    Args:
        param_specs: List of parameter specifications

    Yields:
        Configuration dictionaries
    """
    # Extract param names and values
    param_names = [spec.name for spec in param_specs]
    param_values = []

    for spec in param_specs:
        if spec.values is not None:
            param_values.append(spec.values)
        else:
            raise ValueError(
                f"Grid search requires discrete values for {spec.name}. "
                f"Use random_search for continuous parameters."
            )

    # Generate all combinations
    for combination in itertools.product(*param_values):
        yield dict(zip(param_names, combination))


def random_search(
    param_specs: list[ParamSpec],
    n_samples: int,
    seed: int = 42
) -> Iterator[dict[str, Any]]:
    """
    Generate random search configurations.

    Args:
        param_specs: List of parameter specifications
        n_samples: Number of random configurations to generate
        seed: Random seed for reproducibility

    Yields:
        Configuration dictionaries
    """
    rng = random.Random(seed)

    for _ in range(n_samples):
        config = {}

        for spec in param_specs:
            if spec.values is not None:
                # Discrete choice
                config[spec.name] = rng.choice(spec.values)
            else:
                # Continuous range
                if spec.log_scale:
                    # Sample in log space
                    import math
                    log_min = math.log10(spec.min_val)
                    log_max = math.log10(spec.max_val)
                    log_val = rng.uniform(log_min, log_max)
                    config[spec.name] = 10 ** log_val
                else:
                    # Linear sampling
                    config[spec.name] = rng.uniform(spec.min_val, spec.max_val)

        yield config


def update_config_with_params(base_config: Any, params: dict[str, Any]) -> Any:
    """
    Update a dataclass config with new parameter values.

    Supports nested configs using dot notation (e.g., 'training.learning_rate').

    Args:
        base_config: Base configuration dataclass
        params: Parameter dictionary with keys using dot notation

    Returns:
        New config with updated parameters
    """
    updates = {}

    for param_path, value in params.items():
        parts = param_path.split('.')

        if len(parts) == 1:
            # Top-level parameter
            updates[param_path] = value
        else:
            # Nested parameter - need to reconstruct nested config
            nested_config_name = parts[0]
            nested_param = '.'.join(parts[1:])

            # Get the nested config
            nested_config = getattr(base_config, nested_config_name)

            # Recursively update
            if nested_config_name not in updates:
                updates[nested_config_name] = nested_config

            updates[nested_config_name] = update_config_with_params(
                updates[nested_config_name],
                {nested_param: value}
            )

    return replace(base_config, **updates)


def run_sweep(
    base_config: Any,
    param_specs: list[ParamSpec],
    train_fn: Callable[[Any], dict[str, float]],
    method: str = 'grid',
    n_samples: int = 10,
    seed: int = 42
) -> list[dict]:
    """
    Run a hyperparameter sweep.

    Args:
        base_config: Base configuration to modify
        param_specs: List of parameter specifications
        train_fn: Training function that takes config and returns metrics dict
        method: 'grid' or 'random'
        n_samples: Number of samples for random search
        seed: Random seed

    Returns:
        List of results with configs and metrics
    """
    # Generate configurations
    if method == 'grid':
        configs = grid_search(param_specs)
    elif method == 'random':
        configs = random_search(param_specs, n_samples, seed)
    else:
        raise ValueError(f"Unknown sweep method: {method}")

    results = []

    for i, params in enumerate(configs):
        print(f"\n{'=' * 80}")
        print(f"Sweep iteration {i + 1}")
        print(f"Parameters: {params}")
        print('=' * 80)

        # Update config with sweep parameters
        sweep_config = update_config_with_params(base_config, params)

        # Run training
        try:
            metrics = train_fn(sweep_config)
            status = 'completed'
        except Exception as e:
            print(f"Training failed: {e}")
            metrics = {}
            status = 'failed'

        results.append({
            'params': params,
            'config': sweep_config,
            'metrics': metrics,
            'status': status
        })

    return results


def get_best_config(
    results: list[dict],
    metric_name: str,
    mode: str = 'min'
) -> dict | None:
    """
    Get the best configuration from sweep results.

    Args:
        results: Sweep results
        metric_name: Metric to optimize
        mode: 'min' or 'max'

    Returns:
        Best result dictionary or None
    """
    # Filter successful runs with the metric
    valid_results = [
        r for r in results
        if r['status'] == 'completed' and metric_name in r['metrics']
    ]

    if not valid_results:
        return None

    # Sort by metric
    reverse = (mode == 'max')
    valid_results.sort(
        key=lambda r: r['metrics'][metric_name],
        reverse=reverse
    )

    return valid_results[0]
