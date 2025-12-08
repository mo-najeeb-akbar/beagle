"""Type-safe experiment configuration with hashing and serialization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any


def config_to_dict(config: Any) -> dict:
    """
    Convert a dataclass config to a dictionary, handling nested configs.

    Args:
        config: Dataclass instance

    Returns:
        Dictionary representation
    """
    if hasattr(config, '__dataclass_fields__'):
        result = {}
        for field in fields(config):
            value = getattr(config, field.name)
            if hasattr(value, '__dataclass_fields__'):
                result[field.name] = config_to_dict(value)
            else:
                result[field.name] = value
        return result
    return config


def config_hash(config: Any) -> str:
    """
    Compute deterministic hash of a configuration.

    Uses SHA256 hash of the sorted JSON representation for reproducibility.

    Args:
        config: Dataclass configuration

    Returns:
        8-character hex hash
    """
    config_dict = config_to_dict(config)
    # Sort keys for deterministic hashing
    json_str = json.dumps(config_dict, sort_keys=True, default=str)
    hash_obj = hashlib.sha256(json_str.encode())
    return hash_obj.hexdigest()[:8]


def save_config(config: Any, path: str | Path) -> None:
    """
    Save configuration to JSON file.

    Args:
        config: Dataclass configuration
        path: Output JSON file path
    """
    config_dict = config_to_dict(config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)


def load_config(path: str | Path, config_cls: type) -> Any:
    """
    Load configuration from JSON file.

    Args:
        path: JSON file path
        config_cls: Dataclass type to instantiate

    Returns:
        Instantiated config object
    """
    with open(path, 'r') as f:
        config_dict = json.load(f)

    return config_cls(**config_dict)


def merge_configs(*configs: Any) -> dict:
    """
    Merge multiple configurations into a single dictionary.

    Useful for combining dataset, model, and training configs.

    Args:
        configs: Multiple dataclass configurations

    Returns:
        Merged dictionary
    """
    merged = {}
    for config in configs:
        config_dict = config_to_dict(config)
        # Prefix with config class name
        cls_name = config.__class__.__name__.replace('Config', '').lower()
        merged[cls_name] = config_dict
    return merged


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Top-level experiment configuration.

    Combines all sub-configs into a single immutable config.
    """
    name: str
    dataset: Any
    model: Any
    training: Any
    seed: int = 42

    def hash(self) -> str:
        """Get deterministic hash of this configuration."""
        return config_hash(self)

    def save(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        save_config(self, path)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return config_to_dict(self)
