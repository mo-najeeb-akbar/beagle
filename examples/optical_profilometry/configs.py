"""Configuration dataclasses for optical profilometry experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset configuration."""
    batch_size: int = 32
    crop_size: int = 256
    crop_overlap: int = 192
    val_split: float = 0.2
    split_seed: int = 42


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture configuration."""
    base_features: int = 32
    latent_dim: int = 128
    wavelet_weights: tuple[float, ...] = (20.0, 8.0, 8.0, 12.0)


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters."""
    learning_rate: float = 0.001
    num_epochs: int = 10
    optimizer: str = 'adamw'
