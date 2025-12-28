"""Configuration dataclasses for segmentation experiments."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset configuration."""
    batch_size: int = 32
    val_split: float = 0.2
    split_seed: int = 42
    shuffle: bool = True
    augment: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture configuration."""
    input_size: int = 512
    num_stages: int = 3
    features: int = 32
    target_res: float = 1.0
    train_backbone: bool = True
    upsample_to_full_res: bool = True
    outputs: tuple = field(default_factory=lambda: ((3, False, 2),))


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters."""
    learning_rate: float = 0.001
    num_epochs: int = 30
    optimizer: str = 'adamw'
