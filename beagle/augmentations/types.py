"""Type definitions for augmentation pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any
import numpy as np


class Transform(Protocol):
    """Protocol for augmentation transforms."""

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None, **kwargs: Any) -> dict[str, np.ndarray]:
        """Apply transform to image and optionally mask."""
        ...


@dataclass(frozen=True)
class AugmentConfig:
    """Configuration for augmentation pipeline (immutable)."""

    # Geometric transforms
    flip_horizontal: bool = True
    flip_vertical: bool = True
    rotate_90: bool = True
    rotation_limit: float = 0.0  # degrees
    shift_limit: float = 0.0  # fraction of image size
    scale_limit: tuple[float, float] = (1.0, 1.0)  # (min, max) scale factors
    
    # Color transforms
    brightness_limit: float = 0.0  # additive range [-limit, limit]
    contrast_limit: tuple[float, float] = (1.0, 1.0)  # (min, max) contrast factors
    hue_shift_limit: float = 0.0  # degrees
    saturation_limit: tuple[float, float] = (1.0, 1.0)  # (min, max) saturation factors
    
    # Noise & blur
    gaussian_noise_var: float = 0.0  # variance
    gaussian_blur_limit: tuple[int, int] = (3, 3)  # (min, max) kernel size
    
    # Probability for each transform group
    geometric_prob: float = 1.0
    color_prob: float = 1.0
    noise_prob: float = 1.0
    
    # General
    random_seed: int | None = None


# Default configs for common use cases
MINIMAL_AUGMENT = AugmentConfig(
    flip_horizontal=True,
    flip_vertical=True,
    rotate_90=True,
)

MODERATE_AUGMENT = AugmentConfig(
    flip_horizontal=True,
    flip_vertical=True,
    rotate_90=True,
    rotation_limit=15.0,
    shift_limit=0.1,
    scale_limit=(0.9, 1.1),
    brightness_limit=0.1,
    contrast_limit=(0.9, 1.1),
)

HEAVY_AUGMENT = AugmentConfig(
    flip_horizontal=True,
    flip_vertical=True,
    rotate_90=True,
    rotation_limit=30.0,
    shift_limit=0.2,
    scale_limit=(0.8, 1.2),
    brightness_limit=0.2,
    contrast_limit=(0.8, 1.2),
    hue_shift_limit=20.0,
    saturation_limit=(0.8, 1.2),
    gaussian_noise_var=0.01,
    gaussian_blur_limit=(3, 7),
)

