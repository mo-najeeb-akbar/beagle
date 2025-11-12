"""
Augmentation pipeline with functional interface.

Uses albumentations under the hood but provides a clean, composable API.
Supports both image-only and image+mask augmentation for segmentation tasks.
"""
from __future__ import annotations

from .types import (
    AugmentConfig,
    Transform,
    MINIMAL_AUGMENT,
    MODERATE_AUGMENT,
    HEAVY_AUGMENT,
)
from .transforms import (
    create_transform,
    apply_transform,
    geometric_transforms,
    color_transforms,
    noise_transforms,
)

__all__ = [
    # Types
    "AugmentConfig",
    "Transform",
    # Preset configs
    "MINIMAL_AUGMENT",
    "MODERATE_AUGMENT",
    "HEAVY_AUGMENT",
    # Core functions
    "create_transform",
    "apply_transform",
    # Transform builders
    "geometric_transforms",
    "color_transforms",
    "noise_transforms",
]

