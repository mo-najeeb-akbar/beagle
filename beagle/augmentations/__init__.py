"""
Augmentation pipeline with functional interface.

Two approaches available:
1. Simple TensorFlow augmentations (compose module) - recommended for most use cases
2. Albumentations-based augmentations (transforms module) - for advanced use cases

The compose module provides:
- Simple, composable augmentations
- No assumptions about data ranges (works with any numeric data)
- Easy to write custom augmentations
- Works directly in TensorFlow pipeline (efficient)
"""
from __future__ import annotations

# Always available: TensorFlow-based augmentations (no extra dependencies)
from .compose import (
    compose,
    apply_to_field,
    apply_to_fields,
    apply_geometric,
    random_flip_left_right,
    random_flip_up_down,
    random_rotate_90,
    random_brightness,
    random_contrast,
    random_gaussian_noise,
    random_pixel_dropout,
    random_gaussian_blur,
    clip_values,
)

# Import seed utilities from dataset module for convenience
from ..dataset.seed import set_global_seed, set_tf_deterministic

__all__ = [
    # TensorFlow-based (always available)
    "compose",
    "apply_to_field",
    "apply_to_fields",
    "apply_geometric",
    "random_flip_left_right",
    "random_flip_up_down",
    "random_rotate_90",
    "random_brightness",
    "random_contrast",
    "random_gaussian_noise",
    "random_pixel_dropout",
    "random_gaussian_blur",
    "clip_values",
    # Reproducibility
    "set_global_seed",
    "set_tf_deterministic",
]

# Optional: Albumentations-based augmentations (requires albumentations)
try:
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
    
    __all__.extend([
        # Types
        "AugmentConfig",
        "Transform",
        # Preset configs
        "MINIMAL_AUGMENT",
        "MODERATE_AUGMENT",
        "HEAVY_AUGMENT",
        # Albumentations-based (for advanced use)
        "create_transform",
        "apply_transform",
        "geometric_transforms",
        "color_transforms",
        "noise_transforms",
    ])
except ImportError:
    # Albumentations not available - only TensorFlow augmentations available
    pass

