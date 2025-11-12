"""
Augmentation transform implementations using albumentations.

Pure functional interface - transforms are created once and applied multiple times.
"""
from __future__ import annotations

from typing import Any
import numpy as np

try:
    import albumentations as A
except ImportError as e:
    raise ImportError(
        "albumentations is required for augmentations. "
        "Install with: pip install albumentations"
    ) from e

from .types import AugmentConfig, Transform


def geometric_transforms(config: AugmentConfig) -> list[Any]:
    """Create geometric augmentation transforms from config (pure)."""
    transforms = []
    
    if config.flip_horizontal:
        transforms.append(A.HorizontalFlip(p=0.5))
    
    if config.flip_vertical:
        transforms.append(A.VerticalFlip(p=0.5))
    
    if config.rotate_90:
        transforms.append(A.RandomRotate90(p=0.5))
    
    if config.rotation_limit > 0:
        transforms.append(
            A.Rotate(
                limit=config.rotation_limit,
                border_mode=0,  # BORDER_REFLECT_101
                p=1.0,
            )
        )
    
    if config.shift_limit > 0:
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=config.shift_limit,
                scale_limit=0.0,
                rotate_limit=0,
                border_mode=0,
                p=1.0,
            )
        )
    
    if config.scale_limit != (1.0, 1.0):
        min_scale, max_scale = config.scale_limit
        scale_limit = (min_scale - 1.0, max_scale - 1.0)
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=0.0,
                scale_limit=scale_limit,
                rotate_limit=0,
                border_mode=0,
                p=1.0,
            )
        )
    
    return transforms


def color_transforms(config: AugmentConfig) -> list[Any]:
    """Create color augmentation transforms from config (pure)."""
    transforms = []
    
    if config.brightness_limit > 0:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=config.brightness_limit,
                contrast_limit=0.0,
                p=1.0,
            )
        )
    
    if config.contrast_limit != (1.0, 1.0):
        min_c, max_c = config.contrast_limit
        contrast_limit = (min_c - 1.0, max_c - 1.0)
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=0.0,
                contrast_limit=contrast_limit,
                p=1.0,
            )
        )
    
    if config.hue_shift_limit > 0 or config.saturation_limit != (1.0, 1.0):
        min_sat, max_sat = config.saturation_limit
        sat_limit = (int((min_sat - 1.0) * 100), int((max_sat - 1.0) * 100))
        transforms.append(
            A.HueSaturationValue(
                hue_shift_limit=int(config.hue_shift_limit),
                sat_shift_limit=sat_limit,
                val_shift_limit=0,
                p=1.0,
            )
        )
    
    return transforms


def noise_transforms(config: AugmentConfig) -> list[Any]:
    """Create noise/blur augmentation transforms from config (pure)."""
    transforms = []
    
    if config.gaussian_noise_var > 0:
        # albumentations expects std dev, not variance
        std = config.gaussian_noise_var ** 0.5
        transforms.append(
            A.GaussNoise(
                var_limit=(std * 255) ** 2,  # albumentations expects 0-255 scale
                p=1.0,
            )
        )
    
    if config.gaussian_blur_limit != (3, 3) or config.gaussian_blur_limit[0] > 3:
        transforms.append(
            A.GaussianBlur(
                blur_limit=config.gaussian_blur_limit,
                p=0.5,
            )
        )
    
    return transforms


def create_transform(config: AugmentConfig) -> Transform:
    """
    Create a composed transform from config (pure function).
    
    Args:
        config: Augmentation configuration
    
    Returns:
        A transform function that can be applied to images/masks
    """
    transform_groups = []
    
    # Build geometric transforms
    geom = geometric_transforms(config)
    if geom:
        transform_groups.append(A.OneOf(geom, p=config.geometric_prob))
    
    # Build color transforms
    color = color_transforms(config)
    if color:
        transform_groups.append(A.OneOf(color, p=config.color_prob))
    
    # Build noise transforms
    noise = noise_transforms(config)
    if noise:
        transform_groups.append(A.OneOf(noise, p=config.noise_prob))
    
    if not transform_groups:
        # No augmentations - return identity transform
        return A.Compose([])
    
    return A.Compose(transform_groups)


def apply_transform(
    transform: Transform,
    image: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Apply transform to image and optionally mask (pure function).
    
    Args:
        transform: Transform created by create_transform()
        image: Image array (H, W, C) in range [0, 1]
        mask: Optional mask array (H, W) or (H, W, C)
    
    Returns:
        Dictionary with 'image' key and optionally 'mask' key
    """
    # albumentations expects uint8 images
    image_uint8 = (image * 255).astype(np.uint8)
    
    if mask is not None:
        # Ensure mask is uint8
        if mask.dtype != np.uint8:
            mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
        else:
            mask_uint8 = mask
        
        result = transform(image=image_uint8, mask=mask_uint8)
        return {
            'image': result['image'].astype(np.float32) / 255.0,
            'mask': result['mask'].astype(np.float32) / 255.0,
        }
    else:
        result = transform(image=image_uint8)
        return {
            'image': result['image'].astype(np.float32) / 255.0,
        }

