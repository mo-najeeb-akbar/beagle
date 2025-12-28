from pathlib import Path
import numpy as np
import cv2

# Force TensorFlow to use CPU only - MUST be done before importing TensorFlow modules
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from beagle.dataset import (
    load_fields_from_disk,
    create_disk_iterator,
)
from beagle.augmentations import (
    compose,
    random_flip_left_right,
    random_flip_up_down,
    random_rotate_90,
    random_brightness,
    random_contrast,
    random_gaussian_noise,
    random_pixel_dropout,
    random_gaussian_blur,
)

from fast_augmentations import apply_fast_augmentation


# ============================================================================
# Example: Segmentation with Dataset Multiplication via Augmentations
# ============================================================================

def load_image(path: str) -> np.ndarray:
    """Load and resize grayscale image."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    return img[:, :, np.newaxis]


def load_mask(path: str) -> np.ndarray:
    """Load and resize binary mask."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    # mask = (mask > 0).astype(np.uint8)
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    return mask[:, :, np.newaxis]


def create_segmentation_iterator(dataset_path: str, augmentation_multiplier: int = 20):
    """Load images and masks for semantic segmentation with dataset multiplication.

    Args:
        dataset_path: Path to directory containing images and masks
        augmentation_multiplier: Factor to multiply dataset size (default: 20x)

    Returns:
        tuple: (iterator, num_classes) where iterator yields batches and num_classes is int
    """
    image_dir = Path(dataset_path)
    mask_files = sorted(image_dir.glob("*_mask.png"))

    # Build file list with image-mask pairs
    file_list = []
    for mask_path in mask_files:
        image_path = Path(str(mask_path).split('_mask.png')[0] + '.png')
        if image_path.exists():
            file_list.append({'image': image_path, 'mask': mask_path})

    if not file_list:
        raise ValueError("No valid image/label pairs found")

    original_size = len(file_list)
    print(f"Found {original_size} image-mask pairs")

    # Multiply dataset by repeating file list
    # Each copy will get different augmentations due to random transforms
    file_list = file_list * augmentation_multiplier
    print(f"Dataset multiplied to {len(file_list)} samples ({augmentation_multiplier}x)")

    # Load data with fast geometric augmentations applied first
    data = load_fields_from_disk(
        file_list=file_list,
        field_loaders={
            'image': load_image,
            'mask': load_mask,
        },
        sample_transform=apply_fast_augmentation,  # Apply fast augmentations
        field_transforms={
            'image': lambda x: (x - 127.5) / 127.5,  # Normalize to [-1, 1]
            'mask': lambda x: x,  # Keep masks as-is
        },
    )

    print(f"Loaded {len(data['image'])} augmented samples")

    # Compute number of classes from mask data
    unique_classes = np.unique(data['mask'])
    num_classes = len(unique_classes)
    print(f"Detected {num_classes} classes in masks: {unique_classes}")

    # Chain beagle augmentations on top (runs on CPU)
    aug_fn = compose(
        random_flip_left_right(fields=['image', 'mask']),
        random_flip_up_down(fields=['image', 'mask']),
        random_rotate_90(fields=['image', 'mask']),
        # Photometric augmentations (image only)
        random_brightness(max_delta=0.2, field='image'),
        random_contrast(lower=0.9, upper=1.1, field='image'),
        random_gaussian_noise(stddev=0.05, field='image'),
        random_pixel_dropout(drop_prob=0.1, field='image'),
        random_gaussian_blur(sigma_range=(0.1, 1.0), field='image'),
    )

    # Create iterator with CPU-based augmentations
    iterator = create_disk_iterator(
        data=data,
        batch_size=32,
        shuffle=True,
        repeat=True,
        augment_fn=aug_fn,  # Runs on CPU only
    )

    return iterator, num_classes
