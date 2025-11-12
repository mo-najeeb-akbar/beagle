"""
Example: TFRecord to JAX iterator with augmentations.

This demonstrates how to use the beagle library to create a data pipeline
that loads TFRecords, applies augmentations, and yields JAX arrays.
"""
from __future__ import annotations

import sys
from functools import partial

from beagle.dataset import create_tfrecord_iterator, compute_welford_stats
from beagle.augmentations import (
    create_transform,
    apply_transform,
    AugmentConfig,
    MODERATE_AUGMENT,
)


def main() -> None:
    """Run example dataloader."""
    if len(sys.argv) < 2:
        print("Usage: python tfrecord_to_jax_example.py <tfrecord_pattern>")
        print("Example: python tfrecord_to_jax_example.py 'data/*.tfrecord'")
        sys.exit(1)
    
    tfrecord_pattern = sys.argv[1]
    
    # Option 1: Use preset augmentation config
    print("Creating transform with moderate augmentation...")
    transform = create_transform(MODERATE_AUGMENT)
    
    # Option 2: Custom augmentation config
    # custom_config = AugmentConfig(
    #     flip_horizontal=True,
    #     flip_vertical=True,
    #     rotate_90=True,
    #     rotation_limit=15.0,
    #     shift_limit=0.1,
    #     scale_limit=(0.9, 1.1),
    #     brightness_limit=0.1,
    #     contrast_limit=(0.9, 1.1),
    #     gaussian_noise_var=0.01,
    # )
    # transform = create_transform(custom_config)
    
    # Create augmentation function (numpy -> numpy)
    def augment_fn(img):
        result = apply_transform(transform, img)
        return result['image']
    
    # Create iterator with augmentation
    print("Creating dataloader...")
    iterator, n_batches = create_tfrecord_iterator(
        tfrecord_pattern,
        batch_size=32,
        augment_fn=augment_fn,
        precomputed_stats=(0.5, 0.2),  # Optional: provide precomputed stats
        shuffle=True,
    )
    
    print(f"Batches per epoch: {n_batches}")
    
    # Get a batch
    print("Fetching first batch...")
    batch = next(iterator)
    print(f"Batch shape: {batch['image'].shape}")
    print(f"Batch dtype: {batch['image'].dtype}")
    print(f"Value range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
    
    # Example without augmentation
    print("\nCreating dataloader WITHOUT augmentation...")
    iterator_no_aug, _ = create_tfrecord_iterator(
        tfrecord_pattern,
        batch_size=32,
        augment_fn=None,  # No augmentation
        precomputed_stats=(0.5, 0.2),
        shuffle=False,
    )
    
    batch_no_aug = next(iterator_no_aug)
    print(f"Batch shape: {batch_no_aug['image'].shape}")


if __name__ == "__main__":
    main()

