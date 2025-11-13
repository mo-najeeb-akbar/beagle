"""
Example: TFRecord to JAX iterator with TensorFlow augmentations.

This demonstrates how to use the beagle library to create a data pipeline
that loads TFRecords, applies TensorFlow augmentations, and yields JAX arrays.
"""
from __future__ import annotations

import sys
import tensorflow as tf

from beagle.dataset import create_tfrecord_iterator


def main() -> None:
    """Run example dataloader."""
    if len(sys.argv) < 2:
        print("Usage: python tfrecord_to_jax_example.py <tfrecord_pattern>")
        print("Example: python tfrecord_to_jax_example.py 'data/*.tfrecord'")
        sys.exit(1)
    
    tfrecord_pattern = sys.argv[1]
    
    # Create TensorFlow augmentation function (efficient!)
    def augment_fn(data_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        """TensorFlow-based augmentation applied in the data pipeline."""
        img = data_dict['image']
        
        # Random flips
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        
        # Random brightness/contrast
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        
        # Random rotation (90 degree increments)
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        img = tf.image.rot90(img, k=k)
        
        data_dict['image'] = img
        return data_dict
    
    # Create iterator with augmentation
    print("Creating dataloader with TensorFlow augmentations...")
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

