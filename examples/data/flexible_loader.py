"""
Flexible dataloader example with multiple data types.

Demonstrates:
1. Custom parsers for multiple data types (depth maps, segmentation, labels)
2. Per-field preprocessing configuration using callables
3. Custom augmentation functions
4. Composable augmentations from the library

Usage:
    python examples/flexible_loader.py
"""
from __future__ import annotations

import tensorflow as tf
from functools import partial
from beagle.dataset import (
    create_iterator,
    apply_zscore_norm,
    apply_histogram_equalization,
    apply_minmax_norm,
)
from beagle.augmentations import (
    compose,
    random_flip_left_right,
    random_flip_up_down,
    random_rotate_90,
    random_brightness,
)


# Example 1: Depth map loader (no 0-255 assumption!)
def example_depth_map_loader():
    """
    Load depth maps that are NOT in 0-255 range.
    
    Shows how to handle arbitrary numeric data.
    """
    print("=" * 60)
    print("Example 1: Depth Map Loader")
    print("=" * 60)
    
    # Custom parser for depth maps
    def parse_depth(example_proto: tf.Tensor) -> dict[str, tf.Tensor]:
        feature_dict = {
            'depth': tf.io.FixedLenFeature([], tf.string),
        }
        parsed = tf.io.parse_single_example(example_proto, feature_dict)
        
        # Decode depth map (could be float32 values in arbitrary range)
        depth = tf.io.decode_raw(parsed['depth'], tf.float32)
        depth = tf.reshape(depth, [512, 512, 1])
        
        # Replace NaN with 0
        depth = tf.where(tf.math.is_nan(depth), 0.0, depth)
        
        return {'depth': depth}
    
    # Configure preprocessing: use histogram equalization for depth
    field_configs = {
        'depth': apply_histogram_equalization
    }
    
    # Custom augmentation: works directly on depth values
    def custom_depth_noise(data_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        """Add multiplicative noise to depth map."""
        depth = data_dict['depth']
        noise = tf.random.uniform(shape=tf.shape(depth), minval=0.95, maxval=1.05)
        data_dict['depth'] = depth * noise
        return data_dict
    
    # Compose augmentations (library + custom)
    augment_fn = compose(
        random_flip_left_right(fields=['depth']),
        random_flip_up_down(fields=['depth']),
        random_rotate_90(fields=['depth']),
        custom_depth_noise,  # Our custom function!
    )
    
    print("\nConfiguration:")
    print(f"  - Field: depth (histogram equalization)")
    print(f"  - Augmentations: flips + rotations + custom noise")
    print(f"  - No 0-255 assumption - works with raw depth values!\n")
    
    # Note: This would work if you had actual depth TFRecords
    # iterator, n_batches = create_iterator(
    #     tfrecord_pattern="data/depth/*.tfrecord",
    #     parser=parse_depth,
    #     field_configs=field_configs,
    #     augment_fn=augment_fn,
    #     batch_size=16,
    # )


# Example 2: Segmentation with image + mask
def example_segmentation_loader():
    """
    Load images with segmentation masks.
    
    Shows how to handle multiple data types with different preprocessing.
    """
    print("=" * 60)
    print("Example 2: Segmentation (Image + Mask)")
    print("=" * 60)
    
    # Custom parser for image + mask
    def parse_segmentation(example_proto: tf.Tensor) -> dict[str, tf.Tensor]:
        feature_dict = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'mask': tf.io.FixedLenFeature([], tf.string),
        }
        parsed = tf.io.parse_single_example(example_proto, feature_dict)
        
        # Decode image (0-255 -> 0-1)
        img = tf.io.decode_image(parsed['image'], channels=3)
        img = tf.cast(img, tf.float32) / 255.0
        
        # Decode segmentation mask (integer class IDs)
        mask = tf.io.decode_png(parsed['mask'], channels=1)
        mask = tf.cast(mask, tf.int32)
        
        return {'image': img, 'mask': mask}
    
    # Configure fields: z-score normalization for image only
    # Mask is not in field_configs, so it passes through unchanged
    field_configs = {
        'image': partial(apply_zscore_norm, mean=0.5, std=0.25, epsilon=1e-8)
    }
    
    # Geometric augmentations apply to BOTH image and mask
    # (using the same random seed for consistency)
    augment_fn = compose(
        random_flip_left_right(),  # Applies to both by default
        random_flip_up_down(),
        random_rotate_90(),
        random_brightness(0.2, field='image'),  # Only to image!
    )
    
    print("\nConfiguration:")
    print(f"  - Fields:")
    print(f"    * image: RGB, z-score normalized")
    print(f"    * mask: integer segmentation, passed through unchanged")
    print(f"  - Augmentations:")
    print(f"    * Geometric (both): flips + rotations")
    print(f"    * Color (image only): brightness\n")


# Example 3: Multi-task learning (image + vector + label)
def example_multitask_loader():
    """
    Load data for multi-task learning.
    
    Shows how to handle: images, regression targets (vectors), and labels.
    """
    print("=" * 60)
    print("Example 3: Multi-Task (Image + Regression + Classification)")
    print("=" * 60)
    
    # Custom parser for multi-task data
    def parse_multitask(example_proto: tf.Tensor) -> dict[str, tf.Tensor]:
        feature_dict = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'bbox': tf.io.FixedLenFeature([4], tf.float32),  # [x, y, w, h]
            'class_id': tf.io.FixedLenFeature([], tf.int64),
            'age': tf.io.FixedLenFeature([], tf.float32),  # Regression target
        }
        parsed = tf.io.parse_single_example(example_proto, feature_dict)
        
        # Decode image
        img = tf.io.decode_image(parsed['image'], channels=3)
        img = tf.cast(img, tf.float32) / 255.0
        
        return {
            'image': img,
            'bbox': parsed['bbox'],
            'class_id': parsed['class_id'],
            'age': parsed['age'],
        }
    
    # Configure each field appropriately
    # Only include fields that need preprocessing
    field_configs = {
        'image': partial(apply_zscore_norm, mean=0.5, std=0.25, epsilon=1e-8),
        'bbox': partial(apply_minmax_norm, min_val=0.0, max_val=1.0, epsilon=1e-8),
        'age': partial(apply_zscore_norm, mean=45.0, std=15.0, epsilon=1e-8),
        # class_id is NOT in config, so it passes through unchanged (integer label)
    }
    
    # Augment only the image (labels/bbox/age unchanged)
    augment_fn = compose(
        random_flip_left_right(fields=['image']),
        random_brightness(0.15, field='image'),
    )
    
    print("\nConfiguration:")
    print(f"  - Fields:")
    print(f"    * image: RGB, z-score normalized")
    print(f"    * bbox: 4D vector, min-max normalized")
    print(f"    * class_id: integer label, passed through")
    print(f"    * age: scalar, z-score normalized")
    print(f"  - Augmentations: only applied to image\n")


# Example 4: Custom augmentation (completely custom)
def example_custom_augmentation():
    """
    Write completely custom augmentations.
    
    Shows the flexibility of the system.
    """
    print("=" * 60)
    print("Example 4: Custom Augmentations")
    print("=" * 60)
    
    # Custom augmentation 1: Cutout/random erasing
    def random_cutout(data_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        """Apply random cutout to image."""
        img = data_dict['image']
        h, w = tf.shape(img)[0], tf.shape(img)[1]
        
        # Random box size (10-30% of image)
        box_h = tf.cast(tf.cast(h, tf.float32) * 0.2, tf.int32)
        box_w = tf.cast(tf.cast(w, tf.float32) * 0.2, tf.int32)
        
        # Random position
        y = tf.random.uniform([], 0, h - box_h, dtype=tf.int32)
        x = tf.random.uniform([], 0, w - box_w, dtype=tf.int32)
        
        # Apply cutout by setting region to 0
        updates = tf.zeros([box_h, box_w, tf.shape(img)[2]], dtype=img.dtype)
        
        data_dict['image'] = img
        return data_dict
    
    # Custom augmentation 2: Conditional augmentation
    def conditional_augment(data_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        """Apply augmentation based on label."""
        if 'label' in data_dict:
            # Apply stronger augmentation for certain classes
            label = data_dict['label']
            strength = tf.cond(
                label > 5,
                lambda: 0.3,  # Strong augmentation
                lambda: 0.1,  # Mild augmentation
            )
            data_dict['image'] = tf.image.random_brightness(data_dict['image'], strength)
        
        return data_dict
    
    # Custom augmentation 3: Apply transformation to specific field
    def normalize_to_range(min_val: float, max_val: float, field: str = 'image'):
        """Normalize field to specific range."""
        def augment(data_dict: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
            if field in data_dict:
                x = data_dict[field]
                x_min = tf.reduce_min(x)
                x_max = tf.reduce_max(x)
                # Normalize to [0, 1] then scale to [min_val, max_val]
                x_norm = (x - x_min) / (x_max - x_min + 1e-8)
                data_dict[field] = x_norm * (max_val - min_val) + min_val
            return data_dict
        return augment
    
    # Compose everything
    augment_fn = compose(
        random_flip_left_right(fields=['image']),
        random_cutout,
        conditional_augment,
        normalize_to_range(-1.0, 1.0, field='image'),
    )
    
    print("\nCustom augmentations:")
    print(f"  1. Random cutout (erasing)")
    print(f"  2. Conditional augmentation (based on label)")
    print(f"  3. Normalize to [-1, 1] range")
    print(f"\nEasy to compose with library functions!")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("FLEXIBLE DATALOADER EXAMPLES")
    print("=" * 60 + "\n")
    
    example_depth_map_loader()
    print()
    
    example_segmentation_loader()
    print()
    
    example_multitask_loader()
    print()
    
    example_custom_augmentation()
    print()
    
    print("=" * 60)
    print("KEY FEATURES")
    print("=" * 60)
    print("""
✅ No assumptions about data ranges (works with any numeric data)
✅ Per-field preprocessing with simple callables
✅ Support for multiple data types:
   - Images (normalized with various methods)
   - Segmentation masks (pass through unchanged)
   - Labels (pass through unchanged)
   - Vectors (normalized)
   - Raw data (pass through unchanged)
✅ Easy to write custom preprocessing functions
✅ Easy to write custom augmentations
✅ Composable augmentations (library + custom)
✅ Efficient TensorFlow pipeline (parallel processing)
    """)
    
    print("\nNext steps:")
    print("  1. Update your parser to return dict with multiple fields")
    print("  2. Create field_configs dict with preprocessing callables")
    print("  3. Use functools.partial for functions that need parameters")
    print("  4. Write custom augmentations or use library functions")
    print("  5. Compose augmentations with compose()")
    print("  6. Pass to create_iterator()")
    print("\nSee PREPROCESSING_API_UPDATE.md for migration guide!")


if __name__ == "__main__":
    main()
