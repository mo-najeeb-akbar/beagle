"""
Polymer depth map dataloader example.

Demonstrates:
1. Loading TFRecords with custom parser
2. Overlapping crops for large images
3. Flexible field configuration (no 0-255 assumption!)
4. Custom augmentations with compose()

Usage:
    # Compute statistics
    python examples/polymer_loader.py ~/data/polymer_tfrecords --compute-stats
    
    # Run training loop example
    python examples/polymer_loader.py ~/data/polymer_tfrecords
"""
from __future__ import annotations

import sys
import glob
from pathlib import Path

import tensorflow as tf

from beagle.dataset import (
    load_tfr_dict,
    count_tfrecord_samples,
    create_iterator,
    FieldConfig,
    FieldType,
    compute_field_stats,
)
from beagle.augmentations import (
    compose,
    random_flip_left_right,
    random_flip_up_down,
    random_rotate_90,
    random_brightness,
)


def make_polymer_parser(
    feature_dict: dict[str, tf.io.FixedLenFeature],
    shape_dict: dict[str, list[int]],
):
    """Create parser for polymer TFRecords (pure)."""
    def parse(example_proto: tf.Tensor) -> dict[str, tf.Tensor]:
        parsed = tf.io.parse_single_example(example_proto, feature_dict)
        # feature_dict already handles deserialization, so parsed['surface'] is already float32
        surface = tf.reshape(parsed['surface'], shape_dict['surface'] + [1])
        surface = tf.where(tf.math.is_nan(surface), 0.0, surface)
        # NOTE: We call it 'depth' to be clear it's not a 0-255 image!
        return {'depth': surface}
    
    return parse


def create_augment_fn():
    """
    Create augmentation function for polymer depth data.
    
    Uses the new compose() API for easy customization.
    Works directly on depth values (no 0-255 assumption!).
    """
    return compose(
        random_flip_left_right(fields=['depth']),
        random_flip_up_down(fields=['depth']),
        random_rotate_90(fields=['depth']),
        random_brightness(0.1, field='depth'),  # Additive noise for depth
    )


def main() -> None:
    """Run polymer dataloader example."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    tfrecord_pattern = str(data_dir / "*.tfrecord")
    json_path = str(data_dir / "polymer.json")
    compute_stats = '--compute-stats' in sys.argv
    
    # Load feature dictionary and create parser
    feature_dict, shape_dict = load_tfr_dict(json_path)
    parser = make_polymer_parser(feature_dict, shape_dict)
    
    files = sorted(glob.glob(tfrecord_pattern))
    
    if compute_stats:
        # Compute and display statistics using new API
        print(f"Computing statistics from {len(files)} files...")
        mean, std = compute_field_stats(files, parser=parser, field_name='depth')
        n_imgs = count_tfrecord_samples(files)
        
        print(f"\nDataset: {n_imgs} images")
        print(f"Mean: {mean:.6f}")
        print(f"Std:  {std:.6f}")
        print(f"\nUse with FieldConfig:")
        print(f"  stats=({mean:.6f}, {std:.6f})")
        return
    
    # Configure field: depth map (not 0-255 image!)
    field_configs = {
        'depth': FieldConfig(
            name='depth',
            field_type=FieldType.IMAGE,  # Numeric data to standardize
            standardize=True,
            stats=None,  # Will be computed automatically
        )
    }
    
    # Create augmentation function
    aug_fn = create_augment_fn()
    
    print("Creating polymer dataloader...")
    print("  - Field: depth map (standardized, no 0-255 assumption)")
    print("  - Augmentations: flips + rotations + brightness")
    print("  - Crops: 256x256 with stride 192")
    
    # Get image shape from shape_dict for efficient crop counting
    img_height, img_width = shape_dict['surface']
    
    iterator, n_batches = create_iterator(
        tfrecord_pattern=tfrecord_pattern,
        parser=parser,
        crop_size=256,
        stride=192,
        image_shape=(img_height, img_width),  # For fast crop count
        batch_size=32,
        field_configs=field_configs,  # New flexible API!
        augment_fn=aug_fn,
        shuffle=True,
    )
    
    print(f"Ready! {n_batches} batches/epoch\n")
    
    # Demo: fetch a few batches
    print("Fetching 3 batches...")
    for i in range(3):
        batch = next(iterator)
        # Note: field is now 'depth' not 'image'
        print(f"Batch {i+1}: shape={batch['depth'].shape}, "
              f"range=[{batch['depth'].min():.2f}, {batch['depth'].max():.2f}]")
    
    print("\n✅ Dataloader working!")


if __name__ == "__main__":
    main()
