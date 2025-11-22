"""Polymer depth map dataloader inspector.

Demonstrates dataset inspection and statistics computation.

Usage:
    # Compute statistics
    python examples/polymer_loader.py ~/data/polymer_tfrecords --compute-stats
    
    # Inspect batches
    python examples/polymer_loader.py ~/data/polymer_tfrecords
"""
from __future__ import annotations

import sys
from pathlib import Path

# Import from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from polymer_data import create_polymer_iterator, compute_polymer_stats


def main() -> None:
    """Run polymer dataloader inspector."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    
    if '--compute-stats' in sys.argv:
        print("Computing statistics...")
        mean, std, n_imgs = compute_polymer_stats(data_dir)
        
        print(f"\nDataset: {n_imgs} images")
        print(f"Mean: {mean:.6f}")
        print(f"Std:  {std:.6f}")
        print(f"\nUse with preprocessing:")
        print(f"  from functools import partial")
        print(f"  from beagle.dataset import apply_zscore_norm")
        print(f"  field_configs = {{")
        print(f"      'depth': partial(apply_zscore_norm, mean={mean:.6f}, std={std:.6f}, epsilon=1e-8)")
        print(f"  }}")
        return
    
    # Create iterator using shared module
    print("Creating polymer dataloader...")
    print("  - Field: depth map (standardized, no 0-255 assumption)")
    print("  - Augmentations: flips + rotations + brightness")
    print("  - Crops: 256x256 with stride 192")
    
    iterator, n_batches, img_shape = create_polymer_iterator(
        data_dir=data_dir,
        batch_size=32,
        crop_size=256,
        stride=192,
        shuffle=True,
        augment=True
    )
    
    print(f"Ready! {n_batches} batches/epoch\n")
    
    # Demo: fetch a few batches
    print("Fetching 3 batches...")
    for i in range(3):
        batch = next(iterator)
        print(f"Batch {i+1}: shape={batch['depth'].shape}, "
              f"range=[{batch['depth'].min():.2f}, {batch['depth'].max():.2f}]")
    
    print("\n✅ Dataloader working!")


if __name__ == "__main__":
    main()
