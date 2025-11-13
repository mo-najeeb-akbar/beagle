"""Example: Visualizing datasets with beagle.visualization.

Demonstrates how to use visualization utilities for data inspection.
Run with: make run CMD='python examples/dataset_viz_example.py'
"""

from __future__ import annotations

import numpy as np
from beagle.visualization import (
    visualize_batch,
    inspect_iterator,
    plot_image_grid,
    plot_comparison,
    save_figure,
)


def example_visualize_single_batch():
    """Example 1: Visualize a single batch from iterator."""
    print("Example 1: Single batch visualization")
    print("=" * 50)
    
    # Simulate a data batch
    batch = {
        'image': np.random.rand(8, 64, 64, 1),
        'mask': np.random.randint(0, 2, (8, 64, 64, 1)),
    }
    
    # Visualize all keys
    visualize_batch(batch, num_samples=4)
    
    # Visualize specific keys
    visualize_batch(batch, keys=['image'], num_samples=6)


def example_inspect_multiple_batches():
    """Example 2: Inspect multiple batches from iterator."""
    print("\nExample 2: Multiple batch inspection")
    print("=" * 50)
    
    # Simulate an iterator
    def fake_iterator():
        for _ in range(3):
            yield {
                'image': np.random.rand(4, 32, 32, 3),
                'label': np.random.randint(0, 10, 4)
            }
    
    # Inspect 3 batches
    inspect_iterator(
        fake_iterator(),
        num_batches=3,
        keys=['image'],
        num_samples=4,
        output_dir='/data/inspection'
    )


def example_custom_visualization():
    """Example 3: Custom visualizations with core plotting functions."""
    print("\nExample 3: Custom visualization")
    print("=" * 50)
    
    # Create some data
    originals = np.random.rand(6, 64, 64, 1)
    augmented = originals + np.random.randn(6, 64, 64, 1) * 0.1
    predictions = originals + np.random.randn(6, 64, 64, 1) * 0.2
    
    # Compare multiple versions
    fig = plot_comparison(
        {
            'Original': originals,
            'Augmented': augmented,
            'Predicted': predictions
        },
        num_samples=4,
        suptitle="Data Pipeline Comparison"
    )
    
    save_figure(fig, '/data/comparison.png')
    
    # Grid of images
    fig2 = plot_image_grid(
        originals,
        num_cols=3,
        suptitle="Sample Grid"
    )
    
    save_figure(fig2, '/data/grid.png')


def example_with_tfrecord_iterator():
    """Example 4: Real dataset visualization."""
    print("\nExample 4: Real TFRecord dataset")
    print("=" * 50)
    
    # This example assumes you have tfrecords
    try:
        from beagle.dataset import get_jax_iterator
        
        # Create iterator
        iterator = get_jax_iterator(
            '/data/input/*.tfrecord',
            batch_size=8,
            shuffle=False
        )
        
        # Inspect first 3 batches
        inspect_iterator(
            iterator,
            num_batches=3,
            num_samples=4,
            output_dir='/data/dataset_inspection'
        )
        
    except Exception as e:
        print(f"Skipping TFRecord example: {e}")


if __name__ == "__main__":
    print("🎨 Beagle Visualization Examples\n")
    
    example_visualize_single_batch()
    example_inspect_multiple_batches()
    example_custom_visualization()
    example_with_tfrecord_iterator()
    
    print("\n✅ All examples complete!")

