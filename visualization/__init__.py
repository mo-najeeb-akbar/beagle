"""Visualization utilities for JAX arrays, images, and tensors.

Functional visualization library for:
- Dataset inspection
- Model outputs
- Training progress
- Tensor visualization

All plotting functions are pure side effects - they take numpy arrays
and produce visualizations without modifying inputs.
"""

from beagle.visualization.plotting import (
    plot_images,
    plot_image_grid,
    plot_comparison,
    plot_tensor_stats,
    save_figure,
)

from beagle.visualization.callbacks import (
    create_viz_callback,
    create_simple_reconstruction_callback,
    VizCallback,
    VizConfig,
)

from beagle.visualization.dataset import (
    visualize_batch,
    inspect_iterator,
    preview_augmentations,
)

__all__ = [
    # Core plotting
    "plot_images",
    "plot_image_grid",
    "plot_comparison",
    "plot_tensor_stats",
    "save_figure",
    # Callbacks
    "create_viz_callback",
    "create_simple_reconstruction_callback",
    "VizCallback",
    "VizConfig",
    # Dataset inspection
    "visualize_batch",
    "inspect_iterator",
    "preview_augmentations",
]

