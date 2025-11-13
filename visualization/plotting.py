"""Core plotting functions (pure side effects - don't modify inputs)."""

from __future__ import annotations

from typing import Sequence
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib() -> None:
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "Matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def plot_images(
    images: np.ndarray | Sequence[np.ndarray],
    titles: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
    cmap: str | None = 'gray',
    suptitle: str | None = None
) -> matplotlib.figure.Figure:
    """Plot images in a horizontal row (pure side effect).
    
    Args:
        images: Array of shape (N, H, W) or (N, H, W, C), or list of arrays
        titles: Optional title for each image
        figsize: Figure size (default: auto-scale based on number of images)
        cmap: Colormap (None for RGB, 'gray' for grayscale)
        suptitle: Overall title for the figure
        
    Returns:
        Matplotlib figure object
    """
    _check_matplotlib()
    
    # Convert to list if needed
    if isinstance(images, np.ndarray):
        image_list = [images[i] for i in range(len(images))]
    else:
        image_list = list(images)
    
    num_images = len(image_list)
    
    if figsize is None:
        figsize = (4 * num_images, 4)
    
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    
    # Handle single image case
    if num_images == 1:
        axes = [axes]
    
    for i, (ax, img) in enumerate(zip(axes, image_list)):
        # Handle channel dimension
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
        
        # Auto-select colormap
        plot_cmap = cmap if img.ndim == 2 else None
        
        ax.imshow(img, cmap=plot_cmap)
        ax.axis('off')
        
        if titles is not None and i < len(titles):
            ax.set_title(titles[i])
    
    if suptitle is not None:
        fig.suptitle(suptitle)
    
    plt.tight_layout()
    return fig


def plot_image_grid(
    images: np.ndarray,
    num_cols: int = 4,
    titles: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
    cmap: str | None = 'gray',
    suptitle: str | None = None
) -> matplotlib.figure.Figure:
    """Plot images in a grid (pure side effect).
    
    Args:
        images: Array of shape (N, H, W) or (N, H, W, C)
        num_cols: Number of columns in grid
        titles: Optional title for each image
        figsize: Figure size (default: auto-scale)
        cmap: Colormap for grayscale images
        suptitle: Overall title for the figure
        
    Returns:
        Matplotlib figure object
    """
    _check_matplotlib()
    
    num_images = len(images)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    if figsize is None:
        figsize = (3 * num_cols, 3 * num_rows)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Flatten axes array for easier indexing
    if num_rows == 1 and num_cols == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for idx in range(num_images):
        ax = axes_flat[idx]
        img = images[idx]
        
        # Handle channel dimension
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
        
        plot_cmap = cmap if img.ndim == 2 else None
        
        ax.imshow(img, cmap=plot_cmap)
        ax.axis('off')
        
        if titles is not None and idx < len(titles):
            ax.set_title(titles[idx], fontsize=9)
    
    # Hide empty subplots
    for idx in range(num_images, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=14)
    
    plt.tight_layout()
    return fig


def plot_comparison(
    images_dict: dict[str, np.ndarray],
    num_samples: int = 4,
    figsize: tuple[float, float] | None = None,
    cmap: str | None = 'gray',
    suptitle: str | None = None
) -> matplotlib.figure.Figure:
    """Plot side-by-side comparison of multiple image sets (pure side effect).
    
    Args:
        images_dict: Dict mapping labels to image arrays of shape (N, H, W, C)
        num_samples: Number of samples to show
        figsize: Figure size (default: auto-scale)
        cmap: Colormap for grayscale images
        suptitle: Overall title
        
    Returns:
        Matplotlib figure object
        
    Example:
        >>> fig = plot_comparison({
        ...     'Original': originals,
        ...     'Prediction': predictions,
        ...     'Ground Truth': targets
        ... }, num_samples=3)
    """
    _check_matplotlib()
    
    labels = list(images_dict.keys())
    num_cols = len(labels)
    
    # Get actual number of samples (limited by smallest array)
    min_samples = min(len(arr) for arr in images_dict.values())
    num_samples = min(num_samples, min_samples)
    
    if figsize is None:
        figsize = (4 * num_cols, 3 * num_samples)
    
    fig, axes = plt.subplots(num_samples, num_cols, figsize=figsize, squeeze=False)
    
    for col_idx, label in enumerate(labels):
        images = images_dict[label]
        
        for row_idx in range(num_samples):
            ax = axes[row_idx, col_idx]
            img = images[row_idx]
            
            # Handle channel dimension
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img.squeeze(-1)
            
            plot_cmap = cmap if img.ndim == 2 else None
            
            ax.imshow(img, cmap=plot_cmap)
            ax.axis('off')
            
            # Add column labels on first row
            if row_idx == 0:
                ax.set_title(label)
    
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=14)
    
    plt.tight_layout()
    return fig


def plot_tensor_stats(
    tensors: dict[str, np.ndarray],
    figsize: tuple[float, float] = (12, 4)
) -> matplotlib.figure.Figure:
    """Plot statistical summaries of tensors (pure side effect).
    
    Useful for visualizing activations, gradients, or latent representations.
    
    Args:
        tensors: Dict mapping names to arrays of any shape
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    _check_matplotlib()
    
    num_tensors = len(tensors)
    fig, axes = plt.subplots(1, num_tensors, figsize=figsize)
    
    if num_tensors == 1:
        axes = [axes]
    
    for ax, (name, tensor) in zip(axes, tensors.items()):
        flat = tensor.flatten()
        
        ax.hist(flat, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(flat.mean(), color='red', linestyle='--', 
                   label=f'μ={flat.mean():.3f}')
        ax.axvline(flat.mean() + flat.std(), color='orange', 
                   linestyle='--', alpha=0.7, label=f'σ={flat.std():.3f}')
        ax.axvline(flat.mean() - flat.std(), color='orange', 
                   linestyle='--', alpha=0.7)
        
        ax.set_title(f'{name}\nshape: {tensor.shape}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def save_figure(
    fig: matplotlib.figure.Figure,
    path: str | Path,
    dpi: int = 150,
    close: bool = True
) -> None:
    """Save matplotlib figure to disk (side effect).
    
    Args:
        fig: Matplotlib figure
        path: Output path
        dpi: Resolution
        close: Whether to close figure after saving
    """
    _check_matplotlib()
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"💾 Saved figure to: {path}")
    
    if close:
        plt.close(fig)

