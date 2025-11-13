"""Dataset visualization utilities (inspect data pipelines)."""

from __future__ import annotations

from typing import Callable, Iterator, Any
import numpy as np

from beagle.visualization.plotting import plot_image_grid, plot_images, save_figure


def visualize_batch(
    batch: dict[str, np.ndarray] | np.ndarray,
    keys: list[str] | None = None,
    num_samples: int = 4,
    save_path: str | None = None
) -> None:
    """Visualize a batch from dataset iterator (side effect).
    
    Args:
        batch: Batch dict or array
        keys: Keys to visualize if batch is dict (None = all image-like keys)
        num_samples: Number of samples to show
        save_path: Optional path to save figure
        
    Example:
        >>> ds = get_jax_iterator(...)
        >>> batch = next(ds)
        >>> visualize_batch(batch, keys=['image', 'mask'], num_samples=4)
    """
    # Handle array vs dict
    if isinstance(batch, dict):
        # Auto-detect image-like keys if not provided
        if keys is None:
            keys = [k for k, v in batch.items() 
                   if isinstance(v, (np.ndarray, Any)) and len(v.shape) >= 3]
        
        # Create comparison of different keys
        images_dict = {}
        for key in keys:
            arr = np.array(batch[key])[:num_samples]
            images_dict[key] = arr
        
        from beagle.visualization.plotting import plot_comparison
        fig = plot_comparison(images_dict, num_samples=num_samples)
    else:
        # Simple array
        arr = np.array(batch)[:num_samples]
        fig = plot_image_grid(arr, num_cols=4)
    
    if save_path is not None:
        save_figure(fig, save_path, close=True)
    else:
        import matplotlib.pyplot as plt
        plt.show()
        plt.close(fig)


def inspect_iterator(
    iterator: Iterator[Any],
    num_batches: int = 1,
    keys: list[str] | None = None,
    num_samples: int = 4,
    output_dir: str | None = None
) -> None:
    """Inspect multiple batches from an iterator (side effect).
    
    Useful for debugging data pipelines and checking augmentations.
    
    Args:
        iterator: Data iterator
        num_batches: Number of batches to visualize
        keys: Keys to visualize (None = auto-detect)
        num_samples: Samples per batch
        output_dir: Directory to save plots (None = show only)
        
    Example:
        >>> ds = get_jax_iterator('/data/*.tfrecord', shuffle=True)
        >>> inspect_iterator(ds, num_batches=3, output_dir='/tmp/inspection')
    """
    for batch_idx in range(num_batches):
        batch = next(iterator)
        
        save_path = None
        if output_dir is not None:
            from pathlib import Path
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            save_path = f"{output_dir}/batch_{batch_idx:03d}.png"
        
        print(f"\n📊 Batch {batch_idx + 1}/{num_batches}")
        
        # Print shape info
        if isinstance(batch, dict):
            for k, v in batch.items():
                arr = np.array(v)
                print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}, "
                      f"range=[{arr.min():.3f}, {arr.max():.3f}]")
        else:
            arr = np.array(batch)
            print(f"  shape={arr.shape}, dtype={arr.dtype}, "
                  f"range=[{arr.min():.3f}, {arr.max():.3f}]")
        
        visualize_batch(batch, keys=keys, num_samples=num_samples, save_path=save_path)


def preview_augmentations(
    original_batch: dict[str, np.ndarray] | np.ndarray,
    augmentation_fn: Callable[[Any], Any],
    num_variants: int = 4,
    save_path: str | None = None
) -> None:
    """Preview effect of augmentations on a batch (side effect).
    
    Args:
        original_batch: Original batch
        augmentation_fn: Augmentation function to apply
        num_variants: Number of augmented variants to show
        save_path: Optional save path
        
    Example:
        >>> from beagle.augmentations import random_flip, random_brightness
        >>> batch = next(iterator)
        >>> aug_fn = lambda x: random_brightness(random_flip(x, key), key)
        >>> preview_augmentations(batch, aug_fn, num_variants=4)
    """
    from beagle.visualization.plotting import plot_comparison
    
    # Get one sample
    if isinstance(original_batch, dict):
        # Assume 'image' key exists
        key = 'image' if 'image' in original_batch else list(original_batch.keys())[0]
        original = np.array(original_batch[key])[0:1]
    else:
        original = np.array(original_batch)[0:1]
    
    # Create variants
    variants = {'Original': original}
    
    for i in range(num_variants):
        # Apply augmentation
        if isinstance(original_batch, dict):
            aug_batch = augmentation_fn({key: original})
            variant = np.array(aug_batch[key])
        else:
            variant = np.array(augmentation_fn(original))
        
        variants[f'Variant {i+1}'] = variant
    
    fig = plot_comparison(variants, num_samples=1, 
                         suptitle="Augmentation Preview")
    
    if save_path is not None:
        save_figure(fig, save_path, close=True)
    else:
        import matplotlib.pyplot as plt
        plt.show()
        plt.close(fig)

