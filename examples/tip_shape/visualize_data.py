"""
Visualize root tip data loader with image and mask support.

Shows how geometric augmentations preserve mask alignment with images.

Usage:
    make run CMD='python examples/tip_shape/visualize_data.py'
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import sys

from data_loader import create_root_tip_iterator


def visualize_batch(
    batch: dict[str, np.ndarray],
    num_samples: int = 8,
    title: str = "Root Tip Data"
) -> None:
    """
    Visualize a batch of images with masks (pure visualization).
    
    Args:
        batch: Dictionary with 'image' and 'mask' tensors (JAX or numpy arrays)
        num_samples: Number of samples to show
        title: Figure title
    """
    images = np.array(batch['image'])
    masks = np.array(batch['mask'])
    
    batch_size = min(num_samples, images.shape[0])
    
    fig, axes = plt.subplots(2, batch_size, figsize=(2.5 * batch_size, 5))
    
    if batch_size == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(batch_size):
        # Get image and mask
        img = images[i, :, :, 0]
        mask = masks[i, :, :, 0]
        
        # Show image
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')
        
        # Show mask
        axes[1, i].imshow(mask, cmap='viridis', vmin=0, vmax=1)
        axes[1, i].set_title(f'Mask {i+1}')
        axes[1, i].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def visualize_overlay(
    batch: dict[str, np.ndarray],
    num_samples: int = 8,
    alpha: float = 0.4,
    title: str = "Image + Mask Overlay"
) -> None:
    """
    Visualize images with mask overlays (pure visualization).
    
    Args:
        batch: Dictionary with 'image' and 'mask' tensors (JAX or numpy arrays)
        num_samples: Number of samples to show
        alpha: Transparency of mask overlay
        title: Figure title
    """
    images = np.array(batch['image'])
    masks = np.array(batch['mask'])
    
    batch_size = min(num_samples, images.shape[0])
    
    fig, axes = plt.subplots(1, batch_size, figsize=(3 * batch_size, 3))
    
    if batch_size == 1:
        axes = [axes]
    
    for i in range(batch_size):
        # Get image and mask
        img = images[i, :, :, 0]
        mask = masks[i, :, :, 0]
        
        # Normalize image for display (handle z-scored images)
        img_display = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # Show image
        axes[i].imshow(img_display, cmap='gray')
        
        # Overlay mask (only where mask > threshold)
        mask_binary = mask > 0.0
        masked_overlay = np.ma.masked_where(~mask_binary, mask)
        axes[i].imshow(masked_overlay, cmap='hot', alpha=alpha, vmin=0, vmax=1)
        
        axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')
    
    # Create legend
    gray_patch = mpatches.Patch(color='gray', label='Image')
    red_patch = mpatches.Patch(color='red', alpha=alpha, label='Mask')
    fig.legend(handles=[gray_patch, red_patch], loc='upper right')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def compare_augmented_vs_original(
    data_dir: str | Path,
    batch_size: int = 4,
) -> tuple[plt.Figure, plt.Figure]:
    """
    Compare original data vs augmented data side-by-side.
    
    Args:
        data_dir: Directory containing TFRecords
        batch_size: Number of samples to show
    """
    # Create two iterators
    iter_orig, _ = create_root_tip_iterator(
        data_dir=data_dir,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
    )
    
    iter_aug, _ = create_root_tip_iterator(
        data_dir=data_dir,
        batch_size=batch_size,
        shuffle=False,
        augment=True,
    )
    
    # Get batches
    batch_orig = next(iter_orig)
    batch_aug = next(iter_aug)
    
    # Visualize
    fig1 = visualize_overlay(batch_orig, num_samples=batch_size, 
                            title='Original Data (No Augmentation)')
    
    fig2 = visualize_overlay(batch_aug, num_samples=batch_size,
                            title='Augmented Data (Flips + Rotations)')
    
    plt.savefig('original_vs_augmented.png')
    plt.close()
    
    return fig1, fig2


def main() -> tuple[plt.Figure, plt.Figure]:
    """Main visualization script."""
    data_dir = sys.argv[1]

    
    print("Creating data iterator with augmentation...")
    iterator, batches_per_epoch = create_root_tip_iterator(
        data_dir=data_dir,
        batch_size=8,
        shuffle=True,
        augment=True,
    )
    
    print(f"Batches per epoch: {batches_per_epoch}")
    print("Fetching batch...")
    
    # Get a batch
    batch = next(iterator)
    
    print(f"Image shape: {batch['image'].shape}")
    print(f"Mask shape: {batch['mask'].shape}")
    print(f"Image range: [{np.array(batch['image']).min():.2f}, {np.array(batch['image']).max():.2f}]")
    print(f"Mask range: [{np.array(batch['mask']).min():.2f}, {np.array(batch['mask']).max():.2f}]")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Show images and masks separately
    fig1 = visualize_batch(batch, num_samples=8, title='Root Tip Images and Masks (Augmented)')
    plt.show()
    plt.savefig('images_and_masks_augmented.png')
    # Show overlays
    fig2 = visualize_overlay(batch, num_samples=8, alpha=0.5, 
                            title='Image + Mask Overlay (Augmented)')
    plt.show()
    plt.savefig('image_mask_overlay.png')
    plt.close()
    # Compare with/without augmentation
    print("\nComparing augmented vs original...")
    compare_augmented_vs_original(data_dir, batch_size=4)
    
    print("\nDone! Close the plot windows to exit.")


if __name__ == '__main__':
    main()

