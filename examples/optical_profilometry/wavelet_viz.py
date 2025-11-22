from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import matplotlib.pyplot as plt
import numpy as np

from data_loader import create_polymer_iterator


CONFIG = {
    "learning_rate": 0.001,
    "num_epochs": 50,  # Reduced for testing
    "batch_size": 32,
    "base_features": 48,
    "latent_dim": 256,
    "crop_size": 256,
    "crop_overlap": 192,
}


def tile_wavelet_channels(wavelet_batch: np.ndarray) -> np.ndarray:
    """Tile 4 wavelet channels into a 2x2 grid for visualization.
    
    Args:
        wavelet_batch: Shape (B, H, W, 4) - batch of wavelet coefficients
        
    Returns:
        Tiled wavelet images, shape (B, 2*H, 2*W, 1)
    """
    batch_size, h, w, _ = wavelet_batch.shape
    tiled = np.zeros((batch_size, h * 2, w * 2, 1))
    
    # Arrange channels in 2x2 grid: [LL, LH]
    #                                [HL, HH]
    tiled[:, :h, :w, 0] = wavelet_batch[:, :, :, 0]    # Top-left: LL
    tiled[:, :h, w:, 0] = wavelet_batch[:, :, :, 1]    # Top-right: LH
    tiled[:, h:, :w, 0] = wavelet_batch[:, :, :, 2]    # Bottom-left: HL
    tiled[:, h:, w:, 0] = wavelet_batch[:, :, :, 3]    # Bottom-right: HH
    
    return tiled


def visualize_wavelet_transform(
    original: np.ndarray, 
    wavelets: np.ndarray, 
    reconstructed: np.ndarray
) -> None:
    """Visualize original, wavelet transform, and reconstructed images.
    
    Args:
        original: Shape (B, H, W, 1) - original images
        wavelets: Shape (B, H/2, W/2, 4) - wavelet coefficients
        reconstructed: Shape (B, H, W, 1) - reconstructed images
    """
    batch_size = original.shape[0]
    
    # Tile wavelets into 2x2 grid
    wavelets_tiled = tile_wavelet_channels(wavelets)
    
    # Create figure with 3 columns (original, wavelets, reconstructed)
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
    
    # Handle single image case
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Original
        axes[i, 0].imshow(original[i, :, :, 0], cmap='gray')
        axes[i, 0].set_title(f'Original {i}')
        axes[i, 0].axis('off')
        
        # Wavelets (tiled 2x2)
        axes[i, 1].imshow(wavelets_tiled[i, :, :, 0], cmap='gray')
        axes[i, 1].set_title(f'Wavelets {i} (LL|LH, HL|HH)')
        axes[i, 1].axis('off')
        
        # Reconstructed
        axes[i, 2].imshow(reconstructed[i, :, :, 0], cmap='gray')
        axes[i, 2].set_title(f'Reconstructed {i}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('wavelet_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to wavelet_visualization.png")
    # plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_wavelet_vae.py /path/to/polymer_tfrecords [--compute-stats]")
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    
    # Optional: just compute and display statistics
    if '--compute-stats' in sys.argv:
        mean, std, n_imgs = compute_polymer_stats(data_dir)
        print(f"Dataset: {n_imgs} images")
        print(f"Mean: {mean:.6f}")
        print(f"Std:  {std:.6f}")
        return
    
    # Initialize model
    from beagle.network.wavelets import HaarWaveletConv, HaarWaveletConvTranspose
    
    init_key = random.key(42)

    forward_conv = HaarWaveletConv()
    forward_conv_params = forward_conv.init(init_key, jnp.ones((1, 256, 256, 1)))
    backward_conv = HaarWaveletConvTranspose()
    backward_conv_params = backward_conv.init(init_key, jnp.ones((1, 128, 128, 4)))
    
    
    # Load data using shared module
    print("Loading polymer dataset...")
    iterator, batches_per_epoch = create_polymer_iterator(
        data_dir=data_dir,
        batch_size=CONFIG['batch_size'],
        crop_size=CONFIG['crop_size'],
        stride=CONFIG['crop_overlap'],
        shuffle=True,
        augment=True
    )
    
    batch = next(iterator)
    images = batch['depth']
    wavelets = forward_conv.apply(forward_conv_params, images)
    reconstructed = backward_conv.apply(backward_conv_params, wavelets)
    print(images.shape, wavelets.shape, reconstructed.shape)

    # Visualize the wavelets and reconstructed images
    visualize_wavelet_transform(
        np.array(images), 
        np.array(wavelets), 
        np.array(reconstructed)
    )



if __name__ == "__main__":
    main()
