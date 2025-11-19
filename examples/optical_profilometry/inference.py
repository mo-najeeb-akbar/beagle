"""Run inference on polymer dataset using trained wavelet VAE.

Run: make run CMD='python examples/optical_profilometry/inference.py /data/experiments/wavelet_vae_XXX /data/output /data/polymer_tfrecords'
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from beagle.training import TrainState, load_checkpoint, load_config
from beagle.network.wavelet_vae import VAE

from data_loader import create_polymer_iterator_from_saved_stats


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Normalize array to 0-255 uint8 range.
    
    Args:
        array: Input array
        
    Returns:
        Normalized uint8 array
    """
    if array.ndim == 3:
        array = array.squeeze(-1)
    
    vmin, vmax = array.min(), array.max()
    if vmax > vmin:
        array = (array - vmin) / (vmax - vmin)
    return (array * 255).astype(np.uint8)


def create_comparison_image(
    original: np.ndarray,
    reconstruction: np.ndarray,
    error_scale: float = 5.0
) -> np.ndarray:
    """Create side-by-side comparison image.
    
    Args:
        original: Original image
        reconstruction: Reconstructed image
        error_scale: Scale factor for error visualization
        
    Returns:
        Combined image [Original | Reconstruction | Error]
    """
    error = np.abs(original - reconstruction) * error_scale
    
    # Normalize each to uint8
    orig_norm = normalize_to_uint8(original)
    recon_norm = normalize_to_uint8(reconstruction)
    error_norm = normalize_to_uint8(error)
    
    # Concatenate horizontally
    combined = np.concatenate([orig_norm, recon_norm, error_norm], axis=1)
    return combined


def run_inference(
    checkpoint_dir: Path,
    output_dir: Path,
    data_dir: Path,
) -> None:
    """Run inference on entire dataset and save reconstructions.
    
    Args:
        checkpoint_dir: Path to experiment directory with checkpoints
        output_dir: Path to save output images (will be created)
        data_dir: Path to data directory
    """
    # Load config and checkpoint
    config = load_config(str(checkpoint_dir))
    print(f"Loaded config from {checkpoint_dir}")
    print(f"Config: {config}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"JAX devices: {jax.devices()}")
    
    # Initialize model with same architecture
    model = VAE(
        base_features=config['base_features'],
        latent_dim=config['latent_dim']
    )
    
    # Load checkpoint
    key = random.key(42)
    dummy = jnp.ones((1, 256, 256, 1))
    variables = model.init(key, dummy, random.key(0), training=True)
    
    # Dummy optimizer for loading state
    import optax
    tx = optax.adamw(config['learning_rate'])
    state = TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)
    
    # Find checkpoint path (try final first, then latest numbered)
    checkpoint_path = checkpoint_dir / "checkpoint_final"
    if not checkpoint_path.exists():
        # Find latest numbered checkpoint
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        checkpoint_path = checkpoints[-1]
    
    print(f"Loading checkpoint: {checkpoint_path}")
    state = load_checkpoint(str(checkpoint_path), state)
    print(f"Checkpoint loaded successfully")
    print(f"Data directory: {data_dir}")
    
    # Create iterator without augmentation, using saved stats from training
    print("Creating data iterator (no augmentation, no shuffle)...")
    
    # Try to find stats file: first in checkpoint dir, then in data dir
    stats_path = checkpoint_dir / "polymer_stats.json"
    if not stats_path.exists():
        stats_path = data_dir / "polymer_stats.json"
        print(f"Note: Using stats from data directory: {stats_path}")
    else:
        print(f"Using stats from checkpoint directory: {stats_path}")
    
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Could not find polymer_stats.json in either:\n"
            f"  - {checkpoint_dir / 'polymer_stats.json'}\n"
            f"  - {data_dir / 'polymer_stats.json'}\n"
            f"Please ensure stats were saved during training."
        )
    
    iterator, num_batches, img_shape = create_polymer_iterator_from_saved_stats(
        data_dir=data_dir,
        stats_path=stats_path,
        batch_size=config['batch_size'],
        crop_size=config.get('crop_size', 256),
        stride=config.get('crop_size', 256),  # No overlap for inference
        shuffle=False,
        augment=False,
        val_split=None,  # Use all data
        split_seed=config.get('split_seed', 42),
    )
    
    print(f"Processing {num_batches} batches...")
    print(f"Image shape: {img_shape}")
    
    # Run inference
    @jax.jit
    def infer(state: TrainState, batch: dict, rng_key):
        images = batch['depth']
        x_recon, _, _, _ = state.apply_fn(
            {'params': state.params}, images, training=False, key=rng_key
        )
        return x_recon
    
    batch_idx = 0
    img_idx = 0
    
    # Use tqdm if available
    batch_iter = tqdm(iterator, total=num_batches, desc="Inference") if HAS_TQDM else iterator
    
    for batch in batch_iter:
        # Run inference
        key, infer_key = random.split(key)
        reconstructions = infer(state, batch, infer_key)
        
        # Convert to numpy
        originals = np.array(batch['depth'])
        reconstructions = np.array(reconstructions)
        
        # Save each crop as combined comparison image
        batch_size = originals.shape[0]
        for i in range(batch_size):
            orig = originals[i]
            recon = reconstructions[i]
            
            # Create side-by-side comparison: [Original | Reconstruction | Error]
            combined = create_comparison_image(orig, recon, error_scale=5.0)
            
            # Save with zero-padded index
            Image.fromarray(combined).save(output_dir / f"{img_idx:06d}.png")
            
            img_idx += 1
        
        batch_idx += 1
    
    print(f"\nInference complete!")
    print(f"Processed {img_idx} crops")
    print(f"Comparison images saved to: {output_dir}")
    print(f"Format: [Original | Reconstruction | Error (5x)]")


def main():
    if len(sys.argv) < 4:
        print("Usage: python inference.py /path/to/checkpoint_dir /path/to/output_dir /path/to/data_dir")
        print("\nExample:")
        print("  make run CMD='python examples/optical_profilometry/inference.py /data/experiments/wavelet_vae_20231119_120000 /data/inference_output /data/polymer_tfrecords'")
        sys.exit(1)
    
    checkpoint_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    data_dir = Path(sys.argv[3])
    
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory does not exist: {checkpoint_dir}")
        sys.exit(1)
    
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        sys.exit(1)
    
    run_inference(checkpoint_dir, output_dir, data_dir)


if __name__ == "__main__":
    main()

