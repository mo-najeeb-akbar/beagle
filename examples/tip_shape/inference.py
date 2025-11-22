"""Run inference on TIFF images using trained mask model.

Run: make run CMD='python examples/tip_shape/inference.py /data/experiments/hrnet_masks_XXX /data/tiff_input /data/output'
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
from jax import image as jax_image
from PIL import Image

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from beagle.training import TrainState, load_checkpoint, load_config
from beagle.dataset import load_field_stats

from mask_net import MaskNet


def load_tiff_grayscale(path: Path, target_size: int = 512) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Load TIFF image, convert to grayscale, and resize to target size.
    
    Args:
        path: Path to TIFF file
        target_size: Target size for model input
        
    Returns:
        Tuple of (normalized_image, original_size) or None if file cannot be loaded
            - normalized_image: [H, W, 1] float32 in [0, 1]
            - original_size: (height, width) of original image
    """
    try:
        # Load image
        img = Image.open(path)
        
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        original_size = img.size  # (width, height)
        
        # Resize to target size
        img_resized = img.resize((target_size, target_size), Image.BILINEAR)
        
        # Convert to numpy and normalize to [0, 1]
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        
        # Add channel dimension [H, W, 1]
        img_array = img_array[:, :, np.newaxis]
        
        return img_array, (original_size[1], original_size[0])  # Return (height, width)
    except Exception as e:
        print(f"Warning: Could not load {path.name}: {e}")
        return None


def load_image_stats(checkpoint_dir: Path) -> tuple[float, float] | None:
    """Try to load image statistics from checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        
    Returns:
        Tuple of (mean, std) or None if not found
    """
    stats_path = checkpoint_dir / "image_stats.json"
    if not stats_path.exists():
        return None
    
    try:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        # Stats format: {"image": [[mean], [std]]}
        mean = float(stats['image'][0][0])
        std = float(stats['image'][1][0])
        print(f"Loaded statistics from {stats_path}: mean={mean:.4f}, std={std:.4f}")
        return mean, std
    except Exception as e:
        print(f"Warning: Could not load statistics from {stats_path}: {e}")
        return None


def compute_statistics_from_images(
    tiff_files: list[Path], target_size: int = 512, max_samples: int = 100
) -> tuple[float, float]:
    """Compute mean and std from TIFF images.
    
    Args:
        tiff_files: List of paths to TIFF files
        target_size: Size to resize images to
        max_samples: Maximum number of images to sample
        
    Returns:
        Tuple of (mean, std)
    """
    print(f"Computing statistics from input images (sampling up to {max_samples} images)...")
    all_pixels = []
    loaded_count = 0
    
    sample_files = tiff_files[:max_samples] if len(tiff_files) > max_samples else tiff_files
    
    for tiff_path in sample_files:
        result = load_tiff_grayscale(tiff_path, target_size)
        if result is not None:
            image_resized, _ = result
            all_pixels.append(image_resized.flatten())
            loaded_count += 1
    
    if not all_pixels:
        raise ValueError("No valid images found for computing statistics")
    
    all_pixels = np.concatenate(all_pixels)
    mean = float(np.mean(all_pixels))
    std = float(np.std(all_pixels))
    
    print(f"Computed statistics from {loaded_count} valid images: mean={mean:.4f}, std={std:.4f}")
    return mean, std


def normalize_image_zscore(
    image: np.ndarray, mean: float, std: float
) -> np.ndarray:
    """Apply z-score normalization.
    
    Args:
        image: Image in [0, 1] range
        mean: Mean for normalization
        std: Std for normalization
        
    Returns:
        Normalized image
    """
    return (image - mean) / (std + 1e-8)


def resize_mask_to_original(
    mask: jnp.ndarray, original_size: tuple[int, int]
) -> np.ndarray:
    """Resize mask from model output to original image size.
    
    Args:
        mask: Predicted mask [1, H, W, 1]
        original_size: (height, width) of original image
        
    Returns:
        Resized mask as numpy array [H_orig, W_orig]
    """
    # Resize using JAX
    resized = jax_image.resize(
        mask,
        shape=(1, original_size[0], original_size[1], 1),
        method="bilinear"
    )
    
    # Convert to numpy and remove batch/channel dims
    return np.array(resized[0, :, :, 0])


def overlay_mask_on_image_np(
    image: np.ndarray, mask: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Overlay colored mask on grayscale image.
    
    Args:
        image: Grayscale image [H, W] in [0, 1]
        mask: Binary mask [H, W] in [0, 1]
        alpha: Transparency factor
        
    Returns:
        RGB overlay image [H, W, 3] as uint8
    """
    # Convert grayscale to RGB
    image_rgb = np.stack([image, image, image], axis=-1)
    
    # Create red mask overlay
    mask_rgb = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
    
    # Blend
    overlay = alpha * mask_rgb + (1 - alpha) * image_rgb
    overlay = np.clip(overlay, 0, 1)
    
    # Convert to uint8
    return (overlay * 255).astype(np.uint8)


def create_inference_fn(model, params, batch_stats):
    """Create JIT-compiled inference function."""
    @jax.jit
    def infer(image: jnp.ndarray):
        """Run inference on single image.
        
        Args:
            image: Input image [1, 512, 512, 1]
            
        Returns:
            Predicted mask probabilities [1, 128, 128, 1]
        """
        # Forward pass (training=False for batch norm)
        predictions = model.apply(
            {"params": params, "batch_stats": batch_stats},
            image,
            train=False
        )
        
        # Apply sigmoid to get probabilities
        predictions_prob = jax.nn.sigmoid(predictions)
        
        return predictions_prob
    
    return infer


def run_inference(
    checkpoint_dir: Path,
    input_dir: Path,
    output_dir: Path,
    alpha: float = 0.5
) -> None:
    """Run inference on folder of TIFF images.
    
    Args:
        checkpoint_dir: Path to experiment directory with checkpoints
        input_dir: Directory containing TIFF files
        output_dir: Directory to save overlay images
        alpha: Transparency factor for mask overlay
    """
    # Load config
    config = load_config(str(checkpoint_dir))
    print(f"Loaded config from {checkpoint_dir}")
    print(f"Config: {config}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"JAX devices: {jax.devices()}")
    
    # Initialize model with same architecture
    model = MaskNet(
        num_stages=config["num_stages"],
        features=config["features"],
        target_res=config["target_res"],
    )
    
    # Initialize model
    key = random.key(42)
    dummy = jnp.ones((1, config["input_size"], config["input_size"], 1))
    variables = model.init(key, dummy, train=False)
    
    # Create dummy optimizer for loading state
    tx = optax.adamw(config["learning_rate"])
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        batch_stats=variables["batch_stats"],
        tx=tx
    )
    
    # Find checkpoint path
    checkpoint_path = checkpoint_dir / "checkpoint_final"
    if not checkpoint_path.exists():
        # Find latest numbered checkpoint
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        checkpoint_path = checkpoints[-1]
    
    print(f"Loading checkpoint: {checkpoint_path}")
    model_data = load_checkpoint(str(checkpoint_path))
    params = model_data['params']
    batch_stats = model_data['batch_stats']
    print(f"Checkpoint loaded successfully")
    
    # Create inference function
    infer_fn = create_inference_fn(model, params, batch_stats)
    
    # Find all TIFF files
    tiff_patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    tiff_files = []
    for pattern in tiff_patterns:
        tiff_files.extend(input_dir.glob(pattern))
    
    tiff_files = sorted(tiff_files)
    
    if not tiff_files:
        print(f"No TIFF files found in {input_dir}")
        return
    
    print(f"Found {len(tiff_files)} TIFF files")
    
    # Load or compute statistics for normalization
    stats = load_field_stats("./tip_tfr/image_stats.json")
    mean, std = stats['image']
    
    print(f"Processing with alpha={alpha}...")
    print(f"Using normalization: mean={mean:.4f}, std={std:.4f}")
    
    # Process each image
    file_iter = tqdm(tiff_files, desc="Inference") if HAS_TQDM else tiff_files
    
    processed_count = 0
    skipped_count = 0
    
    for tiff_path in file_iter:
        # Load and preprocess image
        result = load_tiff_grayscale(tiff_path, target_size=config["input_size"])
        
        if result is None:
            skipped_count += 1
            continue
        
        image_resized, original_size = result
        
        # Apply z-score normalization
        image_normalized = normalize_image_zscore(image_resized, mean, std)
        
        # Add batch dimension [1, H, W, 1]
        image_batch = jnp.array(image_normalized[np.newaxis, :, :, :])
        
        # Run inference
        mask_pred = infer_fn(image_batch)
        
        # Resize mask to original resolution
        mask_full = resize_mask_to_original(mask_pred, original_size)
        
        # Load original image for overlay (at original resolution)
        try:
            img_original = Image.open(tiff_path)
            if img_original.mode != 'L':
                img_original = img_original.convert('L')
            img_original_array = np.array(img_original, dtype=np.float32) / 255.0
        except Exception as e:
            print(f"Warning: Could not reload {tiff_path.name} for overlay: {e}")
            skipped_count += 1
            continue
        
        # Create overlay
        overlay = overlay_mask_on_image_np(img_original_array, mask_full, alpha=alpha)
        
        # Save result
        output_path = output_dir / f"{tiff_path.stem}_overlay.png"
        Image.fromarray(overlay).save(output_path)
        
        # Also save the mask separately
        mask_path = output_dir / f"{tiff_path.stem}_mask.png"
        mask_uint8 = (mask_full * 255).astype(np.uint8)
        Image.fromarray(mask_uint8).save(mask_path)
        
        processed_count += 1
    
    print(f"\nInference complete!")
    print(f"Processed {processed_count} images successfully")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} invalid/unreadable images")
    print(f"Overlay images saved to: {output_dir}")
    print(f"Format: *_overlay.png (RGB with mask), *_mask.png (grayscale mask)")


def main():
    if len(sys.argv) < 4:
        print("Usage: python inference.py /path/to/checkpoint_dir /path/to/tiff_dir /path/to/output_dir [alpha]")
        print("\nExample:")
        print("  make run CMD='python examples/tip_shape/inference.py /data/experiments/hrnet_masks_20251121_215044 /data/tiff_input /data/output 0.5'")
        print("\nArguments:")
        print("  checkpoint_dir: Directory containing checkpoint_final and config.json")
        print("  tiff_dir: Directory with TIFF files to process")
        print("  output_dir: Directory to save results")
        print("  alpha: Optional transparency factor for overlay (default: 0.5)")
        sys.exit(1)
    
    checkpoint_dir = Path(sys.argv[1])
    input_dir = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])
    alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory does not exist: {checkpoint_dir}")
        sys.exit(1)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    run_inference(checkpoint_dir, input_dir, output_dir, alpha)


if __name__ == "__main__":
    main()

