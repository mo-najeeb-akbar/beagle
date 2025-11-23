"""Run inference on TIFF images using TFLite model.

Compare outputs with JAX model to verify conversion correctness.

Usage:
    make run CMD='python examples/tip_shape/inference_tflite.py monet_segmentation.tflite /data/tiff_input /data/output [--compare-jax /path/to/checkpoint]'
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import jax.random as random
import numpy as np
import tensorflow as tf
from PIL import Image

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from beagle.dataset import load_field_stats
from beagle.network.hrnet import MoNet
from beagle.training import load_checkpoint, load_config


def load_tiff_grayscale(
    path: Path, target_size: int = 512
) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Load TIFF image, convert to grayscale, and resize.
    
    Args:
        path: Path to TIFF file
        target_size: Target size for model input
        
    Returns:
        Tuple of (normalized_image, original_size) or None
    """
    try:
        img = Image.open(path)
        
        if img.mode != 'L':
            img = img.convert('L')
        
        original_size = img.size
        img_resized = img.resize((target_size, target_size), Image.BILINEAR)
        
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = img_array[:, :, np.newaxis]
        
        return img_array, (original_size[1], original_size[0])
    except Exception as e:
        print(f"Warning: Could not load {path.name}: {e}")
        return None


def normalize_image_zscore(
    image: np.ndarray, mean: float, std: float
) -> np.ndarray:
    """Apply z-score normalization (pure function)."""
    return (image - mean) / (std + 1e-8)


def resize_mask_to_original(
    mask: np.ndarray, original_size: tuple[int, int]
) -> np.ndarray:
    """Resize mask from model output to original image size.
    
    Args:
        mask: Predicted mask [1, H, W, 1]
        original_size: (height, width) of original image
        
    Returns:
        Resized mask [H_orig, W_orig]
    """
    # Use TensorFlow for resizing
    mask_tf = tf.convert_to_tensor(mask)
    resized = tf.image.resize(
        mask_tf,
        size=(original_size[0], original_size[1]),
        method='bilinear'
    )
    return resized.numpy()[0, :, :, 0]


def overlay_mask_on_image_np(
    image: np.ndarray, mask: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Overlay colored mask on grayscale image (pure function).
    
    Args:
        image: Grayscale image [H, W] in [0, 1]
        mask: Binary mask [H, W] in [0, 1]
        alpha: Transparency factor
        
    Returns:
        RGB overlay image [H, W, 3] as uint8
    """
    image_rgb = np.stack([image, image, image], axis=-1)
    mask_rgb = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
    
    overlay = alpha * mask_rgb + (1 - alpha) * image_rgb
    overlay = np.clip(overlay, 0, 1)
    
    return (overlay * 255).astype(np.uint8)


def load_jax_model(checkpoint_dir: Path, input_size: int = 512):
    """Load JAX model for comparison (optional)."""
    config = load_config(str(checkpoint_dir))
    
    model = MoNet(
        num_stages=config["num_stages"],
        features=config["features"],
        target_res=config["target_res"],
        train_bb=config["train_backbone"],
        outputs=config.get("outputs", [(1, True, 2)]),
    )
    
    checkpoint_path = checkpoint_dir / "checkpoint_final"
    if not checkpoint_path.exists():
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        checkpoint_path = checkpoints[-1]
    
    print(f"Loading JAX checkpoint: {checkpoint_path}")
    model_data = load_checkpoint(str(checkpoint_path))
    params = model_data['params']
    batch_stats = model_data['batch_stats']
    
    return model, params, batch_stats, config


def run_inference_tflite(
    tflite_path: Path,
    input_dir: Path,
    output_dir: Path,
    stats_path: str | None = None,
    jax_checkpoint_dir: Path | None = None,
    alpha: float = 0.5,
    input_size: int = 512,
) -> None:
    """Run TFLite inference on TIFF images.
    
    Args:
        tflite_path: Path to TFLite model file
        input_dir: Directory containing TIFF files
        output_dir: Directory to save overlay images
        stats_path: Path to image stats JSON file
        jax_checkpoint_dir: Optional path to JAX checkpoint for comparison
        alpha: Transparency factor for mask overlay
        input_size: Input image size for model
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading TFLite model from: {tflite_path}")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"TFLite model info:")
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Number of outputs: {len(output_details)}")
    for i, detail in enumerate(output_details):
        print(f"  Output {i} shape: {detail['shape']}")
    
    # Load JAX model for comparison if requested
    jax_model = None
    if jax_checkpoint_dir is not None:
        print(f"\nLoading JAX model for comparison...")
        jax_model, jax_params, jax_batch_stats, jax_config = load_jax_model(
            jax_checkpoint_dir, input_size
        )
        print(f"✓ JAX model loaded")
    
    # Load normalization statistics
    if stats_path is None:
        stats_path = "./tip_tfr/image_stats.json"
    
    print(f"\nLoading normalization stats from: {stats_path}")
    stats = load_field_stats(stats_path)
    mean, std = stats['image']
    print(f"  Mean: {mean:.4f}, Std: {std:.4f}")
    
    # Find TIFF files
    tiff_patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    tiff_files = []
    for pattern in tiff_patterns:
        tiff_files.extend(input_dir.glob(pattern))
    
    tiff_files = sorted(tiff_files)
    
    if not tiff_files:
        print(f"No TIFF files found in {input_dir}")
        return
    
    print(f"\nFound {len(tiff_files)} TIFF files")
    print(f"Processing with alpha={alpha}...")
    
    # Process each image
    file_iter = tqdm(tiff_files, desc="Inference") if HAS_TQDM else tiff_files
    
    processed_count = 0
    skipped_count = 0
    comparison_diffs = []
    
    for tiff_path in file_iter:
        # Load and preprocess image
        result = load_tiff_grayscale(tiff_path, target_size=input_size)
        
        if result is None:
            skipped_count += 1
            continue
        
        image_resized, original_size = result
        
        # Apply z-score normalization
        image_normalized = normalize_image_zscore(image_resized, mean, std)
        
        # Add batch dimension [1, H, W, 1]
        image_batch = image_normalized[np.newaxis, :, :, :].astype(np.float32)
        
        # Run TFLite inference
        interpreter.set_tensor(input_details[0]['index'], image_batch)
        interpreter.invoke()
        
        # Get mask prediction (first output is the mask head)
        mask_pred = interpreter.get_tensor(output_details[1]['index'])
        
        # Compare with JAX model if available
        if jax_model is not None:
            jax_input = jnp.array(image_batch)
            jax_outputs = jax_model.apply(
                {'params': jax_params, 'batch_stats': jax_batch_stats},
                jax_input,
                train=False
            )
            jax_mask = np.array(jax_outputs[0])  # First output is mask
            
            diff = np.abs(mask_pred - jax_mask).max()
            comparison_diffs.append(diff)
        
        # Resize mask to original resolution
        mask_full = resize_mask_to_original(mask_pred, original_size)
        
        # Load original image for overlay
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
        
        # Save results
        output_path = output_dir / f"{tiff_path.stem}_tflite_overlay.png"
        Image.fromarray(overlay).save(output_path)
        
        mask_path = output_dir / f"{tiff_path.stem}_tflite_mask.png"
        mask_uint8 = (mask_full * 255).astype(np.uint8)
        Image.fromarray(mask_uint8).save(mask_path)
        
        processed_count += 1
    
    print(f"\n" + "=" * 70)
    print(f"Inference complete!")
    print(f"=" * 70)
    print(f"Processed: {processed_count} images")
    if skipped_count > 0:
        print(f"Skipped: {skipped_count} invalid images")
    print(f"Output directory: {output_dir}")
    print(f"Format: *_tflite_overlay.png (RGB), *_tflite_mask.png (grayscale)")
    
    if comparison_diffs:
        print(f"\n" + "=" * 70)
        print(f"JAX vs TFLite Comparison:")
        print(f"=" * 70)
        print(f"  Images compared: {len(comparison_diffs)}")
        print(f"  Max difference: {max(comparison_diffs):.2e}")
        print(f"  Mean difference: {np.mean(comparison_diffs):.2e}")
        print(f"  Median difference: {np.median(comparison_diffs):.2e}")
        
        if max(comparison_diffs) < 1e-3:
            print(f"\n✓ TFLite and JAX models produce nearly identical outputs!")
        elif max(comparison_diffs) < 0.01:
            print(f"\n✓ TFLite and JAX models are very close (small numerical differences)")
        else:
            print(f"\n⚠ TFLite and JAX models have noticeable differences")


def main():
    if len(sys.argv) < 4:
        print("Usage: python inference_tflite.py model.tflite /path/to/tiff_dir /path/to/output [OPTIONS]")
        print("\nRequired arguments:")
        print("  model.tflite: Path to TFLite model file")
        print("  tiff_dir: Directory with TIFF files to process")
        print("  output_dir: Directory to save results")
        print("\nOptional arguments:")
        print("  --alpha FLOAT: Transparency factor for overlay (default: 0.5)")
        print("  --stats PATH: Path to image_stats.json (default: ./tip_tfr/image_stats.json)")
        print("  --compare-jax PATH: Path to JAX checkpoint directory for comparison")
        print("  --input-size INT: Input size for model (default: 512)")
        print("\nExample:")
        print("  make run CMD='python examples/tip_shape/inference_tflite.py monet_segmentation.tflite /data/tiff_input /data/output --compare-jax /data/experiments/monet_masks_XXX'")
        sys.exit(1)
    
    tflite_path = Path(sys.argv[1])
    input_dir = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])
    
    # Parse optional arguments
    alpha = 0.5
    stats_path = None
    jax_checkpoint_dir = None
    input_size = 512
    
    i = 4
    while i < len(sys.argv):
        if sys.argv[i] == '--alpha' and i + 1 < len(sys.argv):
            alpha = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--stats' and i + 1 < len(sys.argv):
            stats_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--compare-jax' and i + 1 < len(sys.argv):
            jax_checkpoint_dir = Path(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--input-size' and i + 1 < len(sys.argv):
            input_size = int(sys.argv[i + 1])
            i += 2
        else:
            print(f"Unknown argument: {sys.argv[i]}")
            sys.exit(1)
    
    if not tflite_path.exists():
        print(f"Error: TFLite model does not exist: {tflite_path}")
        sys.exit(1)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if jax_checkpoint_dir and not jax_checkpoint_dir.exists():
        print(f"Error: JAX checkpoint directory does not exist: {jax_checkpoint_dir}")
        sys.exit(1)
    
    run_inference_tflite(
        tflite_path=tflite_path,
        input_dir=input_dir,
        output_dir=output_dir,
        stats_path=stats_path,
        jax_checkpoint_dir=jax_checkpoint_dir,
        alpha=alpha,
        input_size=input_size,
    )


if __name__ == "__main__":
    main()

