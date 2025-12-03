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
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

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


def calculate_orientation_angle(mask: np.ndarray, threshold: float = 0.5) -> float:
    """Calculate rotation angle to orient object vertically with tip upward.
    
    Uses second-order central moments to find the major axis orientation.
    Rotates so the major (long) axis is vertical. Determines tip vs base by
    checking which end touches the image edge (edge = base, opposite = tip).
    
    Args:
        mask: Binary mask [H, W] with values in [0, 1]
        threshold: Threshold to binarize mask
        
    Returns:
        Rotation angle in degrees to orient tip upward (range: -180 to 180)
    """
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    
    # Calculate image moments
    M = cv2.moments(binary_mask)
    
    # Check if mask is empty
    if M['m00'] < 1e-6:
        return 0.0
    
    # Calculate centroid
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    
    # Calculate central moments for orientation
    mu20 = M['m20'] / M['m00'] - cx * cx
    mu02 = M['m02'] / M['m00'] - cy * cy
    mu11 = M['m11'] / M['m00'] - cx * cy
    
    # Calculate orientation angle of major axis
    angle_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    angle_deg = np.degrees(angle_rad)
    
    # Rotate to make major axis vertical
    rotation_needed = -90.0 - angle_deg
    
    # Determine which end touches the image edge
    height, width = mask.shape
    edge_margin = max(5, int(min(height, width) * 0.02))
    
    # Find extreme points along the major axis
    axis_length = min(height, width) * 0.5
    dx = axis_length * np.cos(angle_rad)
    dy = axis_length * np.sin(angle_rad)
    
    # Two ends of major axis
    end1_y, end1_x = cy - dy, cx - dx
    end2_y, end2_x = cy + dy, cx + dx
    
    # Check distance to nearest edge for each end
    def distance_to_edge(y: float, x: float) -> float:
        """Calculate minimum distance to any image edge."""
        dist_top = y
        dist_bottom = height - y
        dist_left = x
        dist_right = width - x
        return min(dist_top, dist_bottom, dist_left, dist_right)
    
    dist1 = distance_to_edge(end1_y, end1_x)
    dist2 = distance_to_edge(end2_y, end2_x)
    
    # The end closer to edge is the base, other is the tip
    # We want tip pointing up (lower y value)
    if dist1 < dist2:
        # end1 is closer to edge (base)
        # Check if end1 is higher (lower y) - if so, flip 180
        if end1_y < end2_y:
            rotation_needed += 180.0
    else:
        # end2 is closer to edge (base)
        # Check if end2 is higher (lower y) - if so, flip 180
        if end2_y < end1_y:
            rotation_needed += 180.0
    
    # Normalize to [-180, 180]
    while rotation_needed > 180:
        rotation_needed -= 360
    while rotation_needed < -180:
        rotation_needed += 360
    
    return rotation_needed


def rotate_image_array(
    image: np.ndarray, angle: float, fill_value: float = 0.0
) -> np.ndarray:
    """Rotate image by given angle (pure function).
    
    Args:
        image: Image array [H, W] or [H, W, C]
        angle: Rotation angle in degrees (positive = counter-clockwise)
        fill_value: Value for empty pixels after rotation
        
    Returns:
        Rotated image with same shape
    """
    if not HAS_CV2:
        # Fallback: return original if cv2 not available
        return image.copy()
    
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    
    # Determine output shape (keep same size)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill_value
    )
    
    return rotated


def get_mask_bounding_box(
    mask: np.ndarray, threshold: float = 0.5, padding: int = 10
) -> tuple[int, int, int, int]:
    """Find bounding box of binary mask (pure function).
    
    Args:
        mask: Binary mask [H, W] with values in [0, 1]
        threshold: Threshold to binarize mask
        padding: Pixels to add around bounding box
        
    Returns:
        Tuple of (y_min, y_max, x_min, x_max) inclusive indices
    """
    binary_mask = mask > threshold
    
    # Find rows and columns with any True values
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        # Empty mask, return full image bounds
        return 0, mask.shape[0], 0, mask.shape[1]
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Add padding
    height, width = mask.shape
    y_min = max(0, y_min - padding)
    y_max = min(height - 1, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(width - 1, x_max + padding)
    
    return y_min, y_max + 1, x_min, x_max + 1


def crop_to_mask(
    image: np.ndarray, mask: np.ndarray, threshold: float = 0.5, padding: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Crop image and mask to mask bounding box (pure function).
    
    Args:
        image: Image array [H, W] or [H, W, C]
        mask: Binary mask [H, W] with values in [0, 1]
        threshold: Threshold to binarize mask for bbox calculation
        padding: Pixels to add around bounding box
        
    Returns:
        Tuple of (cropped_image, cropped_mask)
    """
    y_min, y_max, x_min, x_max = get_mask_bounding_box(mask, threshold, padding)
    
    cropped_image = image[y_min:y_max, x_min:x_max].copy()
    cropped_mask = mask[y_min:y_max, x_min:x_max].copy()
    
    return cropped_image, cropped_mask


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
    orient_upward: bool = False,
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
        orient_upward: If True, rotate images so objects point upward
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
        
        # Apply orientation correction if requested
        if orient_upward and HAS_CV2:
            rotation_angle = calculate_orientation_angle(mask_full)
            img_original_array = rotate_image_array(img_original_array, -rotation_angle)
            mask_full = rotate_image_array(mask_full, -rotation_angle)
        elif orient_upward and not HAS_CV2:
            print("Warning: OpenCV (cv2) not available, skipping orientation correction")
            orient_upward = False  # Disable for remaining images
        
        # Create overlay
        overlay = overlay_mask_on_image_np(img_original_array, mask_full, alpha=alpha)
        
        # Save full-size results
        output_path = output_dir / f"{tiff_path.stem}_tflite_overlay.png"
        Image.fromarray(overlay).save(output_path)
        
        mask_path = output_dir / f"{tiff_path.stem}_tflite_mask.png"
        mask_uint8 = (mask_full * 255).astype(np.uint8)
        Image.fromarray(mask_uint8).save(mask_path)
        
        # Save cropped version (preserves pixel resolution)
        if orient_upward and HAS_CV2:
            cropped_image, cropped_mask = crop_to_mask(
                img_original_array, mask_full, threshold=0.5, padding=10
            )
            
            # Apply binary mask to cropped image (set background to black)
            binary_mask = (cropped_mask > 0.5).astype(np.float32)
            cropped_masked = cropped_image * binary_mask
            
            # Save cropped & masked grayscale image
            cropped_path = output_dir / f"{tiff_path.stem}_tflite_cropped.png"
            cropped_uint8 = (cropped_masked * 255).astype(np.uint8)
            Image.fromarray(cropped_uint8).save(cropped_path)
            
            # Save cropped mask
            cropped_mask_path = output_dir / f"{tiff_path.stem}_tflite_cropped_mask.png"
            cropped_mask_uint8 = (cropped_mask * 255).astype(np.uint8)
            Image.fromarray(cropped_mask_uint8).save(cropped_mask_path)
        
        processed_count += 1
    
    print(f"\n" + "=" * 70)
    print(f"Inference complete!")
    print(f"=" * 70)
    print(f"Processed: {processed_count} images")
    if skipped_count > 0:
        print(f"Skipped: {skipped_count} invalid images")
    print(f"Output directory: {output_dir}")
    print(f"\nOutput files per image:")
    print(f"  *_tflite_overlay.png - RGB overlay (full resolution)")
    print(f"  *_tflite_mask.png - Grayscale mask (full resolution)")
    if orient_upward and HAS_CV2:
        print(f"  *_tflite_cropped.png - Cropped & oriented grayscale image")
        print(f"  *_tflite_cropped_mask.png - Cropped & oriented mask")
    
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
        print("  --orient-upward: Rotate images to orient tips upward and save cropped versions (requires cv2)")
        print("\nExamples:")
        print("  # Basic with orientation and cropping:")
        print("  make run CMD='python examples/tip_shape/inference_tflite.py monet_segmentation.tflite /data/tiff_input /data/output --orient-upward'")
        print("\n  # Compare with JAX model:")
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
    orient_upward = False
    
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
        elif sys.argv[i] == '--orient-upward':
            orient_upward = True
            i += 1
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
        orient_upward=orient_upward,
    )


if __name__ == "__main__":
    main()

