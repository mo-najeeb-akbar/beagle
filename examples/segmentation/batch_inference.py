#!/usr/bin/env python3
"""
Fast batched inference for segmentation models.

Usage:
    # Infer on all images in a folder
    python batch_inference.py /data/experiments_nema_segmentation /path/to/images --output /path/to/outputs

    # Specify experiment ID
    python batch_inference.py /data/experiments_nema_segmentation /path/to/images --id 20251203_191813_0df02952

    # Adjust batch size for speed/memory tradeoff
    python batch_inference.py /data/experiments_nema_segmentation /path/to/images --batch-size 16

    # Save class predictions instead of logits
    python batch_inference.py /data/experiments_nema_segmentation /path/to/images --save-classes
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from flax import linen as nn

from beagle.experiments import ExperimentTracker
from beagle.network.hrnet import HRNetBackbone, SegmentationHead
from beagle.training import load_checkpoint


class SegmentationModel(nn.Module):
    """Simple wrapper combining HRNetBackbone + SegmentationHead.

    Returns dict with 'logits' key for inference.
    """
    num_classes: int
    num_stages: int = 3
    features: int = 32
    target_res: float = 1.0
    upsample_steps: int = 0
    use_sigmoid: bool = False

    def setup(self):
        self.backbone = HRNetBackbone(
            num_stages=self.num_stages,
            features=self.features,
            target_res=self.target_res
        )
        self.head = SegmentationHead(
            num_classes=self.num_classes,
            features=self.features,
            upsample_steps=self.upsample_steps,
            use_sigmoid=self.use_sigmoid,
            output_key='logits'
        )

    def __call__(self, x: jnp.ndarray, train: bool = False) -> dict[str, jnp.ndarray]:
        """Forward pass through backbone and head.

        Args:
            x: Input image [B, H, W, 1]
            train: Training mode

        Returns:
            Dict with 'logits' key [B, H, W, num_classes]
        """
        backbone_out = self.backbone(x, train=train)
        features = backbone_out['features']
        head_out = self.head(features, train=train)
        return head_out  # Returns {'logits': ...}


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration extracted from experiment."""
    num_stages: int
    features: int
    target_res: float
    train_backbone: bool
    outputs: tuple
    input_size: int = 512


def extract_model_config(exp_data: dict) -> ModelConfig:
    """Extract model config from experiment metadata."""
    config = exp_data['metadata']['config']
    model_cfg = config.get('model', {})
    return ModelConfig(
        num_stages=model_cfg.get('num_stages', 3),
        features=model_cfg.get('features', 32),
        target_res=model_cfg.get('target_res', 1.0),
        train_backbone=model_cfg.get('train_backbone', True),
        outputs=tuple(model_cfg.get('outputs', [(3, False, 2)])),
        input_size=model_cfg.get('input_size', 512),
    )


def load_model_from_experiment(
    experiments_dir: Path,
    experiment_id: str | None = None,
) -> tuple[jax.tree_util.Partial, ModelConfig]:
    """Load best model from experiments directory.

    Args:
        experiments_dir: Path to experiments directory
        experiment_id: Optional specific experiment ID, otherwise uses best

    Returns:
        tuple: (inference_fn, model_config)
    """
    tracker = ExperimentTracker(experiments_dir=experiments_dir)

    # Get experiment
    if experiment_id is None:
        experiments = tracker.compare_experiments('val_accuracy', mode='max', top_k=1)
        if not experiments:
            raise ValueError("No experiments found")
        experiment_id = experiments[0]['experiment_id']
        print(f"Using best experiment: {experiment_id}")
    else:
        print(f"Using specified experiment: {experiment_id}")

    # Load experiment data
    exp_data = tracker.load_experiment(experiment_id)
    model_config = extract_model_config(exp_data)

    print(f"Model config: stages={model_config.num_stages}, "
          f"features={model_config.features}, "
          f"input_size={model_config.input_size}, "
          f"outputs={model_config.outputs}")

    # Initialize model
    # Parse output configuration from tuple format
    num_classes, use_sigmoid, upsample_steps = model_config.outputs[0]

    model = SegmentationModel(
        num_classes=num_classes,
        num_stages=model_config.num_stages,
        features=model_config.features,
        target_res=model_config.target_res,
        upsample_steps=upsample_steps,
        use_sigmoid=use_sigmoid,
    )

    # Load checkpoint
    checkpoint_dir = exp_data['output_dir'] / 'checkpoints' / 'best'
    checkpoint_path = checkpoint_dir / 'checkpoint_final'

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    model_data = load_checkpoint(str(checkpoint_path))
    params, batch_stats = model_data['params'], model_data['batch_stats']

    # Create JIT-compiled inference function
    @jax.jit
    def inference_fn(images: jnp.ndarray) -> jnp.ndarray:
        """Run inference on batch of images.

        Args:
            images: [B, H, W, 1] preprocessed images

        Returns:
            logits: [B, H, W, num_classes] prediction logits
        """
        outputs = model.apply(
            {'params': params, 'batch_stats': batch_stats},
            images,
            train=False
        )
        # Extract logits from dict output
        return outputs['logits']

    return inference_fn, model_config


def load_and_preprocess_image(
    path: Path,
    target_size: int = 512,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Load and preprocess a single image.

    Args:
        path: Path to image file
        target_size: Size to resize to (square)

    Returns:
        tuple: (preprocessed_image [H, W, 1], original_shape (h, w))
    """
    # Load grayscale
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")

    original_shape = img.shape[:2]

    # Resize
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)

    # Normalize to [-1, 1]
    img = (img - 127.5) / 127.5

    # Add channel dimension
    img = img[:, :, np.newaxis]

    return img, original_shape


def collect_image_paths(image_dir: Path) -> list[Path]:
    """Collect all image paths from directory.

    Args:
        image_dir: Directory containing images

    Returns:
        Sorted list of image paths
    """
    extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
    paths = []

    for ext in extensions:
        paths.extend(image_dir.glob(f'*{ext}'))
        paths.extend(image_dir.glob(f'*{ext.upper()}'))

    return sorted(set(paths))


def batch_inference(
    inference_fn: jax.tree_util.Partial,
    image_paths: list[Path],
    output_dir: Path,
    batch_size: int = 8,
    target_size: int = 512,
    save_classes: bool = False,
) -> None:
    """Run batched inference on images.

    Args:
        inference_fn: JIT-compiled inference function
        image_paths: List of image paths to process
        output_dir: Directory to save predictions
        batch_size: Batch size for inference
        target_size: Input size for model
        save_classes: If True, save argmax classes; otherwise save logits
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_images = len(image_paths)
    n_batches = (n_images + batch_size - 1) // batch_size

    print(f"\nProcessing {n_images} images in {n_batches} batches (batch_size={batch_size})")
    print("=" * 80)

    total_time = 0.0
    processed = 0

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_images)
        batch_paths = image_paths[batch_start:batch_end]

        # Load and preprocess batch
        batch_images = []
        original_shapes = []

        for path in batch_paths:
            img, orig_shape = load_and_preprocess_image(path, target_size)
            batch_images.append(img)
            original_shapes.append(orig_shape)

        # Pad batch if needed (for last incomplete batch)
        current_batch_size = len(batch_images)
        if current_batch_size < batch_size:
            # Pad with zeros
            padding = [np.zeros_like(batch_images[0]) for _ in range(batch_size - current_batch_size)]
            batch_images.extend(padding)

        # Stack into batch
        batch_array = np.stack(batch_images, axis=0)

        # Move to device and run inference
        batch_jax = jax.device_put(batch_array)

        start_time = time.time()
        logits = inference_fn(batch_jax)  # [B, H, W, num_classes]
        logits.block_until_ready()  # Wait for computation to finish
        inference_time = time.time() - start_time

        total_time += inference_time

        # Convert to numpy
        logits_np = np.array(logits)

        # Save predictions for valid images in batch
        for idx, path in enumerate(batch_paths):
            pred = logits_np[idx]  # [H, W, num_classes]

            if save_classes:
                # Save class predictions
                pred_classes = np.argmax(pred, axis=-1).astype(np.uint8)  # [H, W]
                output_path = output_dir / f"{path.stem}_pred.png"
                cv2.imwrite(str(output_path), pred_classes)
            else:
                # Save logits as numpy array
                output_path = output_dir / f"{path.stem}_logits.npy"
                np.save(output_path, pred)

            processed += 1

        # Progress
        imgs_per_sec = current_batch_size / inference_time
        print(f"Batch {batch_idx + 1}/{n_batches}: "
              f"{current_batch_size} images in {inference_time:.3f}s "
              f"({imgs_per_sec:.1f} imgs/sec)")

    # Summary
    avg_time_per_image = total_time / processed
    avg_throughput = processed / total_time

    print("=" * 80)
    print(f"Completed {processed} images in {total_time:.2f}s")
    print(f"Average: {avg_time_per_image*1000:.1f}ms/image ({avg_throughput:.1f} imgs/sec)")
    print(f"Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Fast batched inference for segmentation'
    )
    parser.add_argument('experiments_dir', type=Path,
                        help='Directory containing experiments')
    parser.add_argument('image_dir', type=Path,
                        help='Directory containing images to process')
    parser.add_argument('--output', '-o', type=Path, required=True,
                        help='Output directory for predictions')
    parser.add_argument('--id', type=str, default=None,
                        help='Specific experiment ID (default: best)')
    parser.add_argument('--batch-size', '-b', type=int, default=8,
                        help='Batch size for inference (default: 8)')
    parser.add_argument('--save-classes', action='store_true',
                        help='Save class predictions (uint8 PNG) instead of logits (NPY)')

    args = parser.parse_args()

    if not args.experiments_dir.exists():
        raise FileNotFoundError(f"Experiments directory not found: {args.experiments_dir}")

    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    # Load model
    print("Loading model...")
    inference_fn, model_config = load_model_from_experiment(
        experiments_dir=args.experiments_dir,
        experiment_id=args.id,
    )

    # Collect images
    image_paths = collect_image_paths(args.image_dir)
    if not image_paths:
        raise ValueError(f"No images found in {args.image_dir}")

    print(f"Found {len(image_paths)} images")

    # Run batched inference
    batch_inference(
        inference_fn=inference_fn,
        image_paths=image_paths,
        output_dir=args.output,
        batch_size=args.batch_size,
        target_size=model_config.input_size,
        save_classes=args.save_classes,
    )


if __name__ == '__main__':
    main()
