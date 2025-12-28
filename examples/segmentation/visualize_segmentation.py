#!/usr/bin/env python3
"""
Visualize segmentation predictions as colored overlays on original images.

Usage:
    python visualize_segmentation.py <image_path> <checkpoint_dir> [--output <output_path>]
"""
from __future__ import annotations

import sys
from pathlib import Path
import argparse

import numpy as np
import cv2
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from beagle.network.hrnet import MoNet
from beagle.training import TrainState, load_checkpoint
from configs import ModelConfig


# Default color palette for segmentation classes
DEFAULT_COLORS = [
    [0, 0, 0],        # Class 0: Black (background)
    [255, 0, 0],      # Class 1: Red
    [0, 255, 0],      # Class 2: Green
    [0, 0, 255],      # Class 3: Blue
    [255, 255, 0],    # Class 4: Yellow
    [255, 0, 255],    # Class 5: Magenta
    [0, 255, 255],    # Class 6: Cyan
    [255, 128, 0],    # Class 7: Orange
    [128, 0, 255],    # Class 8: Purple
    [0, 255, 128],    # Class 9: Spring Green
]


def load_image(path: str | Path, target_size: int = 512) -> np.ndarray:
    """Load and preprocess image.

    Args:
        path: Path to image file
        target_size: Size to resize image to (square)

    Returns:
        Preprocessed image [H, W, 1] in range [-1, 1]
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, np.newaxis]
    # Normalize to [-1, 1]
    img = (img - 127.5) / 127.5
    return img


def predict_segmentation(
    state: TrainState,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on a single image.

    Args:
        state: Trained model state
        image: Preprocessed image [H, W, 1]

    Returns:
        tuple: (pred_classes, pred_probs) where
            pred_classes: [H, W] array of predicted class indices
            pred_probs: [H, W, num_classes] array of class probabilities
    """
    # Add batch dimension
    image_batch = jnp.array(image[np.newaxis, ...])

    # Run inference
    outputs = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        image_batch,
        train=False
    )

    # First output is mask prediction - logits [1, H, W, num_classes]
    pred_logits = outputs[0][0]  # Remove batch dimension -> [H, W, num_classes]

    # Convert to probabilities
    pred_probs = jax.nn.softmax(pred_logits, axis=-1)

    # Get predicted class indices
    pred_classes = jnp.argmax(pred_logits, axis=-1)

    return np.array(pred_classes), np.array(pred_probs)


def create_overlay(
    original_image: np.ndarray,
    pred_classes: np.ndarray,
    num_classes: int,
    colors: list[list[int]] | None = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Create colored overlay of segmentation on original image.

    Args:
        original_image: Original grayscale image [H, W, 1] in range [-1, 1]
        pred_classes: Predicted class indices [H, W]
        num_classes: Number of segmentation classes
        colors: List of RGB colors for each class (default: DEFAULT_COLORS)
        alpha: Transparency of overlay (0=transparent, 1=opaque)

    Returns:
        RGB overlay image [H, W, 3] in range [0, 255]
    """
    if colors is None:
        colors = DEFAULT_COLORS[:num_classes]

    # Convert grayscale to RGB and denormalize to [0, 255]
    img_rgb = ((original_image + 1.0) * 127.5).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

    # Create color mask
    h, w = pred_classes.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx in range(num_classes):
        mask = (pred_classes == class_idx)
        color_mask[mask] = colors[class_idx]

    # Blend original image with color mask
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, color_mask, alpha, 0)

    return overlay


def visualize_with_legend(
    original_image: np.ndarray,
    pred_classes: np.ndarray,
    pred_probs: np.ndarray,
    num_classes: int,
    colors: list[list[int]] | None = None,
    alpha: float = 0.5,
    class_names: list[str] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Create visualization with overlay and class distribution.

    Args:
        original_image: Original grayscale image [H, W, 1] in range [-1, 1]
        pred_classes: Predicted class indices [H, W]
        pred_probs: Predicted class probabilities [H, W, num_classes]
        num_classes: Number of segmentation classes
        colors: List of RGB colors for each class
        alpha: Transparency of overlay
        class_names: Optional names for each class
        save_path: Optional path to save visualization
        show: Whether to display the plot
    """
    if colors is None:
        colors = DEFAULT_COLORS[:num_classes]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Create overlay
    overlay = create_overlay(original_image, pred_classes, num_classes, colors, alpha)

    # Compute class distribution
    h, w = pred_classes.shape
    total_pixels = h * w
    class_counts = [(pred_classes == i).sum() for i in range(num_classes)]
    class_percentages = [count / total_pixels * 100 for count in class_counts]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    img_display = ((original_image + 1.0) * 127.5).astype(np.uint8).squeeze()
    axes[0].imshow(img_display, cmap='gray')
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')

    # Overlay
    axes[1].imshow(overlay)
    axes[1].set_title(f'Segmentation Overlay (alpha={alpha})', fontsize=14)
    axes[1].axis('off')

    # Class distribution
    colors_norm = [[c / 255.0 for c in color] for color in colors]
    bars = axes[2].barh(class_names, class_percentages, color=colors_norm)
    axes[2].set_xlabel('Percentage (%)', fontsize=12)
    axes[2].set_title('Class Distribution', fontsize=14)
    axes[2].set_xlim(0, 100)

    # Add percentage labels
    for bar, pct in zip(bars, class_percentages):
        if pct > 0.1:  # Only show label if > 0.1%
            axes[2].text(
                pct, bar.get_y() + bar.get_height() / 2,
                f'{pct:.1f}%',
                ha='left', va='center', fontsize=10
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")

    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize segmentation predictions on an image'
    )
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint directory')
    parser.add_argument('--output', '-o', type=str, help='Path to save visualization')
    parser.add_argument('--alpha', type=float, default=0.5, help='Overlay transparency (0-1)')
    parser.add_argument('--no-show', action='store_true', help='Don\'t display the plot')
    parser.add_argument('--num-classes', type=int, help='Number of classes (auto-detected if not provided)')

    args = parser.parse_args()

    # Load image
    print(f"Loading image: {args.image}")
    image = load_image(args.image)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    config = ModelConfig()
    model = MoNet(
        num_stages=config.num_stages,
        features=config.features,
        target_res=config.target_res,
        train_bb=config.train_backbone,
        outputs=config.outputs,
    )

    # Initialize with dummy input to get structure
    key = jax.random.key(42)
    dummy = jnp.ones((1, config.input_size, config.input_size, 1))
    variables = model.init(key, dummy, train=False)

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    state = load_checkpoint(checkpoint_path, model, variables)

    # Run inference
    print("Running inference...")
    pred_classes, pred_probs = predict_segmentation(state, image)

    # Determine number of classes
    num_classes = args.num_classes if args.num_classes else pred_probs.shape[-1]
    print(f"Detected {num_classes} classes")

    # Print statistics
    unique_classes = np.unique(pred_classes)
    print(f"Predicted classes in image: {unique_classes}")

    # Visualize
    visualize_with_legend(
        original_image=image,
        pred_classes=pred_classes,
        pred_probs=pred_probs,
        num_classes=num_classes,
        alpha=args.alpha,
        save_path=args.output,
        show=not args.no_show,
    )


if __name__ == '__main__':
    main()
