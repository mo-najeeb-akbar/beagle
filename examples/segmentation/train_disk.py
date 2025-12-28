from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import jax.random as random
import optax
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from flax import linen as nn

from beagle.network.hrnet import HRNetBackbone, SegmentationHead
from beagle.experiments import ExperimentConfig, ExperimentTracker, ModelRegistry
from beagle.training import TrainState, save_checkpoint

from data_loader_disk import create_segmentation_iterator
from configs import DatasetConfig, ModelConfig, TrainingConfig

from training_steps import create_train_step
from visualize_segmentation import create_overlay, DEFAULT_COLORS


class SegmentationModel(nn.Module):
    """Simple wrapper combining HRNetBackbone + SegmentationHead.

    Returns dict with 'logits' key for compatibility with training_steps.py.
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

    def __call__(self, x: jnp.ndarray, train: bool = True) -> dict[str, jnp.ndarray]:
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


def save_batch_visualizations(
    state: TrainState,
    batch: dict,
    num_classes: int,
    output_dir: Path,
    epoch: int,
    max_samples: int = 4,
    alpha: float = 0.5,
):
    """Save visualization overlays for a batch during training.

    Args:
        state: Current training state
        batch: Batch dict with 'image' and 'mask' keys
        num_classes: Number of segmentation classes
        output_dir: Directory to save visualizations
        epoch: Current epoch number
        max_samples: Maximum number of samples to visualize from batch
        alpha: Transparency of overlay
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    images = batch['image']
    masks_gt = batch['mask']

    # Run inference
    outputs = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        images,
        train=False
    )

    # Get predictions from dict - logits [B, H, W, num_classes]
    pred_logits = outputs['logits']
    pred_classes = jnp.argmax(pred_logits, axis=-1)  # [B, H, W]

    # Convert to numpy
    images_np = np.array(images)
    masks_gt_np = np.array(masks_gt)
    pred_classes_np = np.array(pred_classes)

    # Get colors
    colors = DEFAULT_COLORS[:num_classes]

    # Process each sample in batch (up to max_samples)
    batch_size = min(images_np.shape[0], max_samples)
    for idx in range(batch_size):
        img = images_np[idx]  # [H, W, 1] in range [-1, 1]
        gt = masks_gt_np[idx, :, :, 0].astype(np.uint8)  # [H, W]
        pred = pred_classes_np[idx].astype(np.uint8)  # [H, W]

        # Create overlays using the visualization function
        gt_overlay = create_overlay(img, gt, num_classes, colors, alpha)
        pred_overlay = create_overlay(img, pred, num_classes, colors, alpha)

        # Denormalize original image for display
        img_rgb = ((img + 1.0) * 127.5).astype(np.uint8)
        img_rgb = np.repeat(img_rgb, 3, axis=2)  # Convert to RGB

        # Create figure with 3 panels
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_rgb)
        axes[0].set_title('Original', fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(gt_overlay)
        axes[1].set_title('Ground Truth', fontsize=12)
        axes[1].axis('off')

        axes[2].imshow(pred_overlay)
        axes[2].set_title('Prediction', fontsize=12)
        axes[2].axis('off')

        plt.tight_layout()

        # Save
        save_path = output_dir / f'epoch_{epoch:04d}_sample_{idx:02d}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    print(f"  Saved {batch_size} visualizations to {output_dir}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py /path/to/nema_dish_tfrecords")
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    print(f"Data directory: {data_dir}")

    # 1. Load data first to determine num_classes
    train_iter, num_classes = create_segmentation_iterator(dataset_path=data_dir, augmentation_multiplier=20)
    print(f"Number of classes: {num_classes}")

    # 2. Create config with correct num_classes
    config = ExperimentConfig(
        name='segmentation_minimal',
        dataset=DatasetConfig(),
        model=ModelConfig(
            outputs=((num_classes, False, 2),)  # Set num_classes dynamically
        ),
        training=TrainingConfig(),
        seed=42
    )

    print(f"Config hash: {config.hash()}")

    # 3. Create experiment
    tracker = ExperimentTracker(experiments_dir='/data/experiments_nema_segmentation')
    run = tracker.create_experiment(name=config.name, config=config)
    print(f"Experiment: {run.experiment_id}")


    # 4. Initialize model
    # Parse output configuration from tuple format
    _, use_sigmoid, upsample_steps = config.model.outputs[0]

    model = SegmentationModel(
        num_classes=num_classes,
        num_stages=config.model.num_stages,
        features=config.model.features,
        target_res=config.model.target_res,
        upsample_steps=upsample_steps,
        use_sigmoid=use_sigmoid,
    )

    key = random.key(config.seed)
    key, init_key = random.split(key)
    dummy = jnp.ones((1, config.model.input_size, config.model.input_size, 1))
    variables = model.init(init_key, dummy, train=False)
    params, batch_stats = variables['params'], variables['batch_stats']

    tx = optax.adamw(config.training.learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, batch_stats=batch_stats, tx=tx)

    # Model steps
    train_step = create_train_step(num_classes=num_classes)

    # Setup visualization directory
    viz_dir = run.output_dir / 'visualizations'
    viz_frequency = 5  # Save visualizations every N epochs
    avg_loss = 0.0
    for epoch in range(config.training.num_epochs):
        train_losses = []
        last_batch = None
        for _ in range(100):
            batch = next(train_iter)
            state, metrics = train_step(state, batch)
            train_losses.append(float(metrics['loss']))
            last_batch = batch

        avg_loss = sum(train_losses) / len(train_losses)
        print(f"Epoch {epoch + 1}: train={avg_loss:.4f}")

        # Save visualizations every viz_frequency epochs
        if (epoch + 1) % viz_frequency == 0 and last_batch is not None:
            print(f"  Saving visualizations for epoch {epoch + 1}...")
            save_batch_visualizations(
                state=state,
                batch=last_batch,
                num_classes=num_classes,
                output_dir=viz_dir,
                epoch=epoch + 1,
                max_samples=4,
                alpha=0.5,
            )

    # 5. Save final checkpoint
    checkpoint_dir = run.output_dir / 'checkpoints' / 'best'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'checkpoint_final'

    print(f"\nSaving checkpoint to: {checkpoint_path}")
    save_checkpoint(
        checkpoint_path=checkpoint_path,
        state=state,
    )

    # 6. Finish experiment
    run.finish(status='completed')
    print(run.summary())

    # 7. Register model
    registry = ModelRegistry(registry_dir='/data/models_nema_segmentation')
    registry.register_model(
        model_name=config.name,
        experiment_id=run.experiment_id,
        config_hash=config.hash(),
        checkpoint_path=checkpoint_dir,
        metrics={'best_val_accuracy': avg_loss},
        timestamp=run.metadata.timestamp
    )


if __name__ == '__main__':
    main()