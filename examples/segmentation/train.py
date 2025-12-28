from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import optax
from flax import linen as nn

from beagle.network.hrnet import HRNetBackbone, SegmentationHead
from beagle.experiments import ExperimentConfig, ExperimentTracker, ModelRegistry
from beagle.training import TrainState, save_checkpoint
from beagle.visualization import create_viz_callback, VizConfig

from data_loader import create_segmentation_iterator
from configs import DatasetConfig, ModelConfig, TrainingConfig

from training_steps import create_train_step, create_val_step


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

# def overlay_mask_on_image(
#     image: jnp.ndarray, mask: jnp.ndarray, num_classes: int = 3, alpha: float = 0.5
# ) -> jnp.ndarray:
#     """Overlay colored mask on grayscale image (pure function).
    
#     Args:
#         image: Grayscale image [B, H, W, 1]
#         mask: Class indices [B, H, W] or probabilities [B, H, W, num_classes]
#         num_classes: Number of classes
#         alpha: Transparency factor
        
#     Returns:
#         RGB overlay image [B, H, W, 3]
#     """
#     batch, height, width, _ = image.shape
    
#     # Convert grayscale to RGB
#     image_rgb = jnp.concatenate([image, image, image], axis=-1)
    
#     # If mask is probabilities, convert to class indices
#     if mask.ndim == 4:  # [B, H, W, num_classes]
#         mask_indices = jnp.argmax(mask, axis=-1)  # [B, H, W]
#     else:  # Already indices [B, H, W]
#         mask_indices = mask
    
#     # Define colors for each class (RGB)
#     # Class 0: black (background), Class 1: red (glass), Class 2: blue (white tag)
#     colors = jnp.array([
#         [0.0, 0.0, 0.0],  # Class 0: black/background
#         [1.0, 0.0, 0.0],  # Class 1: red
#         [0.0, 0.0, 1.0],  # Class 2: blue
#     ])
    
#     # Create colored mask by indexing into colors
#     mask_rgb = colors[mask_indices]  # [B, H, W, 3]
    
#     # Blend (only where mask is non-background)
#     is_background = (mask_indices == 0)[..., None]  # [B, H, W, 1]
#     overlay = jnp.where(is_background, image_rgb, alpha * mask_rgb + (1 - alpha) * image_rgb)
    
#     return jnp.clip(overlay, 0, 1)

# def create_viz_fn(model_apply, num_classes: int = 3):
#     """Create visualization function."""
#     @jax.jit
#     def viz_fn(state, batch, rng_key):
#         images = batch["image"]
#         masks_indices = batch["mask"]  # [B, H, W, 1] with float class indices
        
#         # Squeeze and convert to int
#         masks_indices = jnp.squeeze(masks_indices, axis=-1).astype(jnp.int32)  # [B, H, W]
        
#         # Forward pass - MoNet returns list of outputs + backbone
#         outputs = model_apply(
#             {"params": state.params, "batch_stats": state.batch_stats},
#             images,
#             train=False
#         )
        
#         # First output is mask prediction - logits [B, H, W, num_classes]
#         pred_logits_full = outputs[0]
        
#         # Convert logits to probabilities
#         pred_probs_full = jax.nn.softmax(pred_logits_full, axis=-1)
        
#         # Get predicted class indices for overlay
#         pred_indices_full = jnp.argmax(pred_logits_full, axis=-1)
        
#         # Denormalize image for visualization (approximate)
#         images_viz = (images - jnp.min(images)) / (jnp.max(images) - jnp.min(images) + 1e-7)
        
#         # Create overlays
#         overlay_gt = overlay_mask_on_image(images_viz, masks_indices, num_classes)
#         overlay_pred = overlay_mask_on_image(images_viz, pred_indices_full, num_classes)
        
#         # Convert masks to RGB for visualization (using same coloring)
#         colors = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
#         masks_viz = colors[masks_indices]  # [B, H, W, 3]
#         pred_viz = colors[pred_indices_full]  # [B, H, W, 3]
        
#         return {
#             "Input Image": images_viz,
#             "Ground Truth Mask": masks_viz,
#             "Predicted Mask": pred_viz,
#             "GT Overlay": overlay_gt,
#             "Pred Overlay": overlay_pred,
#         }
    
#     return viz_fn


def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py /path/to/nema_dish_tfrecords")
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    print(f"Data directory: {data_dir}")
    
    # 1. Create config
    config = ExperimentConfig(
        name='segmentation_minimal',
        dataset=DatasetConfig(),
        model=ModelConfig(),
        training=TrainingConfig(),
        seed=42
    )

    print(f"Config hash: {config.hash()}")
    
    # 2. Create experiment
    tracker = ExperimentTracker(experiments_dir='/data/experiments')
    run = tracker.create_experiment(name=config.name, config=config)
    print(f"Experiment: {run.experiment_id}")

    # 3. Load data
    train_iter, val_iter, train_batches, val_batches = create_segmentation_iterator(
        data_dir=data_dir,
        batch_size=config.dataset.batch_size,
        shuffle=config.dataset.shuffle,
        augment=config.dataset.augment,
        val_fraction=config.dataset.val_split,
        seed=config.dataset.split_seed,
    )

    # 4. Initialize model
    # Parse output configuration from old tuple format
    # config.model.outputs = ((3, False, 2),) means:
    #   3 classes, no sigmoid, 2 upsample steps
    num_classes, use_sigmoid, upsample_steps = config.model.outputs[0]

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
    train_step = create_train_step(num_classes)
    val_step = create_val_step()

    # 5. Training loop with tracking
    best_val_accuracy = 0.0

    for epoch in range(config.training.num_epochs):
        train_losses = []
        for _ in range(train_batches):
            batch = next(train_iter)
            state, metrics = train_step(state, batch)
            train_losses.append(float(metrics['loss']))

        # Validate
        val_accuracies = []
        for _ in range(val_batches):
            batch = next(val_iter)
            metrics = val_step(state, batch)
            val_accuracies.append(float(metrics['val_accuracy']))

        # Log metrics
        metrics = {
            'train_loss': sum(train_losses) / len(train_losses),
            'val_accuracy': sum(val_accuracies) / len(val_accuracies)
        }
        run.log_metrics(metrics, step=epoch)
        print(f"Epoch {epoch + 1}: train={metrics['train_loss']:.4f}, val={metrics['val_accuracy']:.4f}")

        # Save best checkpoint
        if metrics['val_accuracy'] > best_val_accuracy:
            print(f"New best validation accuracy: {metrics['val_accuracy']:.4f}")
            best_val_accuracy = metrics['val_accuracy']
            checkpoint_dir = run.output_dir / 'checkpoints' / 'best'
            save_checkpoint(state, str(checkpoint_dir))

    # 6. Finish experiment
    run.finish(status='completed')
    print(run.summary())

    # 7. Register model
    registry = ModelRegistry(registry_dir='/data/models')
    registry.register_model(
        model_name=config.name,
        experiment_id=run.experiment_id,
        config_hash=config.hash(),
        checkpoint_path=run.output_dir / 'checkpoints' / 'best',
        metrics={'best_val_accuracy': best_val_accuracy},
        timestamp=run.metadata.timestamp
    )


if __name__ == '__main__':
    main()