"""Segmentation training using Beagle's new Graph API.

This is a rewrite of train_disk.py showcasing the Graph API's power:
- Clean, composable architecture
- Automatic mutable collection handling (batch_stats)
- Easy multi-stage training (freeze/unfreeze)
- Generic training utilities
- Per-node checkpointing

Compare this to train_disk.py to see the difference!
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# NEW: Import Graph API
from beagle.graph import (
    ComputeNode,
    ComputeGraph,
    GraphState,
    create_train_step,
    create_eval_step,
)
from beagle.checkpoint import save_graph_state, save_node

# Existing imports
from beagle.network.hrnet import HRNetBackbone, SegmentationHead
from beagle.experiments import ExperimentConfig, ExperimentTracker, ModelRegistry

from data_loader_disk import create_segmentation_iterator
from configs import DatasetConfig, ModelConfig, TrainingConfig
from visualize_segmentation import create_overlay, DEFAULT_COLORS


# ============================================================================
# LOSS & METRICS (graph-compatible)
# ============================================================================

def create_segmentation_loss_fn(num_classes: int):
    """Create loss function for segmentation.

    Args:
        num_classes: Number of segmentation classes

    Returns:
        loss_fn: Function(outputs, batch) -> scalar loss
    """
    def loss_fn(outputs, batch):
        """Compute segmentation loss.

        Args:
            outputs: Dict with 'logits' key from graph
            batch: Dict with 'image' and 'mask' keys
        """
        # MoNet outputs multiple things, first is the segmentation logits
        # In graph API, we'll name it 'seg_logits'
        mask_pred = outputs['seg_logits']  # [B, H, W, num_classes]
        masks_full = batch['mask']  # [B, H, W, 1]

        # Convert to one-hot
        targets_int = masks_full[..., 0].astype(jnp.int32)
        targets_one_hot = jax.nn.one_hot(targets_int, num_classes)

        # Safe softmax cross-entropy
        loss = optax.losses.safe_softmax_cross_entropy(mask_pred, targets_one_hot)
        return jnp.mean(loss)

    return loss_fn


def create_segmentation_metrics_fn(num_classes: int):
    """Create metrics function for segmentation.

    Args:
        num_classes: Number of segmentation classes

    Returns:
        metrics_fn: Function(outputs, batch) -> dict of metrics
    """
    def metrics_fn(outputs, batch):
        """Compute segmentation metrics.

        Args:
            outputs: Dict with 'seg_logits' key
            batch: Dict with 'mask' key
        """
        mask_pred = outputs['seg_logits']  # [B, H, W, num_classes]
        masks_indices = batch['mask'][..., 0].astype(jnp.int32)  # [B, H, W]

        # Predicted class indices
        pred_indices = jnp.argmax(mask_pred, axis=-1)  # [B, H, W]

        # Accuracy
        accuracy = jnp.mean(pred_indices == masks_indices)

        # IoU per class
        pred_one_hot = jax.nn.one_hot(pred_indices, num_classes)
        gt_one_hot = jax.nn.one_hot(masks_indices, num_classes)

        intersection = jnp.sum(pred_one_hot * gt_one_hot, axis=(0, 1, 2))
        pred_area = jnp.sum(pred_one_hot, axis=(0, 1, 2))
        gt_area = jnp.sum(gt_one_hot, axis=(0, 1, 2))
        union = pred_area + gt_area - intersection

        iou_per_class = intersection / (union + 1e-8)
        mean_iou = jnp.mean(iou_per_class)

        return {
            'accuracy': accuracy,
            'mean_iou': mean_iou,
        }

    return metrics_fn


# ============================================================================
# WRAPPER: Make MoNet output a dict instead of list
# ============================================================================

# MoNet has been refactored into HRNetBackbone + SegmentationHead
# No wrapper needed - both classes natively return dicts!


# ============================================================================
# VISUALIZATION
# ============================================================================

def save_batch_visualizations(
    state: GraphState,
    batch: dict,
    num_classes: int,
    output_dir: Path,
    epoch: int,
    max_samples: int = 4,
    alpha: float = 0.5,
):
    """Save visualization overlays for a batch during training.

    Args:
        state: Current GraphState
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

    # Run inference using GraphState
    outputs = state({"image": images}, train=False)

    # Get predictions (using dict-based I/O)
    pred_logits = outputs['logits']  # [B, H, W, num_classes]
    pred_classes = jnp.argmax(pred_logits, axis=-1)  # [B, H, W]

    # Convert to numpy
    images_np = np.array(images)
    masks_gt_np = np.array(masks_gt)
    pred_classes_np = np.array(pred_classes)

    # Get colors
    colors = DEFAULT_COLORS[:num_classes]

    # Process each sample in batch
    batch_size = min(images_np.shape[0], max_samples)
    for idx in range(batch_size):
        img = images_np[idx]  # [H, W, 1] in range [-1, 1]
        gt = masks_gt_np[idx, :, :, 0].astype(np.uint8)  # [H, W]
        pred = pred_classes_np[idx].astype(np.uint8)  # [H, W]

        # Create overlays
        gt_overlay = create_overlay(img, gt, num_classes, colors, alpha)
        pred_overlay = create_overlay(img, pred, num_classes, colors, alpha)

        # Denormalize image
        img_rgb = ((img + 1.0) * 127.5).astype(np.uint8)
        img_rgb = np.repeat(img_rgb, 3, axis=2)

        # Create figure
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


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python train_graph.py /path/to/nema_dish_tfrecords")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    print(f"Data directory: {data_dir}")

    # 1. Load data to determine num_classes
    train_iter, num_classes = create_segmentation_iterator(
        dataset_path=data_dir,
        augmentation_multiplier=20
    )
    print(f"Number of classes: {num_classes}")

    # 2. Create config
    config = ExperimentConfig(
        name='segmentation_graph_api',
        dataset=DatasetConfig(),
        model=ModelConfig(
            outputs=((num_classes, False, 2),)
        ),
        training=TrainingConfig(),
        seed=42
    )

    print(f"Config hash: {config.hash()}")

    # 3. Create experiment
    tracker = ExperimentTracker(experiments_dir='/data/experiments_nema_segmentation')
    run = tracker.create_experiment(name=config.name, config=config)
    print(f"Experiment: {run.experiment_id}")

    # ========================================================================
    # 4. CREATE COMPUTE GRAPH (the new way!)
    # ========================================================================

    print("\n" + "="*80)
    print("CREATING COMPUTE GRAPH")
    print("="*80)

    # Parse output configuration from old tuple format
    # config.model.outputs = ((3, False, 2),) means:
    #   3 classes, no sigmoid, 2 upsample steps
    num_classes, use_sigmoid, upsample_steps = config.model.outputs[0]

    # Create backbone node
    backbone_node = ComputeNode(
        name="backbone",
        module=HRNetBackbone(
            num_stages=config.model.num_stages,
            features=config.model.features,
            target_res=config.model.target_res,
        ),
        inputs=["image"],
        outputs=["features"],
        trainable=config.model.train_backbone
    )

    # Create segmentation head node
    head_node = ComputeNode(
        name="head",
        module=SegmentationHead(
            num_classes=num_classes,
            features=config.model.features,
            upsample_steps=upsample_steps,
            use_sigmoid=use_sigmoid,
            output_key='logits'
        ),
        inputs=["features"],
        outputs=["logits"],
        trainable=True
    )

    # Create graph with backbone + head composition
    graph = ComputeGraph(
        nodes={"backbone": backbone_node, "head": head_node}
    )

    print(f"Graph created: {graph}")
    print(f"Trainable nodes: {graph.trainable_nodes()}")

    # Initialize
    key = random.key(config.seed)
    key, init_key = random.split(key)
    dummy_batch = {"image": jnp.ones((1, config.model.input_size, config.model.input_size, 1))}

    variables = graph.init(init_key, dummy_batch)

    print(f"Initialized variables:")
    for node_name, node_vars in variables.items():
        print(f"  {node_name}: collections = {list(node_vars.keys())}")

    # Create GraphState
    tx = optax.adamw(config.training.learning_rate)
    state = GraphState.create(
        graph=graph,
        variables=variables,
        tx=tx
    )

    print(f"\n{state}")

    # ========================================================================
    # 5. CREATE TRAINING & EVAL STEPS (generic!)
    # ========================================================================

    print("\n" + "="*80)
    print("CREATING TRAINING FUNCTIONS")
    print("="*80)

    loss_fn = create_segmentation_loss_fn(num_classes)
    metrics_fn = create_segmentation_metrics_fn(num_classes)

    # Generic training step works for ANY graph!
    train_step = create_train_step(loss_fn, aux_metrics_fn=metrics_fn)
    eval_step = create_eval_step(loss_fn, metrics_fn=metrics_fn)

    print("✓ Training step created (JIT-compiled)")
    print("✓ Eval step created (JIT-compiled)")
    print("✓ Handles batch_stats automatically!")

    # ========================================================================
    # 6. TRAINING LOOP
    # ========================================================================

    print("\n" + "="*80)
    print("TRAINING")
    print("="*80 + "\n")

    viz_dir = run.output_dir / 'visualizations'
    viz_frequency = 5

    best_mean_iou = 0.0

    for epoch in range(config.training.num_epochs):
        # Training
        train_losses = []
        train_accs = []
        train_ious = []
        last_batch = None

        for _ in range(100):  # 100 batches per epoch
            batch = next(train_iter)
            state, metrics = train_step(state, batch)

            train_losses.append(float(metrics['loss']))
            train_accs.append(float(metrics['accuracy']))
            train_ious.append(float(metrics['mean_iou']))
            last_batch = batch

        avg_loss = np.mean(train_losses)
        avg_acc = np.mean(train_accs)
        avg_iou = np.mean(train_ious)

        print(f"Epoch {epoch + 1:3d}/{config.training.num_epochs} | "
              f"loss: {avg_loss:.4f} | "
              f"acc: {avg_acc:.4f} | "
              f"IoU: {avg_iou:.4f}")

        # Track best model
        if avg_iou > best_mean_iou:
            best_mean_iou = avg_iou

        # Visualizations
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

    # ========================================================================
    # 7. SAVE CHECKPOINT (the new way!)
    # ========================================================================

    print("\n" + "="*80)
    print("SAVING CHECKPOINT")
    print("="*80)

    checkpoint_dir = run.output_dir / 'checkpoints' / 'best'

    # NEW: Save full graph state with metadata
    save_graph_state(state, str(checkpoint_dir))

    print(f"✓ Saved full graph state to: {checkpoint_dir}")
    print(f"  Structure:")
    print(f"    {checkpoint_dir}/")
    print(f"      ├─ graph_config.json (graph structure + metadata)")
    print(f"      ├─ optimizer.msgpack (optimizer state)")
    print(f"      └─ node_segmentation/ (per-node checkpoint)")
    print(f"          ├─ metadata.json (shapes, dtypes, collections)")
    print(f"          └─ variables.msgpack (params, batch_stats)")

    # BONUS: Save just the segmentation node for transfer learning!
    node_only_dir = run.output_dir / 'checkpoints' / 'segmentation_node'
    save_node("segmentation", state.variables["segmentation"], str(node_only_dir))
    print(f"\n✓ Also saved segmentation node alone to: {node_only_dir}")
    print(f"  Can be loaded independently for transfer learning!")

    # ========================================================================
    # 8. FINISH
    # ========================================================================

    run.finish(status='completed')
    print(f"\n{run.summary()}")

    # Register model
    registry = ModelRegistry(registry_dir='/data/models_nema_segmentation')
    registry.register_model(
        model_name=config.name,
        experiment_id=run.experiment_id,
        config_hash=config.hash(),
        checkpoint_path=checkpoint_dir,
        metrics={'best_mean_iou': float(best_mean_iou)},
        timestamp=run.metadata.timestamp
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best mean IoU: {best_mean_iou:.4f}")
    print(f"Experiment ID: {run.experiment_id}")
    print(f"Checkpoint: {checkpoint_dir}")


# ============================================================================
# BONUS: MULTI-STAGE TRAINING EXAMPLE
# ============================================================================

def multi_stage_training_example():
    """Example showing how easy multi-stage training is with Graph API.

    This demonstrates:
    1. Training the full model
    2. Freezing the backbone
    3. Training only the segmentation head
    4. Unfreezing everything for fine-tuning
    """

    # Create segmentation graph (same as above)
    segmentation_node = ComputeNode(...)
    graph = ComputeGraph(nodes={"segmentation": segmentation_node})

    # Initialize
    state = GraphState.create(graph, variables, tx)

    # ========================================================================
    # STAGE 1: Train full model
    # ========================================================================
    print("Stage 1: Training full model...")
    for epoch in range(10):
        # ... train ...
        pass

    # ========================================================================
    # STAGE 2: Freeze backbone, train only head
    # ========================================================================
    print("Stage 2: Freezing backbone, training only head...")

    # In MoNet, you could split it into backbone + head nodes
    # For simplicity, we'll show the concept:
    # (In reality, you'd create separate nodes for backbone and head)

    # freeze_part_of_model(state, "backbone")
    # ... train only head ...

    # ========================================================================
    # STAGE 3: Unfreeze everything for fine-tuning
    # ========================================================================
    print("Stage 3: Unfreezing all for fine-tuning...")
    graph.unfreeze_all()
    state = state.unfreeze_node("segmentation")

    # ... fine-tune ...

    print("Multi-stage training complete!")


if __name__ == '__main__':
    main()
