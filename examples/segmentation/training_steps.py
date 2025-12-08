from __future__ import annotations

import jax
import optax
import jax.numpy as jnp
from beagle.training import TrainState


def create_train_step():
    """Create JIT-compiled training step."""
    @jax.jit
    def train_step(state: TrainState, batch: dict):
        images = batch["image"]
        masks_full = batch["mask"]
        
        def loss_fn(params):
            # Forward pass - MoNet returns list of outputs + backbone
            outputs, updates = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                images,
                train=True,
                mutable=["batch_stats"],
            )
            
            # First output is mask logits (already at full resolution with sigmoid)
            mask_pred = outputs[0]
            
            targets_int = masks_full[..., 0].astype(jnp.int32)
            targets_one_hot = jax.nn.one_hot(targets_int, 3)
            
            # Compute safe softmax cross-entropy
            # This handles logits=-inf and labels=0 gracefully
            loss = optax.losses.safe_softmax_cross_entropy(mask_pred, targets_one_hot)
            loss = jnp.mean(loss)
            
            return loss, updates
        
        (loss, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        new_state = new_state.replace(batch_stats=updates["batch_stats"])
        
        return new_state, {"loss": loss}
    
    return train_step


def create_val_step():
    """Create visualization function."""
    @jax.jit
    def val_step(state: TrainState, batch: dict):
        images = batch["image"]
        masks_indices = batch["mask"]  # [B, H, W, 1] with float class indices
        
        # Squeeze and convert to int
        masks_indices = jnp.squeeze(masks_indices, axis=-1).astype(jnp.int32)  # [B, H, W]
        
        # Forward pass - MoNet returns list of outputs + backbone
        outputs = state.apply_fn(
            {"params": state.params, "batch_stats": state.batch_stats},
            images,
            train=False
        )
        
        # First output is mask prediction - logits [B, H, W, num_classes]
        pred_logits_full = outputs[0]
        
        # Convert logits to probabilities
        pred_probs_full = jax.nn.softmax(pred_logits_full, axis=-1)
        
        # Get predicted class indices for overlay
        pred_indices_full = jnp.argmax(pred_logits_full, axis=-1)
        
        # Calculate accuracy
        accuracy = jnp.mean(pred_indices_full == masks_indices)
        
        # Vectorized IoU calculation
        num_classes = pred_logits_full.shape[-1]
        pred_one_hot = jax.nn.one_hot(pred_indices_full, num_classes)  # [B, H, W, C]
        gt_one_hot = jax.nn.one_hot(masks_indices, num_classes)  # [B, H, W, C]
        
        intersection = jnp.sum(pred_one_hot * gt_one_hot, axis=(0, 1, 2))  # [C]
        pred_area = jnp.sum(pred_one_hot, axis=(0, 1, 2))  # [C]
        gt_area = jnp.sum(gt_one_hot, axis=(0, 1, 2))  # [C]
        union = pred_area + gt_area - intersection  # [C]
        
        iou_per_class = intersection / (union + 1e-8)
        mean_iou = jnp.mean(iou_per_class)
        
        return {
            'val_accuracy': accuracy,
            'val_mean_iou': mean_iou,
            'val_iou_per_class': iou_per_class,
            'pred_indices_full': pred_indices_full,
        }
    
    return val_step
