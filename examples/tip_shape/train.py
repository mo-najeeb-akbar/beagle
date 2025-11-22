"""Train HRNet for root tip mask prediction at 1/4 resolution.

Run: make run CMD='python examples/tip_shape/train_hrnet.py /data/root_tip_tfrecords'
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import optax
from jax import image as jax_image

from beagle.training import TrainState, train_loop, save_config, save_metrics_history
from beagle.visualization import create_viz_callback, VizConfig

from data_loader import create_root_tip_iterator
from mask_net import MaskNet


CONFIG = {
    "learning_rate": 0.001,
    "num_epochs": 10,
    "batch_size": 8,
    "num_stages": 3,
    "features": 32,
    "target_res": 1.0,  # 1.0 = 128x128 (after 2x initial downsampling from 512)
    "input_size": 512,
    "shuffle": True,
    "augment": True,
}


def binary_cross_entropy_loss(
    predictions: jnp.ndarray, targets: jnp.ndarray, eps: float = 1e-7
) -> jnp.ndarray:
    """Binary cross-entropy loss (pure function).
    
    Args:
        predictions: Predicted probabilities after sigmoid [0, 1]
        targets: Ground truth binary masks [0, 1]
        eps: Small value for numerical stability
        
    Returns:
        Scalar loss value
    """
    predictions = jnp.clip(predictions, eps, 1 - eps)
    loss = -(targets * jnp.log(predictions) + (1 - targets) * jnp.log(1 - predictions))
    return jnp.mean(loss)


def resize_mask(mask: jnp.ndarray, target_size: int) -> jnp.ndarray:
    """Resize mask to target resolution (pure function)."""
    batch, height, width, channels = mask.shape
    return jax_image.resize(
        mask,
        shape=(batch, target_size, target_size, channels),
        method="nearest",
    )


def create_train_step():
    """Create JIT-compiled training step."""
    @jax.jit
    def train_step(state: TrainState, batch: dict, rng_key):
        images = batch["image"]
        masks_full = batch["mask"]
        
        # Downsample masks to 1/4 resolution (128x128)
        masks_quarter = resize_mask(masks_full, CONFIG["input_size"] // 4)
        
        def loss_fn(params):
            # Forward pass - HRNet outputs at target resolution
            predictions, updates = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                images,
                train=True,
                mutable=["batch_stats"],
            )
            
            # Apply sigmoid to get probabilities
            predictions_prob = jax.nn.sigmoid(predictions)
            
            # Binary cross-entropy loss
            loss = binary_cross_entropy_loss(predictions_prob, masks_quarter)
            
            return loss, updates
        
        (loss, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        new_state = new_state.replace(batch_stats=updates["batch_stats"])
        
        return new_state, {"loss": loss}
    
    return train_step


def overlay_mask_on_image(
    image: jnp.ndarray, mask: jnp.ndarray, alpha: float = 0.5
) -> jnp.ndarray:
    """Overlay colored mask on grayscale image (pure function).
    
    Args:
        image: Grayscale image [B, H, W, 1]
        mask: Binary mask [B, H, W, 1]
        alpha: Transparency factor
        
    Returns:
        RGB overlay image [B, H, W, 3]
    """
    batch, height, width, _ = image.shape
    
    # Convert grayscale to RGB
    image_rgb = jnp.concatenate([image, image, image], axis=-1)
    
    # Create red mask overlay
    mask_rgb = jnp.concatenate([mask, jnp.zeros_like(mask), jnp.zeros_like(mask)], axis=-1)
    
    # Blend
    overlay = alpha * mask_rgb + (1 - alpha) * image_rgb
    
    return jnp.clip(overlay, 0, 1)


def create_viz_fn(model_apply):
    """Create visualization function."""
    @jax.jit
    def viz_fn(state, batch, rng_key):
        images = batch["image"]
        masks_full = batch["mask"]
        
        # Forward pass
        predictions = model_apply({"params": state.params, "batch_stats": state.batch_stats}, images, train=False)
        predictions_prob = jax.nn.sigmoid(predictions)
        
        # Resize predicted mask to full resolution for visualization
        pred_mask_full = resize_mask(predictions_prob, CONFIG["input_size"])
        
        # Denormalize image for visualization (approximate)
        # Assuming z-score normalization, rescale to [0, 1]
        images_viz = (images - jnp.min(images)) / (jnp.max(images) - jnp.min(images) + 1e-7)
        
        # Create overlays
        overlay_gt = overlay_mask_on_image(images_viz, masks_full)
        overlay_pred = overlay_mask_on_image(images_viz, pred_mask_full)
        
        return {
            "Input Image": images_viz,
            "Ground Truth Mask": masks_full,
            "Predicted Mask": pred_mask_full,
            "GT Overlay": overlay_gt,
            "Pred Overlay": overlay_pred,
        }
    
    return viz_fn


def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py /path/to/root_tip_tfrecords")
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        sys.exit(1)
    
    # Setup experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"/data/experiments/hrnet_masks_{timestamp}"
    save_config(CONFIG, exp_dir)
    
    print(f"Experiment: {exp_dir}")
    print(f"JAX devices: {jax.devices()}")
    
    # Initialize mask prediction model
    model = MaskNet(
        num_stages=CONFIG["num_stages"],
        features=CONFIG["features"],
        target_res=CONFIG["target_res"],
    )
    
    key = random.key(42)
    key, init_key = random.split(key)
    
    # Initialize with dummy input
    dummy = jnp.ones((1, CONFIG["input_size"], CONFIG["input_size"], 1))
    variables = model.init(init_key, dummy, train=False)
    params, batch_stats = variables['params'], variables['batch_stats']
    
    # Create training state
    tx = optax.adamw(CONFIG["learning_rate"])
    state = TrainState.create(apply_fn=model.apply, params=params, batch_stats=batch_stats, tx=tx)
    
    # Load data
    print("Loading root tip dataset...")
    train_iter, train_batches = create_root_tip_iterator(
        data_dir=data_dir,
        batch_size=CONFIG["batch_size"],
        shuffle=CONFIG["shuffle"],
        augment=CONFIG["augment"],
    )
    
    print(f"Train batches per epoch: {train_batches}")
    
    # Training step
    train_step_fn = create_train_step()
    
    # Get a batch for visualization
    viz_batch = next(train_iter)
    
    # Setup visualization
    viz_config = VizConfig(
        plot_every=5, num_samples=4, output_dir=f"{exp_dir}/viz"
    )
    viz_callback = create_viz_callback(create_viz_fn(model.apply), viz_config)
    
    # Training data iterator function
    def data_fn():
        return train_iter
    
    # Train
    print(f"Training for {CONFIG['num_epochs']} epochs...")
    print(f"Input size: {CONFIG['input_size']}x{CONFIG['input_size']}")
    print(f"Output size: {CONFIG['input_size']//4}x{CONFIG['input_size']//4} (1/4 resolution)")
    
    key, train_key = random.split(key)
    final_state, history = train_loop(
        state=state,
        train_step_fn=train_step_fn,
        data_iterator_fn=data_fn,
        num_epochs=CONFIG["num_epochs"],
        num_batches=train_batches,
        rng_key=train_key,
        checkpoint_dir=exp_dir,
        viz_callback=viz_callback,
        viz_batch=viz_batch,
    )
    
    # Save results
    save_metrics_history(history, exp_dir)
    
    print(f"Complete. Final loss: {history['train_loss'][-1]:.4f}")
    print(f"Results: {exp_dir}")


if __name__ == "__main__":
    main()

