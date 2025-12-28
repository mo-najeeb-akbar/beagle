"""Train MoNet for nema dish mask prediction at full resolution.

Run: make run CMD='python examples/nema_dish/train.py /data/nema_dish_tfrecords'
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import optax

from beagle.network.hrnet import MoNet
from beagle.training import TrainState, train_loop, save_config, save_metrics_history
from beagle.visualization import create_viz_callback, VizConfig

from data_loader import create_nema_dish_iterator


CONFIG = {
    "learning_rate": 0.001,
    "num_epochs": 10,
    "batch_size": 8,
    "num_stages": 3,
    "features": 32,
    "target_res": 1.0,
    "input_size": 512,
    "upsample_to_full_res": True,
    "train_backbone": True,
    "shuffle": True,
    "augment": True,
}


def create_train_step():
    """Create JIT-compiled training step."""
    @jax.jit
    def train_step(state: TrainState, batch: dict, rng_key):
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


def overlay_mask_on_image(
    image: jnp.ndarray, mask: jnp.ndarray, num_classes: int = 3, alpha: float = 0.5
) -> jnp.ndarray:
    """Overlay colored mask on grayscale image (pure function).
    
    Args:
        image: Grayscale image [B, H, W, 1]
        mask: Class indices [B, H, W] or probabilities [B, H, W, num_classes]
        num_classes: Number of classes
        alpha: Transparency factor
        
    Returns:
        RGB overlay image [B, H, W, 3]
    """
    batch, height, width, _ = image.shape
    
    # Convert grayscale to RGB
    image_rgb = jnp.concatenate([image, image, image], axis=-1)
    
    # If mask is probabilities, convert to class indices
    if mask.ndim == 4:  # [B, H, W, num_classes]
        mask_indices = jnp.argmax(mask, axis=-1)  # [B, H, W]
    else:  # Already indices [B, H, W]
        mask_indices = mask
    
    # Define colors for each class (RGB)
    # Class 0: black (background), Class 1: red (glass), Class 2: blue (white tag)
    colors = jnp.array([
        [0.0, 0.0, 0.0],  # Class 0: black/background
        [1.0, 0.0, 0.0],  # Class 1: red
        [0.0, 0.0, 1.0],  # Class 2: blue
    ])
    
    # Create colored mask by indexing into colors
    mask_rgb = colors[mask_indices]  # [B, H, W, 3]
    
    # Blend (only where mask is non-background)
    is_background = (mask_indices == 0)[..., None]  # [B, H, W, 1]
    overlay = jnp.where(is_background, image_rgb, alpha * mask_rgb + (1 - alpha) * image_rgb)
    
    return jnp.clip(overlay, 0, 1)


def create_viz_fn(model_apply, num_classes: int = 3):
    """Create visualization function."""
    @jax.jit
    def viz_fn(state, batch, rng_key):
        images = batch["image"]
        masks_indices = batch["mask"]  # [B, H, W, 1] with float class indices
        
        # Squeeze and convert to int
        masks_indices = jnp.squeeze(masks_indices, axis=-1).astype(jnp.int32)  # [B, H, W]
        
        # Forward pass - MoNet returns list of outputs + backbone
        outputs = model_apply(
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
        
        # Denormalize image for visualization (approximate)
        images_viz = (images - jnp.min(images)) / (jnp.max(images) - jnp.min(images) + 1e-7)
        
        # Create overlays
        overlay_gt = overlay_mask_on_image(images_viz, masks_indices, num_classes)
        overlay_pred = overlay_mask_on_image(images_viz, pred_indices_full, num_classes)
        
        # Convert masks to RGB for visualization (using same coloring)
        colors = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        masks_viz = colors[masks_indices]  # [B, H, W, 3]
        pred_viz = colors[pred_indices_full]  # [B, H, W, 3]
        
        return {
            "Input Image": images_viz,
            "Ground Truth Mask": masks_viz,
            "Predicted Mask": pred_viz,
            "GT Overlay": overlay_gt,
            "Pred Overlay": overlay_pred,
        }
    
    return viz_fn



def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py /path/to/nema_dish_tfrecords")
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        sys.exit(1)
    
    # Setup experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"/data/experiments/nema_dish_{timestamp}"
    save_config(CONFIG, exp_dir)
    
    print(f"Experiment: {exp_dir}")
    print(f"JAX devices: {jax.devices()}")
    
    # Configure outputs: (num_outputs, use_sigmoid, upsample_steps)
    # Backbone at target_res=1.0 is 1/4 of input (after 2x initial downsample)
    # To reach full resolution, need 2 upsample steps
    if CONFIG["upsample_to_full_res"]:
        outputs = [(3, False, 2)]  # 1 channel, sigmoid, 2x upsample
        output_res_str = "full resolution (512x512)"
    else:
        outputs = [(3, False)]  # 1 channel, sigmoid, no upsample (128x128)
        output_res_str = "1/4 resolution (128x128)"
    
    # Initialize multi-output model
    model = MoNet(
        num_stages=CONFIG["num_stages"],
        features=CONFIG["features"],
        target_res=CONFIG["target_res"],
        train_bb=CONFIG["train_backbone"],
        outputs=outputs,
    )
    
    key = random.key(42)
    key, init_key = random.split(key)
    
    # Initialize with dummy input
    dummy = jnp.ones((1, CONFIG["input_size"], CONFIG["input_size"], 1))
    variables = model.init(init_key, dummy, train=False)
    params, batch_stats = variables['params'], variables['batch_stats']
    
    print(f"Model output: {output_res_str}")
    
    # Create training state
    tx = optax.adamw(CONFIG["learning_rate"])
    state = TrainState.create(apply_fn=model.apply, params=params, batch_stats=batch_stats, tx=tx)
    
    # Load data
    print("Loading nema dish dataset...")
    train_iter, train_batches = create_nema_dish_iterator(
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
    print(f"Backbone: {CONFIG['num_stages']} stages, {CONFIG['features']} features")
    print(f"Train backbone: {CONFIG['train_backbone']}")
    
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

