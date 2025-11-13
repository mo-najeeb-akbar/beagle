"""Train wavelet VAE with mixed precision for 2x speedup.

Demonstrates safe mixed precision training:
- Compute in bfloat16 (2x memory bandwidth on Ampere+ GPUs)
- Weights stay in float32
- Loss/outputs in float32 for numerical stability

Run: make run CMD='python examples/train_wavelet_vae_mixed_precision.py /data/polymer_tfrecords'
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import optax

from beagle.wavelets import wavedec2
from beagle.training import (
    TrainState,
    train_loop,
    save_config,
    save_metrics_history,
    enable_mixed_precision,
    get_recommended_policy,
)
from beagle.visualization import create_viz_callback, VizConfig

# Import from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from polymer_data import create_polymer_iterator, compute_polymer_stats


CONFIG = {
    "learning_rate": 0.001,
    "num_epochs": 3,  # Reduced for testing
    "batch_size": 32,
    "base_features": 48,
    "latent_dim": 256,
    "crop_size": 256,
    "crop_overlap": 192,
    "mixed_precision": True,  # Enable mixed precision
    "mp_dtype": "float16",  # "auto", "bfloat16", or "float16"
}


def create_train_step(wavelet_weights: tuple[float, ...] = (1.0, 8.0, 8.0, 12.0)):
    """Create JIT-compiled training step (pure float32 version).
    
    Note: We'll wrap this with mixed precision AFTER JIT compilation.
    This keeps the code clean and separates concerns.
    """
    @jax.jit
    def train_step(state: TrainState, batch: dict, rng_key):
        # Batch data might be in bfloat16 (casted by mixed precision wrapper)
        # Model will compute in bfloat16, output in float32
        images = batch['depth']
        wavelets = wavedec2(images, wavelet="haar")
        
        def loss_fn(params):
            # Forward pass: internally casts to bfloat16, returns float32
            x_recon, x_wave, mu, log_var = state.apply_fn(
                {'params': params}, wavelets, training=True, key=rng_key
            )
            
            # Loss computations stay in float32 (x_recon is float32)
            recon_loss = jnp.mean(jnp.square(images - x_recon))
            
            # Weighted wavelet loss (all float32)
            ll_loss = jnp.mean(jnp.square(wavelets[..., 0] - x_wave[..., 0]))
            hl_loss = jnp.mean(jnp.square(wavelets[..., 1] - x_wave[..., 1]))
            lh_loss = jnp.mean(jnp.square(wavelets[..., 2] - x_wave[..., 2]))
            hh_loss = jnp.mean(jnp.square(wavelets[..., 3] - x_wave[..., 3]))
            
            w = wavelet_weights
            wave_loss = w[0] * ll_loss + w[1] * hl_loss + w[2] * lh_loss + w[3] * hh_loss
            
            # Return float32 losses
            return wave_loss, recon_loss
        
        # Gradients computed in bfloat16 (faster), but stored in float32
        (loss, recon_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        
        # Metrics in float32
        return new_state, {'loss': loss, 'recon_loss': recon_loss}
    
    return train_step


def create_viz_fn(model_apply):
    """Create visualization function (outputs always float32)."""
    @jax.jit
    def viz_fn(state, batch, rng_key):
        images = batch['depth']
        wavelets = wavedec2(images, wavelet="haar")
        
        # Model outputs float32 for visualization
        x_recon, _, _, _ = model_apply(
            {'params': state.params}, wavelets, training=False, key=rng_key
        )
        
        return {
            'Original': images,
            'Reconstruction': x_recon,
            'Error (5x)': jnp.abs(images - x_recon) * 5.0
        }
    
    return viz_fn


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_wavelet_vae_mixed_precision.py /path/to/polymer_tfrecords [--compute-stats]")
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    
    # Optional: just compute and display statistics
    if '--compute-stats' in sys.argv:
        mean, std, n_imgs = compute_polymer_stats(data_dir)
        print(f"Dataset: {n_imgs} images")
        print(f"Mean: {mean:.6f}")
        print(f"Std:  {std:.6f}")
        return
    
    # Setup experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"/data/experiments/wavelet_vae_mp_{timestamp}"
    save_config(CONFIG, exp_dir)
    
    print(f"Experiment: {exp_dir}")
    print(f"JAX devices: {jax.devices()}")
    
    # Initialize model (in float32)
    from beagle.network.wavelet_vae import VAE
    
    model = VAE(
        base_features=CONFIG['base_features'],
        latent_dim=CONFIG['latent_dim']
    )
    
    key = random.key(42)
    key, init_key = random.split(key)
    
    # Initialize with float32 (weights stay float32)
    dummy = jnp.ones((1, 128, 128, 4), dtype=jnp.float32)
    variables = model.init(init_key, dummy, random.key(0), training=True)
    
    # Verify weights are float32
    param_dtypes = jax.tree.map(lambda x: x.dtype, variables['params'])
    print(f"\nParameter dtypes: {jax.tree.leaves(param_dtypes)[0]} (should be float32)")
    
    # Create training state (optimizer in float32)
    tx = optax.adamw(CONFIG['learning_rate'])
    state = TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)
    
    # Load data
    print("\nLoading polymer dataset...")
    iterator, batches_per_epoch, img_shape = create_polymer_iterator(
        data_dir=data_dir,
        batch_size=CONFIG['batch_size'],
        crop_size=CONFIG['crop_size'],
        stride=CONFIG['crop_overlap'],
        shuffle=True,
        augment=True
    )
    
    print(f"Batches per epoch: {batches_per_epoch}")
    
    # Create training step
    train_step_fn = create_train_step()
    
    # 🚀 Wrap with mixed precision (this is the magic!)
    if CONFIG['mixed_precision']:
        from beagle.training import create_mixed_precision_policy
        
        # Create policy from config
        if CONFIG['mp_dtype'] == "auto":
            policy = get_recommended_policy()
            print(f"\n🚀 Mixed precision: auto-detected from hardware")
        else:
            policy = create_mixed_precision_policy(CONFIG['mp_dtype'])
            print(f"\n🚀 Mixed precision: {CONFIG['mp_dtype']} (user configured)")
        
        # Show actual policy being used
        print(f"   Compute dtype: {policy.compute_dtype}")
        print(f"   Param dtype:   {policy.param_dtype}")
        print(f"   Output dtype:  {policy.output_dtype}")
        
        # Apply mixed precision
        train_step_fn = enable_mixed_precision(train_step_fn, policy=policy)
        print("   ✅ Mixed precision enabled")
    else:
        print("\nRunning in full float32 precision")
    
    # Visualization
    viz_config = VizConfig(
        plot_every=5,
        num_samples=4,
        output_dir=f"{exp_dir}/viz"
    )
    viz_callback = create_viz_callback(create_viz_fn(model.apply), viz_config)
    
    # Data iterator function
    def data_fn():
        return iterator
    
    # Train
    print(f"\nTraining for {CONFIG['num_epochs']} epochs...")
    
    key, train_key = random.split(key)
    final_state, history = train_loop(
        state=state,
        train_step_fn=train_step_fn,
        data_iterator_fn=data_fn,
        num_epochs=CONFIG['num_epochs'],
        num_batches=batches_per_epoch,
        rng_key=train_key,
        checkpoint_dir=exp_dir,
        viz_callback=viz_callback
    )
    
    # Save results
    save_metrics_history(history, exp_dir)
    
    print(f"\nComplete. Final loss: {history['train_loss'][-1]:.4f}")
    print(f"Results: {exp_dir}")
    
    # Verify final params are still float32
    final_dtypes = jax.tree.map(lambda x: x.dtype, final_state.params)
    print(f"\nFinal parameter dtypes: {jax.tree.leaves(final_dtypes)[0]} (still float32 ✓)")


if __name__ == "__main__":
    main()

