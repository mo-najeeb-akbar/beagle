"""Train wavelet VAE using beagle library.

Run: make run CMD='python examples/train_wavelet_vae.py /data/polymer_tfrecords'
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
from beagle.training import TrainState, train_loop, save_config, save_metrics_history
from beagle.visualization import create_viz_callback, VizConfig

# Import from parent directory
from data_loader import create_nema_vae_iterator


CONFIG = {
    "learning_rate": 0.0001,
    "num_epochs": 50,  
    "batch_size": 64,
    "base_features": 64,
    "latent_dim": 128,
}


def create_train_step(wavelet_weights: tuple[float, ...] = (10.0, 8.0, 8.0, 12.0)):
    """Create JIT-compiled training step."""
    @jax.jit
    def train_step(state: TrainState, batch: dict, rng_key):
        images = batch['image']
        
        def loss_fn(params):
            x_recon, x_wave, mu, log_var = state.apply_fn(
                {'params': params}, images, training=True, key=rng_key
            )
            wavelets = wavedec2(images, wavelet="haar")
            recon_loss = jnp.mean(jnp.square(images - x_recon))
            
            # Weighted wavelet loss
            ll_loss = jnp.mean(jnp.square(wavelets[..., 0] - x_wave[..., 0]))
            hl_loss = jnp.mean(jnp.square(wavelets[..., 1] - x_wave[..., 1]))
            lh_loss = jnp.mean(jnp.square(wavelets[..., 2] - x_wave[..., 2]))
            hh_loss = jnp.mean(jnp.square(wavelets[..., 3] - x_wave[..., 3]))
            
            w = wavelet_weights
            wave_loss = w[0] * ll_loss + w[1] * hl_loss + w[2] * lh_loss + w[3] * hh_loss
            
            return wave_loss, recon_loss
        
        (loss, recon_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        
        return new_state, {'loss': loss, 'recon_loss': recon_loss}
    
    return train_step


def create_viz_fn(model_apply):
    """Create visualization function."""
    @jax.jit
    def viz_fn(state, batch, rng_key):
        images = batch['image']
        
        x_recon, _, _, _ = model_apply(
            {'params': state.params}, images, training=False, key=rng_key
        )
        
        return {
            'Original': images,
            'Reconstruction': x_recon,
            'Error (5x)': jnp.abs(images - x_recon) * 5.0
        }
    
    return viz_fn


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_wavelet_vae.py /path/to/nema_vae_tfrecords [--fast]")
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    
    # Fast mode: no logging, metrics, or visualization
    fast_mode = '--fast' in sys.argv
    
    # Setup experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"/data/experiments/nema_vae_{timestamp}"
    save_config(CONFIG, exp_dir)
    
    print(f"Experiment: {exp_dir}")
    print(f"JAX devices: {jax.devices()}")
    
    # Initialize model
    from beagle.network.wavelet_vae import VAE
    
    model = VAE(
        base_features=CONFIG['base_features'],
        latent_dim=CONFIG['latent_dim']
    )
    
    key = random.key(42)
    key, init_key = random.split(key)
    
    dummy = jnp.ones((1, 512, 512, 1))
    variables = model.init(init_key, dummy, random.key(0), training=True)
    
    # Create training state
    tx = optax.adamw(CONFIG['learning_rate'])
    state = TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)
    
    # Load data with train/val split
    print("Loading polymer dataset with train/val split...")

    def data_fn():
        # Create fresh iterator each time
        train_iter, _ = create_nema_vae_iterator(
            data_dir=data_dir,
            batch_size=CONFIG['batch_size'],
            shuffle=True,
            augment=True,
        )
        return train_iter

    # Get batch count
    _, train_batches = create_nema_vae_iterator(
        data_dir=data_dir,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        augment=True,
    )

    # viz_batch = next(train_iter)
    
    print(f"Train batches per epoch: {train_batches}")

    
    # Training step and visualization
    train_step_fn = create_train_step()
    
    # Skip visualization in fast mode
    if not fast_mode:
        viz_config = VizConfig(
            plot_every=1,
            num_samples=10,
            output_dir=f"{exp_dir}/viz"
        )
        viz_callback = create_viz_callback(create_viz_fn(model.apply), viz_config)
    else:
        viz_callback = None
        print("FAST MODE: Skipping all logging, metrics, and visualization for maximum speed!")
    
    # # Training data iterator function
    # def data_fn():
    #     return train_iter
    
    # Train
    print(f"Training for {CONFIG['num_epochs']} epochs...")
    if not fast_mode:
        print("Visualization will use validation data (no augmentation)")
    
    key, train_key = random.split(key)
    final_state, history = train_loop(
        state=state,
        train_step_fn=train_step_fn,
        data_iterator_fn=data_fn,
        num_epochs=CONFIG['num_epochs'],
        num_batches=train_batches,
        rng_key=train_key,
        checkpoint_dir=exp_dir,
        viz_callback=viz_callback,
        # viz_batch=viz_batch,  # Use validation batch for visualization
        fast_mode=fast_mode,
    )
    
    # Save results
    if not fast_mode:
        save_metrics_history(history, exp_dir)
        print(f"Complete. Final loss: {history['train_loss'][-1]:.4f}")
    else:
        print(f"Complete. (Fast mode - no metrics recorded)")
    print(f"Results: {exp_dir}")


if __name__ == "__main__":
    main()
