"""Train wavelet VAE with performance profiling.

Demonstrates profiling integration for training step and dataloader monitoring.

Run: make run CMD='python examples/train_wavelet_vae_profiled.py /data/polymer_tfrecords'
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
from beagle.training import TrainState, save_config, save_metrics_history
from beagle.visualization import create_viz_callback, VizConfig
from beagle.profiling import (
    create_step_profiler,
    format_step_metrics,
    compute_epoch_summary,
    format_epoch_summary,
)

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
}


def create_train_step(wavelet_weights: tuple[float, ...] = (1.0, 8.0, 8.0, 12.0)):
    """Create JIT-compiled training step."""
    @jax.jit
    def train_step(state: TrainState, batch: dict, rng_key):
        images = batch['depth']
        wavelets = wavedec2(images, wavelet="haar")
        
        def loss_fn(params):
            x_recon, x_wave, mu, log_var = state.apply_fn(
                {'params': params}, images, training=True, key=rng_key
            )
            
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
        images = batch['depth']
        
        x_recon, _, _, _ = model_apply(
            {'params': state.params}, images, training=False, key=rng_key
        )
        
        return {
            'Original': images,
            'Reconstruction': x_recon,
            'Error (5x)': jnp.abs(images - x_recon) * 5.0
        }
    
    return viz_fn


def profiled_train_loop(
    state: TrainState,
    train_step_fn,
    data_iterator,
    num_epochs: int,
    num_batches: int,
    rng_key,
    checkpoint_dir: str,
    viz_callback=None,
):
    """Training loop with integrated profiling."""
    import time
    from beagle.training import save_checkpoint
    
    # Create profiler
    profile_step, _ = create_step_profiler(batch_size=CONFIG['batch_size'])
    
    history = {'train_loss': [], 'recon_loss': []}
    
    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        epoch_profiles = []
        epoch_loss = 0.0
        epoch_recon = 0.0
        
        for step in range(num_batches):
            # Time data loading
            data_start = time.perf_counter()
            batch = next(data_iterator)
            data_time = time.perf_counter() - data_start
            
            # Profile training step
            rng_key, step_key = random.split(rng_key)
            
            (state, metrics), profile = profile_step(
                step_num=step,
                step_fn=train_step_fn,
                args=(state, batch, step_key),
                kwargs={},
                data_time=data_time
            )
            
            epoch_profiles.append(profile)
            epoch_loss += float(metrics['loss'])
            epoch_recon += float(metrics['recon_loss'])
            
            # Log every 10 steps
            if step % 10 == 0:
                print(format_step_metrics(step, profile, metrics))
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_recon = epoch_recon / num_batches
        history['train_loss'].append(avg_loss)
        history['recon_loss'].append(avg_recon)
        
        summary = compute_epoch_summary(epoch_profiles)
        print(format_epoch_summary(epoch, summary))
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f}")
        
        # Save checkpoint
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            save_checkpoint(state, checkpoint_dir, epoch)
        
        # Visualization
        if viz_callback is not None:
            rng_key, viz_key = random.split(rng_key)
            viz_callback(state, batch, viz_key, epoch)
    
    return state, history


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_wavelet_vae_profiled.py /path/to/polymer_tfrecords [--compute-stats]")
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
    exp_dir = f"/data/experiments/wavelet_vae_{timestamp}"
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
    
    dummy = jnp.ones((1, 256, 256, 1))
    variables = model.init(init_key, dummy, random.key(0), training=True)
    
    # Create training state
    tx = optax.adamw(CONFIG['learning_rate'])
    state = TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)
    
    # Load data using shared module
    print("Loading polymer dataset...")
    iterator, batches_per_epoch, img_shape = create_polymer_iterator(
        data_dir=data_dir,
        batch_size=CONFIG['batch_size'],
        crop_size=CONFIG['crop_size'],
        stride=CONFIG['crop_overlap'],
        shuffle=True,
        augment=True
    )
    
    print(f"Batches per epoch: {batches_per_epoch}")
    
    # Training step and visualization
    train_step_fn = create_train_step()
    
    viz_config = VizConfig(
        plot_every=5,
        num_samples=4,
        output_dir=f"{exp_dir}/viz"
    )
    viz_callback = create_viz_callback(create_viz_fn(model.apply), viz_config)
    
    # Train with profiling
    print(f"\nTraining for {CONFIG['num_epochs']} epochs with profiling...\n")
    
    key, train_key = random.split(key)
    final_state, history = profiled_train_loop(
        state=state,
        train_step_fn=train_step_fn,
        data_iterator=iterator,
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


if __name__ == "__main__":
    main()

