"""Example: Training with visualization using beagle library.

Demonstrates how to integrate visualization callbacks into training loops.
Run with: make run CMD='python examples/training_with_viz_example.py'
"""

from __future__ import annotations

import os
import jax
import jax.numpy as jnp
import jax.random as random
import optax
from flax import linen as nn

from beagle.training import (
    TrainState,
    train_loop,
    save_config,
    save_metrics_history,
)

from beagle.visualization import (
    create_viz_callback,
    VizConfig,
    create_simple_reconstruction_callback,
)


# =============================================================================
# Example Model: Simple Autoencoder
# =============================================================================

class SimpleAutoencoder(nn.Module):
    """Simple convolutional autoencoder for demonstration."""
    latent_dim: int = 64
    
    @nn.compact
    def __call__(self, x, training: bool = False, key=None):
        # Encoder
        z = nn.Conv(32, (3, 3), strides=2, padding='SAME')(x)
        z = nn.relu(z)
        z = nn.Conv(64, (3, 3), strides=2, padding='SAME')(z)
        z = nn.relu(z)
        
        # Flatten and bottleneck
        z = z.reshape((z.shape[0], -1))
        z = nn.Dense(self.latent_dim)(z)
        latent = nn.relu(z)
        
        # Decoder
        z = nn.Dense(16 * 16 * 64)(latent)
        z = nn.relu(z)
        z = z.reshape((z.shape[0], 16, 16, 64))
        
        z = nn.ConvTranspose(32, (3, 3), strides=2, padding='SAME')(z)
        z = nn.relu(z)
        z = nn.ConvTranspose(1, (3, 3), strides=2, padding='SAME')(z)
        
        reconstruction = nn.sigmoid(z)
        
        return reconstruction, latent


# =============================================================================
# Training Setup
# =============================================================================

def create_train_step_fn():
    """Create JIT-compiled training step function."""
    @jax.jit
    def train_step(state, batch, rng_key):
        """Single training step (pure computation)."""
        inputs = batch['image']
        
        def loss_fn(params):
            (reconstruction, latent) = state.apply_fn(
                {'params': params},
                inputs,
                training=True
            )
            
            # MSE reconstruction loss
            recon_loss = jnp.mean(jnp.square(inputs - reconstruction))
            
            return recon_loss, {'reconstruction': reconstruction, 'latent': latent}
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(state.params)
        
        new_state = state.apply_gradients(grads=grads)
        
        # Return metrics
        metrics = {
            'loss': loss,
            'grad_norm': optax.global_norm(grads),
            'latent_mean': jnp.mean(jnp.abs(aux['latent']))
        }
        
        return new_state, metrics
    
    return train_step


def create_data_iterator(batch_size: int, num_batches: int, image_size: int = 64):
    """Create a simple synthetic data iterator."""
    def iterator_fn():
        """Returns fresh iterator for each epoch."""
        key = random.key(42)
        
        for _ in range(num_batches):
            key, subkey = random.split(key)
            
            # Generate synthetic images (random noise)
            images = random.uniform(subkey, (batch_size, image_size, image_size, 1))
            
            yield {'image': images}
    
    return iterator_fn


# =============================================================================
# Example 1: Using simple reconstruction callback
# =============================================================================

def example_1_simple_reconstruction():
    """Example with simple reconstruction visualization."""
    print("=" * 70)
    print("Example 1: Simple Reconstruction Visualization")
    print("=" * 70)
    
    # Setup
    batch_size = 16
    num_epochs = 20
    num_batches = 10
    
    # Initialize model
    model = SimpleAutoencoder(latent_dim=64)
    key = random.key(0)
    key, init_key = random.split(key)
    
    dummy_input = jnp.ones((1, 64, 64, 1))
    variables = model.init(init_key, dummy_input, training=False)
    
    # Create state
    tx = optax.adam(0.001)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )
    
    # Create visualization callback
    viz_config = VizConfig(
        plot_every=5,
        num_samples=4,
        output_dir='/data/experiments/example1_viz'
    )
    
    viz_callback = create_simple_reconstruction_callback(
        model.apply,
        viz_config,
        input_key='image'
    )
    
    # Training
    train_step_fn = create_train_step_fn()
    data_iterator_fn = create_data_iterator(batch_size, num_batches)
    
    key, train_key = random.split(key)
    
    final_state, history = train_loop(
        state=state,
        train_step_fn=train_step_fn,
        data_iterator_fn=data_iterator_fn,
        num_epochs=num_epochs,
        num_batches=num_batches,
        rng_key=train_key,
        checkpoint_dir='/data/experiments/example1',
        checkpoint_every=10,
        viz_callback=viz_callback
    )
    
    print(f"✅ Training complete! Final loss: {history['train_loss'][-1]:.4f}")


# =============================================================================
# Example 2: Custom visualization with latent space
# =============================================================================

def example_2_custom_visualization():
    """Example with custom visualization including latent space."""
    print("\n" + "=" * 70)
    print("Example 2: Custom Visualization (with latent space)")
    print("=" * 70)
    
    # Setup
    batch_size = 16
    num_epochs = 20
    num_batches = 10
    
    # Initialize model
    model = SimpleAutoencoder(latent_dim=64)
    key = random.key(0)
    key, init_key = random.split(key)
    
    dummy_input = jnp.ones((1, 64, 64, 1))
    variables = model.init(init_key, dummy_input, training=False)
    
    # Create state
    tx = optax.adam(0.001)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )
    
    # Create custom visualization function
    @jax.jit
    def custom_viz_fn(state, batch, rng_key):
        """Custom visualization including latent space stats."""
        inputs = batch['image']
        
        reconstruction, latent = state.apply_fn(
            {'params': state.params},
            inputs,
            training=False,
            key=rng_key
        )
        
        return {
            'Input': inputs,
            'Reconstruction': reconstruction,
            'Error': jnp.abs(inputs - reconstruction) * 5.0,  # Amplify errors
        }
    
    viz_config = VizConfig(
        plot_every=5,
        num_samples=4,
        output_dir='/data/experiments/example2_viz'
    )
    
    viz_callback = create_viz_callback(custom_viz_fn, viz_config)
    
    # Training
    train_step_fn = create_train_step_fn()
    data_iterator_fn = create_data_iterator(batch_size, num_batches)
    
    key, train_key = random.split(key)
    
    final_state, history = train_loop(
        state=state,
        train_step_fn=train_step_fn,
        data_iterator_fn=data_iterator_fn,
        num_epochs=num_epochs,
        num_batches=num_batches,
        rng_key=train_key,
        checkpoint_dir='/data/experiments/example2',
        checkpoint_every=10,
        viz_callback=viz_callback
    )
    
    print(f"✅ Training complete! Final loss: {history['train_loss'][-1]:.4f}")
    
    # Save metrics
    save_metrics_history(history, '/data/experiments/example2')


if __name__ == "__main__":
    print("🎨 Beagle Training + Visualization Examples\n")
    print(f"JAX devices: {jax.devices()}\n")
    
    # Run examples
    example_1_simple_reconstruction()
    example_2_custom_visualization()
    
    print("\n✅ All examples complete!")
    print("📁 Check /data/experiments/ for outputs")

