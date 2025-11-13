"""Simple profiling demonstration with synthetic data.

Shows how to use profiling utilities without real datasets.

Run: make run CMD='python examples/demo_profiling.py'
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import jax.random as random

from beagle.profiling import (
    create_step_profiler,
    format_step_metrics,
    compute_epoch_summary,
    format_epoch_summary,
    profile_iterator,
)


def create_synthetic_data(batch_size: int, img_size: int = 64):
    """Create synthetic data iterator with variable loading times."""
    def data_generator():
        key = random.key(42)
        for i in range(100):
            # Simulate varying data loading times
            time.sleep(0.001 * (1 + i % 5))
            key, subkey = random.split(key)
            batch = random.normal(subkey, (batch_size, img_size, img_size, 3))
            yield {'images': batch, 'step': i}
    
    return data_generator()


@jax.jit
def dummy_train_step(state, batch):
    """Dummy training step for demonstration."""
    images = batch['images']
    # Simulate some computation
    processed = jnp.mean(images ** 2, axis=(1, 2, 3))
    loss = jnp.mean(processed)
    # Return updated state and metrics (keep as JAX arrays)
    new_state = state + 1
    return new_state, {'loss': loss, 'batch_mean': jnp.mean(images)}


def main():
    print("=" * 70)
    print("Beagle Profiling Demo")
    print("=" * 70)
    print()
    
    # Configuration
    batch_size = 8
    num_epochs = 2
    steps_per_epoch = 20
    
    # Create profiler
    profile_step, _ = create_step_profiler(batch_size=batch_size)
    
    # Initialize dummy state
    state = 0
    rng_key = random.key(42)
    
    print(f"Training with batch_size={batch_size} for {num_epochs} epochs")
    print(f"Steps per epoch: {steps_per_epoch}")
    print()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}")
        print("-" * 70)
        
        # Create data iterator
        data_iter = create_synthetic_data(batch_size)
        
        # Optionally profile the iterator
        profiled_iter = profile_iterator(data_iter, name=f"epoch_{epoch}_data")
        
        epoch_profiles = []
        epoch_loss = 0.0
        
        for step in range(steps_per_epoch):
            # Get batch with timing
            batch, data_time = next(profiled_iter)
            
            # Profile training step
            rng_key, step_key = random.split(rng_key)
            
            (state, metrics), profile = profile_step(
                step_num=step,
                step_fn=dummy_train_step,
                args=(state, batch),
                kwargs={},
                data_time=data_time
            )
            
            epoch_profiles.append(profile)
            epoch_loss += float(metrics['loss'])
            
            # Log every 5 steps
            if step % 5 == 0:
                # Convert JAX arrays to floats for display
                display_metrics = {k: float(v) for k, v in metrics.items()}
                print(format_step_metrics(step, profile, display_metrics))
        
        # Epoch summary
        avg_loss = epoch_loss / steps_per_epoch
        summary = compute_epoch_summary(epoch_profiles)
        
        print()
        print(format_epoch_summary(epoch, summary))
        print(f"Average Loss: {avg_loss:.4f}")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print()
    print("Key Insights:")
    print("- Data loading time varies (simulated with sleep)")
    print("- Compute time is consistent (JIT-compiled)")
    print("- First step is slower (JIT compilation)")
    print("- Profiling has minimal overhead")


if __name__ == "__main__":
    main()

