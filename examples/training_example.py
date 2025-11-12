"""Example: Using beagle.training module to refactor seg_train.py

This demonstrates how to use the functional training library.
Run with: make run CMD='python examples/training_example.py'
"""

from __future__ import annotations

import os
import hashlib

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import hydra
from omegaconf import DictConfig, OmegaConf

from beagle.training import (
    TrainState,
    train_loop,
    save_config,
    save_metrics_history,
)


def generate_unique_key_from_dict(cfg: DictConfig) -> str:
    """Generate unique experiment ID from config (pure function)."""
    json_str = OmegaConf.to_yaml(cfg)
    hash_object = hashlib.sha256(json_str.encode('utf-8'))
    return hash_object.hexdigest()[:10]


def create_train_step_fn(state_template: TrainState):
    """Create JIT-compiled training step function.
    
    This is a higher-order function that returns the step function.
    Separates pure training logic from state management.
    """
    @jax.jit
    def train_step(state, batch, rng_key):
        """Single training step (pure computation)."""
        inputs = batch['image']
        targets = batch['mask']
        
        def loss_fn(params):
            (seg_pred), updates = state.apply_fn(
                {'params': params, 'batch_stats': state.batch_stats},
                inputs,
                mutable=['batch_stats'],
                train=True
            )
            
            seg_pred = seg_pred[0]  # TODO: fix model output structure
            loss = jnp.square(targets - seg_pred).mean()
            
            return loss, updates
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, updates), grads = grad_fn(state.params)
        
        new_state = state.apply_gradients(grads=grads)
        new_state = new_state.replace(batch_stats=updates['batch_stats'])
        
        # Return metrics as dict
        metrics = {
            'loss': loss,
            'grad_norm': optax.global_norm(grads)
        }
        
        return new_state, metrics
    
    return train_step


def create_dataloader_iterator(dataloader, batch_size: int, rng_key):
    """Create data iterator with shuffling."""
    def iterator_fn():
        """Returns fresh iterator for each epoch."""
        jax_ds = dataloader.get_jax_iterator()
        num_batches = dataloader.get_batches_per_epoch()
        
        for _ in range(num_batches):
            batch = next(jax_ds)
            
            # Shuffle batch
            nonlocal rng_key
            rng_key, shuffle_key = random.split(rng_key)
            perm = random.permutation(shuffle_key, batch_size)
            
            shuffled_batch = {
                'image': batch['image'][perm, ...],
                'mask': batch['mask'][perm, ...],
            }
            
            yield shuffled_batch
    
    return iterator_fn


@hydra.main(version_base=None, config_path="/code/configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function using functional training library."""
    
    # ========================================================================
    # Setup (side effects isolated)
    # ========================================================================
    
    print("🔧 Initializing experiment...")
    
    # Generate experiment ID
    experiment_id = generate_unique_key_from_dict(cfg)
    checkpoint_dir = f'/checkpoints/distillation_{experiment_id}'
    
    # Save config
    save_config(OmegaConf.to_container(cfg), checkpoint_dir)
    
    # ========================================================================
    # Model Initialization
    # ========================================================================
    
    from deeplib import generate_model
    
    compute_dtype = jnp.float32
    model_jax = generate_model(cfg)
    
    # Get dimensions from config
    cols_in = int(cfg['camera']['image']['width'])
    rows_in = int(cfg['camera']['image']['height'])
    input_data = jnp.ones((1, rows_in, cols_in, 1), compute_dtype)
    
    # Initialize model
    key = random.key(0)
    key, init_key = random.split(key)
    
    variables = model_jax.init(init_key, input_data, train=False)
    params, batch_stats = variables['params'], variables['batch_stats']
    
    # ========================================================================
    # Training Setup
    # ========================================================================
    
    learning_rate = cfg['train']['learning_rate']
    num_epochs = cfg['train']['epochs']
    batch_size = cfg['train']['batch_size']
    
    # Create optimizer and state
    tx = optax.adamw(learning_rate)
    state = TrainState.create(
        apply_fn=model_jax.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats
    )
    
    # Create train step function
    train_step_fn = create_train_step_fn(state)
    
    # Setup dataloader
    from load_dataset import Dataloader
    
    dataset_path = cfg['dataset']['output_dir']
    ds_factor = cfg['train']['model']['output_res']
    
    dataloader = Dataloader(
        os.path.join(dataset_path, '*.tfrecord'),
        batch_size=batch_size,
        ds_factor=ds_factor
    )
    
    num_batches = dataloader.get_batches_per_epoch()
    
    # Create data iterator function
    key, data_key = random.split(key)
    data_iterator_fn = create_dataloader_iterator(dataloader, batch_size, data_key)
    
    # ========================================================================
    # Training Loop (functional core)
    # ========================================================================
    
    print(f"🚀 Starting training for {num_epochs} epochs...")
    print(f"📊 Batches per epoch: {num_batches}")
    print(f"💾 Checkpoints: {checkpoint_dir}")
    
    key, train_key = random.split(key)
    
    final_state, history = train_loop(
        state=state,
        train_step_fn=train_step_fn,
        data_iterator_fn=data_iterator_fn,
        num_epochs=num_epochs,
        num_batches=num_batches,
        rng_key=train_key,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=10,  # Save every 10 epochs
        log_every=None  # Only epoch-level logging
    )
    
    # ========================================================================
    # Save Results
    # ========================================================================
    
    save_metrics_history(history, checkpoint_dir)
    
    print(f"✅ Training complete!")
    print(f"📁 Results saved to: {checkpoint_dir}")
    print(f"🎯 Final loss: {history['train_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()

