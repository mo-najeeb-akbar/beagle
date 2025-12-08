"""Minimal example: Train with experiment tracking (< 250 lines)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.random as random
import optax

from beagle.experiments import ExperimentConfig, ExperimentTracker, ModelRegistry
from beagle.training import TrainState, save_checkpoint
from data_loader import create_polymer_iterator


@dataclass(frozen=True)
class DatasetConfig:
    batch_size: int = 32
    crop_size: int = 256
    crop_overlap: int = 192
    val_split: float = 0.2


@dataclass(frozen=True)
class ModelConfig:
    base_features: int = 32
    latent_dim: int = 128


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.001
    num_epochs: int = 10


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_simple_experiment.py /path/to/data")
        sys.exit(1)

    data_dir = Path(sys.argv[1])

    # 1. Create config
    config = ExperimentConfig(
        name='wavelet_vae_minimal',
        dataset=DatasetConfig(),
        model=ModelConfig(),
        training=TrainingConfig(),
        seed=42
    )

    print(f"Config hash: {config.hash()}")

    # 2. Create experiment
    tracker = ExperimentTracker(experiments_dir='/data/experiments')
    run = tracker.create_experiment(name=config.name, config=config)
    print(f"Experiment: {run.experiment_id}")

    # 3. Load data
    train_iter, val_iter, n_train, n_val = create_polymer_iterator(
        data_dir=data_dir,
        batch_size=config.dataset.batch_size,
        crop_size=config.dataset.crop_size,
        stride=config.dataset.crop_overlap,
        val_fraction=config.dataset.val_split,
        augment=True
    )

    # 4. Initialize model (import locally to keep example focused)
    from beagle.network.wavelet_vae import VAE
    import jax.numpy as jnp

    model = VAE(
        base_features=config.model.base_features,
        latent_dim=config.model.latent_dim
    )

    key = random.key(config.seed)
    key, init_key = random.split(key)
    dummy = jnp.ones((1, config.dataset.crop_size, config.dataset.crop_size, 1))
    variables = model.init(init_key, dummy, random.key(0), training=True)

    tx = optax.adamw(config.training.learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)

    # 5. Training loop with tracking
    best_val_loss = float('inf')

    for epoch in range(config.training.num_epochs):
        # Train (simplified - see full example for complete implementation)
        train_losses = []
        for _ in range(n_train):
            batch = next(train_iter)
            # ... training step ...
            train_losses.append(0.5)  # Placeholder

        # Validate
        val_losses = []
        for _ in range(n_val):
            batch = next(val_iter)
            # ... validation step ...
            val_losses.append(0.4)  # Placeholder

        # Log metrics
        metrics = {
            'train_loss': sum(train_losses) / len(train_losses),
            'val_loss': sum(val_losses) / len(val_losses)
        }
        run.log_metrics(metrics, step=epoch)
        print(f"Epoch {epoch + 1}: train={metrics['train_loss']:.4f}, val={metrics['val_loss']:.4f}")

        # Save best checkpoint
        if metrics['val_loss'] < best_val_loss:
            best_val_loss = metrics['val_loss']
            checkpoint_dir = run.output_dir / 'checkpoints' / 'best'
            save_checkpoint(state, str(checkpoint_dir))

    # 6. Finish experiment
    run.finish(status='completed')
    print(run.summary())

    # 7. Register model
    registry = ModelRegistry(registry_dir='/data/models')
    registry.register_model(
        model_name=config.name,
        experiment_id=run.experiment_id,
        config_hash=config.hash(),
        checkpoint_path=run.output_dir / 'checkpoints' / 'best',
        metrics={'best_val_loss': best_val_loss},
        timestamp=run.metadata.timestamp
    )

    # 8. Compare with past experiments
    top_5 = tracker.compare_experiments('val_loss', mode='min', top_k=5)
    print("\nTop 5 experiments:")
    for i, exp in enumerate(top_5):
        print(f"{i + 1}. {exp['experiment_id'][:12]}... - val_loss: {exp['val_loss']:.4f}")

    # 9. Show best model
    best_model = registry.find_best_model('best_val_loss', mode='min')
    if best_model:
        print(f"\nBest model: {best_model['experiment_id']}")
        print(f"Val loss: {best_model['metrics']['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
