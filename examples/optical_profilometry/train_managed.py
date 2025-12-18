"""
Complete training example with experiment management (< 250 lines).

Usage:
  python train_managed.py /data/polymer_tfrecords
  python train_managed.py /data/polymer_tfrecords --sweep
  python train_managed.py /data/polymer_tfrecords --compare
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import optax

from beagle.training import TrainState, save_checkpoint
from beagle.experiments import (
    ExperimentConfig,
    ExperimentTracker,
    ModelRegistry,
    ParamSpec,
    run_sweep,
    get_best_config,
)

from configs import DatasetConfig, ModelConfig, TrainingConfig
from data_loader import create_polymer_iterator
from training_steps import create_train_step, create_val_step, run_epoch


def train_model(config: ExperimentConfig, data_dir: Path) -> dict[str, float]:
    """Train model with experiment tracking."""
    from beagle.network.wavelet_vae import VAE

    print(f"\n{'=' * 80}")
    print(f"Experiment: {config.name}")
    print(f"Config hash: {config.hash()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"{'=' * 80}\n")

    # Create experiment
    tracker = ExperimentTracker(experiments_dir='/data/experiments')
    run = tracker.create_experiment(name=config.name, config=config)
    print(f"Experiment ID: {run.experiment_id}")
    print(f"Output: {run.output_dir}\n")

    # Load data
    print("Loading dataset...")
    train_iter, val_iter, n_train, n_val = create_polymer_iterator(
        data_dir=data_dir,
        batch_size=config.dataset.batch_size,
        crop_size=config.dataset.crop_size,
        stride=config.dataset.crop_overlap,
        shuffle=True,
        augment=True,
        val_fraction=config.dataset.val_split,
        seed=config.dataset.split_seed,
    )
    print(f"Train: {n_train} batches, Val: {n_val} batches\n")

    # Initialize model
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

    # Training steps
    train_step_fn = create_train_step(config.model.wavelet_weights)
    val_step_fn = create_val_step(config.model.wavelet_weights)

    # Training loop
    print(f"Training for {config.training.num_epochs} epochs...\n")
    best_val_loss = float('inf')
    best_state = None  # Keep best checkpoint in memory

    for epoch in range(config.training.num_epochs):
        state, metrics, key = run_epoch(
            state, train_iter, val_iter, train_step_fn, val_step_fn,
            n_train, n_val, key
        )
        run.log_metrics(metrics, step=epoch)

        print(f"Epoch {epoch + 1:02d}/{config.training.num_epochs}: "
              f"train={metrics['train_loss']:.4f}, val={metrics['val_loss']:.4f}")

        # Keep best checkpoint in memory (no disk I/O during training)
        if metrics['val_loss'] < best_val_loss:
            best_val_loss = metrics['val_loss']
            best_state = state  # Just store reference (cheap)
            print(f"  → New best model (val_loss={best_val_loss:.4f})")
    
    # Save best checkpoint once at the end
    if best_state is not None:
        print(f"\nSaving best checkpoint (val_loss={best_val_loss:.4f})...")
        checkpoint_dir = run.output_dir / 'checkpoints' / 'best'
        save_checkpoint(best_state, str(checkpoint_dir))
        print(f"  ✓ Saved to {checkpoint_dir}")

    # Finish experiment
    final_metrics = {
        'best_val_loss': run.get_best_metric('val_loss', mode='min'),
        'final_train_loss': run.get_latest_metric('train_loss'),
        'final_val_loss': run.get_latest_metric('val_loss'),
    }
    run.finish(status='completed')

    # Register model
    registry = ModelRegistry(registry_dir='/data/models')
    registry.register_model(
        model_name=config.name,
        experiment_id=run.experiment_id,
        config_hash=config.hash(),
        checkpoint_path=run.output_dir / 'checkpoints' / 'best',
        metrics=final_metrics,
        timestamp=run.metadata.timestamp
    )

    print(f"\n✓ Model registered with best_val_loss={best_val_loss:.4f}\n")
    return final_metrics


def run_hyperparameter_sweep(base_config: ExperimentConfig, data_dir: Path):
    """Run hyperparameter sweep."""
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SWEEP")
    print("=" * 80)

    param_specs = [
        ParamSpec(name='model.latent_dim', values=[64, 128, 256]),
        ParamSpec(name='training.learning_rate', values=[1e-4, 5e-4, 1e-3]),
    ]

    results = run_sweep(
        base_config=base_config,
        param_specs=param_specs,
        train_fn=lambda cfg: train_model(cfg, data_dir),
        method='grid'
    )

    best = get_best_config(results, 'best_val_loss', mode='min')
    print(f"\n{'=' * 80}")
    print("BEST CONFIGURATION")
    print(f"{'=' * 80}")
    if best is None:
        print("No successful training runs! Check errors above.")
        failed = [r for r in results if r['status'] == 'failed']
        print(f"Failed runs: {len(failed)}/{len(results)}")
    else:
        print(f"Params: {best['params']}")
        print(f"Best val loss: {best['metrics']['best_val_loss']:.4f}")


def compare_experiments():
    """Compare existing experiments."""
    tracker = ExperimentTracker(experiments_dir='/data/experiments')
    top_10 = tracker.compare_experiments('val_loss', mode='min', top_k=10)

    print(f"\n{'=' * 80}")
    print("TOP 10 EXPERIMENTS BY VALIDATION LOSS")
    print(f"{'=' * 80}\n")

    for i, exp in enumerate(top_10):
        print(f"{i + 1:2d}. {exp['experiment_id']}")
        print(f"    Name: {exp['name']}")
        print(f"    Val loss: {exp['val_loss']:.4f}")
        print(f"    Config hash: {exp['config_hash']}\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else None

    # Base configuration
    base_config = ExperimentConfig(
        name='wavelet_vae_polymer',
        dataset=DatasetConfig(),
        model=ModelConfig(),
        training=TrainingConfig(),
        seed=42
    )

    if mode == '--sweep':
        run_hyperparameter_sweep(base_config, data_dir)

    elif mode == '--compare':
        compare_experiments()

    else:
        # Single training run
        metrics = train_model(base_config, data_dir)

        # Show comparison with past experiments
        tracker = ExperimentTracker(experiments_dir='/data/experiments')
        top_5 = tracker.compare_experiments('val_loss', mode='min', top_k=5)

        print(f"{'=' * 80}")
        print("TOP 5 EXPERIMENTS")
        print(f"{'=' * 80}\n")
        for i, exp in enumerate(top_5):
            print(f"{i + 1}. {exp['experiment_id'][:16]}... - "
                  f"val_loss={exp['val_loss']:.4f} ({exp['name']})")


if __name__ == "__main__":
    main()
