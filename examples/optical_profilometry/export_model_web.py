"""Export trained models to TensorFlow SavedModel format.

Usage:
  # Export best model
  python export_model_web.py /data/experiments --best

  # Export top N models
  python export_model_web.py /data/experiments --top 3

  # Export specific experiment by ID
  python export_model_web.py /data/experiments --id 20251203_191813_0df02952

  # List available experiments
  python export_model_web.py /data/experiments --list
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.random as jrandom
import numpy as np
import tensorflow as tf

from beagle.conversions import (
    Tolerance,
    create_tf_registry,
    extract_structure,
    transfer_weights,
    verify_transfer,
)
from beagle.experiments import ExperimentTracker
from beagle.network.tf.wavelet_vae import VAETF
from beagle.network.wavelet_vae import VAE
from beagle.training import load_checkpoint


def is_keras_layer(x: Any) -> bool:
    return hasattr(x, "get_weights") and hasattr(x, "set_weights")


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration extracted from experiment."""
    latent_dim: int
    base_features: int
    block_size: int = 8


def extract_model_config(exp_data: dict) -> ModelConfig:
    """Extract model config from experiment metadata."""
    config = exp_data['metadata']['config']
    model_cfg = config.get('model', {})
    return ModelConfig(
        latent_dim=model_cfg.get('latent_dim', 128),
        base_features=model_cfg.get('base_features', 32),
        block_size=model_cfg.get('block_size', 8),
    )


def export_single_model(
    checkpoint_dir: Path,
    output_dir: Path,
    model_config: ModelConfig,
    experiment_id: str,
    verify: bool = True,
) -> bool:
    """
    Export a single model from checkpoint to TensorFlow SavedModel.

    Returns True if export succeeded.
    """
    checkpoint_path = checkpoint_dir / 'checkpoint_final'
    if not checkpoint_path.exists():
        print(f"  Checkpoint not found: {checkpoint_path}")
        return False

    print(f"\n{'=' * 60}")
    print(f"Exporting: {experiment_id}")
    print(f"  latent_dim={model_config.latent_dim}, "
          f"base_features={model_config.base_features}")
    print(f"{'=' * 60}")

    # Initialize Flax model to get haar conv kernels
    key = jrandom.PRNGKey(0)
    dummy_input = jrandom.normal(key, (1, 256, 256, 1))
    model_flax = VAE(
        latent_dim=model_config.latent_dim,
        base_features=model_config.base_features,
        block_size=model_config.block_size,
    )
    params_random_init = model_flax.init(key, dummy_input, key)['params']

    # Load checkpoint
    print("  Loading checkpoint...")
    model_data = load_checkpoint(str(checkpoint_path))
    params = model_data['params']

    # Copy haar conv kernels (constants that need initialization)
    params['Encoder']['haar_conv']['Conv_0']['kernel'] = \
        params_random_init['Encoder']['haar_conv']['Conv_0']['kernel']
    params['Decoder']['haar_conv_transpose']['ConvTranspose_0']['kernel'] = \
        params_random_init['Decoder']['haar_conv_transpose']['ConvTranspose_0']['kernel']

    # Initialize TensorFlow model
    print("  Creating TensorFlow model...")
    input_tf = tf.keras.Input(shape=(256, 256, 1))
    model_tf = VAETF(
        latent_dim=model_config.latent_dim,
        base_features=model_config.base_features,
        block_size=model_config.block_size,
    )
    _ = model_tf(input_tf)
    model_tf.compile()

    # Transfer weights
    print("  Transferring weights...")
    registry = create_tf_registry()
    num_layers = transfer_weights(
        model_tf, params, registry, is_keras_layer, extract_structure
    )
    print(f"  Transferred {num_layers} layers")

    # Verify transfer
    if verify:
        result = verify_transfer(model_tf, params, registry, is_keras_layer)
        if not result.success:
            print(f"  Weight verification failed: {len(result.mismatches)} mismatches")
            return False

        # Quick numerical check
        tolerance = Tolerance(rtol=1e-3, atol=0.015)
        test_input = jrandom.normal(jrandom.PRNGKey(42), (1, 256, 256, 1))

        flax_out = model_flax.apply({'params': params}, test_input, key, training=False)
        tf_out = model_tf(tf.constant(np.array(test_input)), training=False)

        max_diff = np.abs(tf_out[0].numpy() - np.array(flax_out[0])).max()
        if max_diff > tolerance.atol:
            print(f"  Numerical verification failed: max_diff={max_diff:.2e}")
            return False
        print(f"  Verified (max_diff={max_diff:.2e})")

    # Export
    export_path = output_dir / f"vae_{experiment_id[:16]}"
    print(f"  Exporting to: {export_path}")
    model_tf.export(str(export_path))

    return True


def list_experiments(tracker: ExperimentTracker) -> None:
    """List all experiments sorted by validation loss."""
    experiments = tracker.compare_experiments('val_loss', mode='min', top_k=100)

    if not experiments:
        print("No completed experiments found.")
        return

    print(f"\n{'=' * 80}")
    print("AVAILABLE EXPERIMENTS")
    print(f"{'=' * 80}\n")

    for i, exp in enumerate(experiments):
        exp_data = tracker.load_experiment(exp['experiment_id'])
        config = extract_model_config(exp_data)
        print(f"{i + 1:3d}. {exp['experiment_id']}")
        print(f"     val_loss={exp['val_loss']:.4f} | "
              f"latent={config.latent_dim} base={config.base_features}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained models to TensorFlow")
    parser.add_argument("experiments_dir", type=Path,
                        help="Directory containing experiments")
    parser.add_argument("--output", "-o", type=Path, default=Path("/data/exported_models"),
                        help="Output directory for exported models")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available experiments")
    parser.add_argument("--best", action="store_true",
                        help="Export only the best model")
    parser.add_argument("--top", type=int, default=None,
                        help="Export top N models by validation loss")
    parser.add_argument("--id", type=str, default=None,
                        help="Export specific experiment by ID")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip verification (faster)")

    args = parser.parse_args()

    tracker = ExperimentTracker(experiments_dir=args.experiments_dir)

    if args.list:
        list_experiments(tracker)
        return

    # Determine which experiments to export
    if args.id:
        experiments = [{'experiment_id': args.id}]
    elif args.best:
        experiments = tracker.compare_experiments('val_loss', mode='min', top_k=1)
    elif args.top:
        experiments = tracker.compare_experiments('val_loss', mode='min', top_k=args.top)
    else:
        # Default: export best
        experiments = tracker.compare_experiments('val_loss', mode='min', top_k=1)

    if not experiments:
        print("No experiments found to export.")
        return

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Export each experiment
    success_count = 0
    for exp in experiments:
        exp_id = exp['experiment_id']
        try:
            exp_data = tracker.load_experiment(exp_id)
            config = extract_model_config(exp_data)
            checkpoint_dir = exp_data['output_dir'] / 'checkpoints' / 'best'

            if export_single_model(
                checkpoint_dir=checkpoint_dir,
                output_dir=args.output,
                model_config=config,
                experiment_id=exp_id,
                verify=not args.no_verify,
            ):
                success_count += 1
        except Exception as e:
            print(f"  Failed to export {exp_id}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Exported {success_count}/{len(experiments)} models to {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
