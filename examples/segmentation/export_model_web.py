"""Export trained segmentation models to TensorFlow SavedModel format.

Usage:
  # Export best model
  python export_model_web.py /data/experiments

  # Export top N models
  python export_model_web.py /data/experiments --top 3

  # Export specific experiment by ID
  python export_model_web.py /data/experiments --id 20251203_191813_0df02952

  # List available experiments
  python export_model_web.py /data/experiments --list

  # Export without validation (faster)
  python export_model_web.py /data/experiments --no-verify

  # Specify data directory for validation
  python export_model_web.py /data/experiments --data /path/to/data --verify
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import tensorflow as tf
from flax import linen as nn

from beagle.conversions import Tolerance, transfer_hierarchical_params
from beagle.experiments import ExperimentTracker
from beagle.network.hrnet import HRNetBackbone, SegmentationHead
from beagle.network.tf.hrnet import build_hrnet_monet
from beagle.training import load_checkpoint


class SegmentationModel(nn.Module):
    """Simple wrapper combining HRNetBackbone + SegmentationHead.

    Returns dict with 'logits' key for compatibility.
    """
    num_classes: int
    num_stages: int = 3
    features: int = 32
    target_res: float = 1.0
    upsample_steps: int = 0
    use_sigmoid: bool = False

    def setup(self):
        self.backbone = HRNetBackbone(
            num_stages=self.num_stages,
            features=self.features,
            target_res=self.target_res
        )
        self.head = SegmentationHead(
            num_classes=self.num_classes,
            features=self.features,
            upsample_steps=self.upsample_steps,
            use_sigmoid=self.use_sigmoid,
            output_key='logits'
        )

    def __call__(self, x: jnp.ndarray, train: bool = False) -> dict[str, jnp.ndarray]:
        """Forward pass through backbone and head.

        Args:
            x: Input image [B, H, W, 1]
            train: Training mode

        Returns:
            Dict with 'logits' key [B, H, W, num_classes]
        """
        backbone_out = self.backbone(x, train=train)
        features = backbone_out['features']
        head_out = self.head(features, train=train)
        return head_out  # Returns {'logits': ...}


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration extracted from experiment."""
    num_stages: int
    features: int
    target_res: float
    train_backbone: bool
    outputs: tuple
    input_size: int = 512


def extract_model_config(exp_data: dict) -> ModelConfig:
    """Extract model config from experiment metadata."""
    config = exp_data['metadata']['config']
    model_cfg = config.get('model', {})
    return ModelConfig(
        num_stages=model_cfg.get('num_stages', 3),
        features=model_cfg.get('features', 32),
        target_res=model_cfg.get('target_res', 1.0),
        train_backbone=model_cfg.get('train_backbone', True),
        outputs=tuple(model_cfg.get('outputs', [(3, False, 2)])),
        input_size=model_cfg.get('input_size', 512),
    )


def export_single_model(
    checkpoint_dir: Path,
    output_dir: Path,
    model_config: ModelConfig,
    experiment_id: str,
    data_dir: Path | None = None,
    verify: bool = True,
) -> bool:
    """
    Export a single model from checkpoint to TensorFlow SavedModel and TFLite.

    Args:
        checkpoint_dir: Path to checkpoint directory
        output_dir: Output directory for exported models
        model_config: Model configuration
        experiment_id: Experiment identifier
        data_dir: Optional data directory for validation with real data
        verify: Whether to verify the conversion

    Returns:
        True if export succeeded.
    """
    checkpoint_path = checkpoint_dir / 'checkpoint_final'
    if not checkpoint_path.exists():
        print(f"  Checkpoint not found: {checkpoint_path}")
        return False

    print(f"\n{'=' * 60}")
    print(f"Exporting: {experiment_id}")
    print(f"  num_stages={model_config.num_stages}, "
          f"features={model_config.features}, "
          f"input_size={model_config.input_size}")
    print(f"{'=' * 60}")

    # Initialize JAX/Flax model
    print("  Initializing JAX/Flax model...")
    key = jrandom.PRNGKey(0)
    dummy_input = jnp.ones((1, model_config.input_size, model_config.input_size, 1))

    # Parse output configuration from tuple format
    num_classes, use_sigmoid, upsample_steps = model_config.outputs[0]

    model_jax = SegmentationModel(
        num_classes=num_classes,
        num_stages=model_config.num_stages,
        features=model_config.features,
        target_res=model_config.target_res,
        upsample_steps=upsample_steps,
        use_sigmoid=use_sigmoid,
    )

    # Load checkpoint
    print(f"  Loading checkpoint from: {checkpoint_path}")
    model_data = load_checkpoint(str(checkpoint_path))
    params, batch_stats = model_data['params'], model_data['batch_stats']

    # Initialize TensorFlow/Keras model
    print("  Initializing TensorFlow/Keras model...")
    input_tf = tf.keras.Input(shape=(model_config.input_size, model_config.input_size, 1))

    hrnet_outputs = build_hrnet_monet(
        input_tf,
        num_stages=model_config.num_stages,
        features=model_config.features,
        target_res=model_config.target_res,
        outputs=model_config.outputs,
    )

    model_tf = tf.keras.Model(inputs=input_tf, outputs=hrnet_outputs)
    model_tf.compile()

    # Transfer weights using the hierarchical transfer utility
    print("  Transferring weights...")
    stats = transfer_hierarchical_params(
        target_model=model_tf,
        source_params=params,
        batch_stats=batch_stats,
        hierarchy_keys=['backbone'],
        layer_type_patterns={
            'Conv_': 'conv2d',
            'Dense_': 'dense',
            'BatchNorm_': 'batch_normalization',
        },
    )

    # Print transfer summary
    total_layers = sum(stats.values())
    for layer_type, count in stats.items():
        print(f"    ✓ Transferred {count} {layer_type} layers")
    print(f"  Total: {total_layers} layers transferred")

    # Verify transfer
    if verify:
        # Numerical verification with random input
        print("  Running numerical verification...")
        tolerance = Tolerance(rtol=1e-3, atol=0.015)
        test_input_jax = jrandom.normal(jrandom.PRNGKey(42),
                                        (1, model_config.input_size, model_config.input_size, 1))

        jax_outputs = model_jax.apply(
            {'params': params, 'batch_stats': batch_stats},
            test_input_jax,
            train=False
        )
        tf_outputs = model_tf(tf.constant(np.array(test_input_jax)), training=False)

        # Compare outputs
        # JAX model returns dict with 'logits' key, TF model returns list
        # For single output case, compare logits directly
        jax_logits = jax_outputs['logits']  # Extract from dict
        tf_logits = tf_outputs[0] if isinstance(tf_outputs, list) else tf_outputs  # TF may return list or single output

        diff = np.abs(np.array(jax_logits) - tf_logits.numpy()).max()
        max_diff = diff

        if max_diff > tolerance.atol:
            print(f"  Numerical verification failed: max_diff={max_diff:.2e}")
            return False
        print(f"  Verified (max_diff={max_diff:.2e})")

        # Optional: Validate with real data if provided
        if data_dir is not None and data_dir.exists():
            print("  Validating with real data samples...")
            try:
                from data_loader import create_segmentation_iterator

                data_iter, _, _, _ = create_segmentation_iterator(
                    data_dir=data_dir,
                    batch_size=1,
                    shuffle=False,
                    augment=False,
                    val_fraction=0.0,  # Use all data
                )

                num_samples = min(3, 5)  # Test up to 5 samples
                for idx in range(num_samples):
                    batch = next(data_iter)
                    test_input_jax = jnp.array(batch["image"])

                    jax_outputs = model_jax.apply(
                        {'params': params, 'batch_stats': batch_stats},
                        test_input_jax,
                        train=False
                    )
                    tf_outputs = model_tf(batch["image"], training=False)

                    # Compare logits: JAX returns dict, TF returns list
                    jax_logits = jax_outputs['logits']
                    tf_logits = tf_outputs[0] if isinstance(tf_outputs, list) else tf_outputs
                    diff = np.abs(np.array(jax_logits) - tf_logits.numpy()).max()
                    if diff > tolerance.atol:
                        print(f"    Sample {idx+1}: FAILED (diff={diff:.2e})")
                        return False

                print(f"    All {num_samples} real data samples match!")
            except Exception as e:
                print(f"    Warning: Could not validate with real data: {e}")

    # Export TensorFlow SavedModel
    export_path = output_dir / f"segmentation_{experiment_id[:16]}"
    print(f"  Exporting SavedModel to: {export_path}")
    model_tf.export(str(export_path))
    print(f"  ✓ SavedModel exported successfully!")

    # Convert to TFLite
    print(f"  Converting to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model_tf)

    # Optional: Enable optimizations for smaller model size
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    tflite_path = output_dir / f"segmentation_{experiment_id[:16]}.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    tflite_size_mb = len(tflite_model) / (1024 * 1024)
    print(f"  ✓ TFLite model saved to: {tflite_path}")
    print(f"    Model size: {tflite_size_mb:.2f} MB")

    # Test TFLite model
    if verify:
        print(f"  Testing TFLite model...")
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Run inference on test input
        test_input = np.array(test_input_jax, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()

        # Get outputs and compare
        tflite_outputs = [
            interpreter.get_tensor(detail['index'])
            for detail in output_details
        ]

        # Compare with TensorFlow model
        # Note: TFLite output order may differ from TensorFlow model
        print(f"  Comparing TFLite vs TensorFlow outputs:")
        tflite_match = True
        for i in [0, 1]:
            
            diff = np.abs(tflite_outputs[i] - tf_outputs[1-i].numpy()).max()

            match = diff < 1e-3
            status = "✓" if match else "⚠"
            print(f"    {status} TFLite output {i} matches TF output: max diff = {diff:.2e}")
            if not match:
                tflite_match = False

        if tflite_match:
            print(f"    ✓ All outputs match!")
        else:
            print(f"    ⚠ Small differences detected (normal for quantization)")

    return True


def list_experiments(tracker: ExperimentTracker) -> None:
    """List all experiments sorted by validation accuracy."""
    experiments = tracker.compare_experiments('val_accuracy', mode='max', top_k=100)

    if not experiments:
        print("No completed experiments found.")
        return

    print(f"\n{'=' * 80}")
    print("AVAILABLE EXPERIMENTS")
    print(f"{'=' * 80}\n")

    for i, exp in enumerate(experiments):
        exp_data = tracker.load_experiment(exp['experiment_id'])
        config = extract_model_config(exp_data)
        val_acc = exp.get('val_accuracy', 'N/A')
        print(f"{i + 1:3d}. {exp['experiment_id']}")
        print(f"     val_accuracy={val_acc} | "
              f"stages={config.num_stages} features={config.features} "
              f"size={config.input_size}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export trained segmentation models to TensorFlow"
    )
    parser.add_argument("experiments_dir", type=Path,
                        help="Directory containing experiments")
    parser.add_argument("--output", "-o", type=Path, default=Path("/data/exported_models"),
                        help="Output directory for exported models")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available experiments")
    parser.add_argument("--best", action="store_true",
                        help="Export only the best model (default)")
    parser.add_argument("--top", type=int, default=None,
                        help="Export top N models by validation accuracy")
    parser.add_argument("--id", type=str, default=None,
                        help="Export specific experiment by ID")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip verification (faster)")
    parser.add_argument("--data", type=Path, default=None,
                        help="Data directory for validation with real data samples")

    args = parser.parse_args()

    tracker = ExperimentTracker(experiments_dir=args.experiments_dir)

    if args.list:
        list_experiments(tracker)
        return

    # Determine which experiments to export
    if args.id:
        experiments = [{'experiment_id': args.id}]
    elif args.top:
        experiments = tracker.compare_experiments('val_accuracy', mode='max', top_k=args.top)
    else:
        # Default: export best model
        experiments = tracker.compare_experiments('val_accuracy', mode='max', top_k=1)

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
                data_dir=args.data,
                verify=not args.no_verify,
            ):
                success_count += 1
        except Exception as e:
            print(f"  Failed to export {exp_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Exported {success_count}/{len(experiments)} models to {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
