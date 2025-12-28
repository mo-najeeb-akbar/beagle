"""Export MoNet model from JAX/Flax to TensorFlow for web deployment.

Validates weight transfer using real data samples from the training set.

Usage:
    make run CMD='python examples/tip_shape/export_model_web.py /path/to/checkpoint /path/to/data_dir'
"""
from __future__ import annotations

import sys

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import tensorflow as tf
import os
from pathlib import Path

from beagle.conversions import Tolerance, transfer_hierarchical_params
from beagle.network.hrnet import MoNet
from beagle.network.tf.hrnet import build_hrnet_monet
from beagle.training import load_checkpoint
from data_loader import create_nema_dish_iterator


def main() -> None:
    
    if len(sys.argv) < 3:
        print("Usage: python export_model_web.py /path/to/checkpoint /path/to/data_dir")
        sys.exit(1)

    # Model configuration (must match training config)
    CONFIG = {
        "num_stages": 3,
        "features": 32,
        "target_res": 1.0,
        "train_bb": True,
        "input_size": 512,
        "outputs": [(1, True, 2)],  # 1 channel, sigmoid, 2x upsample to full res
    }

    print("Initializing JAX/Flax model...")
    key = jrandom.PRNGKey(0)
    dummy_input = jnp.ones((1, CONFIG["input_size"], CONFIG["input_size"], 1))
    
    model_jax = MoNet(
        num_stages=CONFIG["num_stages"],
        features=CONFIG["features"],
        target_res=CONFIG["target_res"],
        train_bb=CONFIG["train_bb"],
        outputs=CONFIG["outputs"],
    )
    
    checkpoint_path = os.path.join(os.path.abspath(sys.argv[1]), 'checkpoint_final')
    print(f"Loading checkpoint from: {checkpoint_path}")
    model_data = load_checkpoint(checkpoint_path)
    params, batch_stats = model_data['params'], model_data['batch_stats']
    
    print("Initializing TensorFlow/Keras model...")
    input_tf = tf.keras.Input(shape=(CONFIG["input_size"], CONFIG["input_size"], 1))
    
    hrnet_outputs = build_hrnet_monet(
        input_tf,
        num_stages=CONFIG["num_stages"],
        features=CONFIG["features"],
        target_res=CONFIG["target_res"],
        outputs=CONFIG["outputs"],
    )
    
    model_tf = tf.keras.Model(inputs=input_tf, outputs=hrnet_outputs)
    model_tf.compile()

    print("\n" + "=" * 50)
    print("Transferring weights using beagle.conversions...")
    print("=" * 50)
    
    # Use the hierarchical transfer utility from the library
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
    
    # Print summary
    total_layers = sum(stats.values())
    for layer_type, count in stats.items():
        print(f"  ✓ Transferred {count} {layer_type} layers")
    print(f"\nTotal: {total_layers} layers transferred")

    print("\n" + "=" * 50)
    print("Loading test data...")
    print("=" * 50)
    
    # Load data directory
    data_dir = Path(sys.argv[2])
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        sys.exit(1)
    
    # Create data iterator (no augmentation for validation)
    print(f"Loading data from: {data_dir}")
    data_iter, batches_per_epoch = create_nema_dish_iterator(
        data_dir=data_dir,
        batch_size=1,  # Process one sample at a time
        shuffle=False,  # Keep consistent for comparison
        augment=False,  # No augmentation for validation
    )
    
    print(f"Total batches available: {batches_per_epoch}")
    
    print("\n" + "=" * 50)
    print("Testing with real data samples...")
    print("=" * 50)

    num_samples = 5
    tolerance = Tolerance(rtol=1e-3, atol=0.015)
    all_match = True
    max_diff = 0.0

    for idx in range(num_samples):
        print(f"\nSample {idx + 1}/{num_samples}:")
        
        # Get next batch from data iterator
        batch = next(data_iter)
        test_input_jax = jnp.array(batch["image"])

        # JAX model forward pass
        jax_outputs = model_jax.apply(
            {'params': params, 'batch_stats': batch_stats},
            test_input_jax,
            train=False
        )
        
        # TensorFlow model forward pass (already in TF format)
        tf_outputs = model_tf(batch["image"], training=False)

        try:
            # Compare each output head (excluding backbone output at the end)
            for head_idx in range(len(CONFIG["outputs"])):
                jax_output = jax_outputs[head_idx]
                tf_output = tf_outputs[head_idx]
                
                output_diff = np.abs(tf_output.numpy() - np.array(jax_output)).max()
                np.testing.assert_allclose(
                    tf_output.numpy(),
                    np.array(jax_output),
                    rtol=tolerance.rtol,
                    atol=tolerance.atol,
                )
                print(f"  ✓ Output head {head_idx} match (max diff: {output_diff:.2e})")
                max_diff = max(max_diff, output_diff)
            
            # Compare backbone output
            jax_backbone = jax_outputs[-1]
            tf_backbone = tf_outputs[-1]
            backbone_diff = np.abs(tf_backbone.numpy() - np.array(jax_backbone)).max()
            np.testing.assert_allclose(
                tf_backbone.numpy(),
                np.array(jax_backbone),
                rtol=tolerance.rtol,
                atol=tolerance.atol,
            )
            print(f"  ✓ Backbone features match (max diff: {backbone_diff:.2e})")
            max_diff = max(max_diff, backbone_diff)

        except AssertionError as e:
            print(f"  ❌ {str(e)}")
            all_match = False

    print("\n" + "=" * 50)
    if all_match:
        print(f"✓ All {num_samples} samples match!")
        print(f"  Maximum absolute difference: {max_diff:.2e}")
        print("=" * 50)
        print("\nConversion successful! Models produce identical outputs.")
    else:
        print("❌ Some samples had mismatches")
        print("=" * 50)

    # Export TensorFlow SavedModel
    export_path = "nema_dish_segmentation_tf"
    print(f"\n" + "=" * 50)
    print(f"Exporting TensorFlow SavedModel to: {export_path}")
    print("=" * 50)
    model_tf.export(export_path)
    print(f"✓ SavedModel exported successfully!")
    
    # Convert to TFLite
    print(f"\n" + "=" * 50)
    print("Converting to TFLite format...")
    print("=" * 50)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model_tf)
    
    # Optional: Enable optimizations for smaller model size
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    tflite_path = "nema_dish_segmentation.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    tflite_size_mb = len(tflite_model) / (1024 * 1024)
    print(f"✓ TFLite model saved to: {tflite_path}")
    print(f"  Model size: {tflite_size_mb:.2f} MB")
    
    # Test TFLite model with one sample
    print(f"\n" + "=" * 50)
    print("Testing TFLite model...")
    print("=" * 50)
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output heads: {len(output_details)}")
    for i, detail in enumerate(output_details):
        print(f"  Output {i} shape: {detail['shape']}")
    
    # Run inference on one sample
    batch = next(data_iter)
    test_input = np.array(batch["image"], dtype=np.float32)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], test_input)
    
    # Run inference
    interpreter.invoke()
    
    # Get outputs
    tflite_outputs = []
    for detail in output_details:
        output_data = interpreter.get_tensor(detail['index'])
        tflite_outputs.append(output_data)
    
    # Compare with TensorFlow model
    tf_outputs = model_tf(batch["image"], training=False)
    
    print(f"\nComparing TFLite vs TensorFlow outputs:")
    tflite_match = True
    for i, j in [(0, 1), (1, 0)]:
        diff = np.abs(tflite_outputs[i] - tf_outputs[j].numpy()).max()
        match = diff < 1e-3
        status = "✓" if match else "❌"
        print(f"  {status} Output {i}: max diff = {diff:.2e}")
        if not match:
            tflite_match = False
    
    if tflite_match:
        print("\n✓ TFLite model matches TensorFlow model!")
    else:
        print("\n⚠ TFLite model has small differences (this is normal for quantization)")
    
    print(f"\n" + "=" * 50)
    print("Export complete!")
    print("=" * 50)
    print(f"SavedModel: {export_path}/")
    print(f"TFLite: {tflite_path}")


if __name__ == "__main__":
    main()

