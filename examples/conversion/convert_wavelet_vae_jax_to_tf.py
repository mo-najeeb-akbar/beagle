from __future__ import annotations

import sys
from typing import Any

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import tensorflow as tf

from beagle.conversions import (
    create_tf_registry,
    extract_structure,
    transfer_weights,
    verify_transfer,
    Tolerance,
)
from beagle.network.tf.wavelet_vae import VAETF
from beagle.network.wavelet_vae import VAE
from beagle.training import load_params


def is_keras_layer(x: Any) -> bool:
    return hasattr(x, "get_weights") and hasattr(x, "set_weights")


def main() -> None:
    print("Initializing Flax model...")
    key = jrandom.PRNGKey(0)
    dummy_input = jrandom.normal(key, (1, 256, 256, 1))
    model_flax = VAE(latent_dim=128, base_features=32, block_size=8)
    params_random_init = model_flax.init(key, dummy_input, key)['params']
    
    if len(sys.argv) >= 2:
        checkpoint_path = sys.argv[1]
        print(f"Loading checkpoint from: {checkpoint_path}")
        params = load_params(checkpoint_path)
        params['Encoder']['haar_conv']['Conv_0']['kernel'] = params_random_init['Encoder']['haar_conv']['Conv_0']['kernel']
        params['Decoder']['haar_conv_transpose']['ConvTranspose_0']['kernel'] = params_random_init['Decoder']['haar_conv_transpose']['ConvTranspose_0']['kernel']
    else:
        print("No checkpoint provided, using randomly initialized weights")
        params = params_random_init
    
    print("Initializing TensorFlow/Keras model...")
    input_tf = tf.keras.Input(shape=(256, 256, 1))
    model_tf = VAETF(latent_dim=128, base_features=32, block_size=8)
    _ = model_tf(input_tf)
    model_tf.compile()

    print("\n" + "=" * 50)
    print("Transferring weights...")
    print("=" * 50)
    registry = create_tf_registry()
    num_layers = transfer_weights(
        model_tf, params, registry, is_keras_layer, extract_structure
    )
    print(f"Transferred {num_layers} layers")

    print("\n" + "=" * 50)
    print("Verifying weight equality...")
    print("=" * 50)
    result = verify_transfer(model_tf, params, registry, is_keras_layer)

    if result.success:
        print(f"\n✓ All {len(result.matches)} weight checks passed!")
    else:
        print(f"\n❌ Found {len(result.mismatches)} mismatches:")
        for mismatch in result.mismatches:
            print(f"  {mismatch}")

    print("\n" + "=" * 50)
    print("Testing with sample inputs...")
    print("=" * 50)

    num_samples = 5
    test_inputs = [
        jrandom.normal(jrandom.PRNGKey(i + 100), (1, 256, 256, 1))
        for i in range(num_samples)
    ]

    tolerance = Tolerance(rtol=1e-3, atol=0.015)
    all_match = True
    max_diff = 0.0

    for idx, test_input_jax in enumerate(test_inputs):
        print(f"\nSample {idx + 1}/{num_samples}:")

        flax_output = model_flax.apply({'params': params}, test_input_jax, key, training=False)
        flax_recon, flax_haar, flax_mu, flax_logvar = flax_output

        test_input_tf = tf.constant(np.array(test_input_jax), dtype=tf.float32)
        tf_output = model_tf(test_input_tf, training=False)
        tf_recon, tf_haar, tf_mu, tf_logvar = tf_output

        try:
            recon_diff = np.abs(tf_recon.numpy() - np.array(flax_recon)).max()
            np.testing.assert_allclose(
                tf_recon.numpy(),
                np.array(flax_recon),
                rtol=tolerance.rtol,
                atol=tolerance.atol,
            )
            print(f"  ✓ Reconstruction match (max diff: {recon_diff:.2e})")
            max_diff = max(max_diff, recon_diff)

            haar_diff = np.abs(tf_haar.numpy() - np.array(flax_haar)).max()
            np.testing.assert_allclose(
                tf_haar.numpy(),
                np.array(flax_haar),
                rtol=tolerance.rtol,
                atol=tolerance.atol,
            )
            print(f"  ✓ Haar coefficients match (max diff: {haar_diff:.2e})")
            max_diff = max(max_diff, haar_diff)

            mu_diff = np.abs(tf_mu.numpy() - np.array(flax_mu)).max()
            np.testing.assert_allclose(
                tf_mu.numpy(), np.array(flax_mu), rtol=tolerance.rtol, atol=0.01
            )
            print(f"  ✓ Mu match (max diff: {mu_diff:.2e})")
            max_diff = max(max_diff, mu_diff)

            logvar_diff = np.abs(tf_logvar.numpy() - np.array(flax_logvar)).max()
            np.testing.assert_allclose(
                tf_logvar.numpy(),
                np.array(flax_logvar),
                rtol=tolerance.rtol,
                atol=0.01,
            )
            print(f"  ✓ Log variance match (max diff: {logvar_diff:.2e})")
            max_diff = max(max_diff, logvar_diff)

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

    model_tf.save("wavelet_vae_tf.keras")


if __name__ == "__main__":
    main()

