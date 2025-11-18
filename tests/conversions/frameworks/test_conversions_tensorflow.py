from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf

from beagle.conversions.frameworks.tensorflow import (
    create_tf_registry,
    transfer_conv2d,
    transfer_conv2d_transpose,
    transfer_dense,
    transfer_group_norm,
    verify_conv2d,
    verify_conv2d_transpose,
    verify_dense,
    verify_group_norm,
)
from beagle.conversions.types import Tolerance


def test_create_tf_registry() -> None:
    registry = create_tf_registry()

    assert registry.has_transfer("Conv2D")
    assert registry.has_transfer("Conv2DTranspose")
    assert registry.has_transfer("Dense")
    assert registry.has_transfer("GroupNormalization")
    assert registry.has_transfer("BatchNormalization")

    assert registry.has_verify("Conv2D")
    assert registry.has_verify("Conv2DTranspose")
    assert registry.has_verify("Dense")
    assert registry.has_verify("GroupNormalization")
    assert registry.has_verify("BatchNormalization")


def test_transfer_conv2d() -> None:
    layer = tf.keras.layers.Conv2D(32, (3, 3), use_bias=True)
    layer.build((None, 28, 28, 1))

    kernel = jnp.ones((3, 3, 1, 32))
    bias = jnp.zeros((32,))
    params = {"kernel": kernel, "bias": bias}

    transfer_conv2d(layer, params)

    weights = layer.get_weights()
    assert weights[0].shape == (3, 3, 1, 32)
    assert weights[1].shape == (32,)
    np.testing.assert_allclose(weights[0], np.ones((3, 3, 1, 32)))
    np.testing.assert_allclose(weights[1], np.zeros((32,)))


def test_transfer_conv2d_no_bias() -> None:
    layer = tf.keras.layers.Conv2D(32, (3, 3), use_bias=False)
    layer.build((None, 28, 28, 1))

    kernel = jnp.ones((3, 3, 1, 32))
    params = {"kernel": kernel}

    transfer_conv2d(layer, params)

    weights = layer.get_weights()
    assert len(weights) == 1
    assert weights[0].shape == (3, 3, 1, 32)


def test_transfer_conv2d_transpose() -> None:
    layer = tf.keras.layers.Conv2DTranspose(32, (3, 3), use_bias=True)
    layer.build((None, 28, 28, 1))

    kernel = jnp.ones((3, 3, 1, 32))
    bias = jnp.zeros((32,))
    params = {"kernel": kernel, "bias": bias}

    transfer_conv2d_transpose(layer, params)

    weights = layer.get_weights()
    assert weights[0].shape == (3, 3, 32, 1)
    assert weights[1].shape == (32,)


def test_transfer_dense() -> None:
    layer = tf.keras.layers.Dense(64, use_bias=True)
    layer.build((None, 128))

    kernel = jnp.ones((128, 64))
    bias = jnp.zeros((64,))
    params = {"kernel": kernel, "bias": bias}

    transfer_dense(layer, params)

    weights = layer.get_weights()
    assert weights[0].shape == (128, 64)
    assert weights[1].shape == (64,)
    np.testing.assert_allclose(weights[0], np.ones((128, 64)))
    np.testing.assert_allclose(weights[1], np.zeros((64,)))


def test_transfer_group_norm() -> None:
    layer = tf.keras.layers.GroupNormalization(groups=8)
    layer.build((None, 28, 28, 32))

    scale = jnp.ones((32,))
    bias = jnp.zeros((32,))
    params = {"scale": scale, "bias": bias}

    transfer_group_norm(layer, params)

    weights = layer.get_weights()
    assert weights[0].shape == (32,)
    assert weights[1].shape == (32,)
    np.testing.assert_allclose(weights[0], np.ones((32,)))
    np.testing.assert_allclose(weights[1], np.zeros((32,)))


def test_verify_conv2d_success() -> None:
    layer = tf.keras.layers.Conv2D(32, (3, 3), use_bias=True)
    layer.build((None, 28, 28, 1))

    kernel = jnp.ones((3, 3, 1, 32))
    bias = jnp.zeros((32,))
    params = {"kernel": kernel, "bias": bias}

    transfer_conv2d(layer, params)

    tolerance = Tolerance()
    success, messages = verify_conv2d(layer, params, tolerance)

    assert success
    assert len(messages) == 2
    assert "kernel" in messages[0]
    assert "bias" in messages[1]


def test_verify_conv2d_failure() -> None:
    layer = tf.keras.layers.Conv2D(32, (3, 3), use_bias=True)
    layer.build((None, 28, 28, 1))

    kernel = jnp.ones((3, 3, 1, 32))
    bias = jnp.zeros((32,))
    params = {"kernel": kernel, "bias": bias}

    layer.set_weights([np.zeros((3, 3, 1, 32)), np.ones((32,))])

    tolerance = Tolerance()
    success, messages = verify_conv2d(layer, params, tolerance)

    assert not success


def test_verify_dense_success() -> None:
    layer = tf.keras.layers.Dense(64, use_bias=True)
    layer.build((None, 128))

    kernel = jnp.ones((128, 64))
    bias = jnp.zeros((64,))
    params = {"kernel": kernel, "bias": bias}

    transfer_dense(layer, params)

    tolerance = Tolerance()
    success, messages = verify_dense(layer, params, tolerance)

    assert success
    assert len(messages) == 2


def test_verify_conv2d_transpose_success() -> None:
    layer = tf.keras.layers.Conv2DTranspose(32, (3, 3), use_bias=True)
    layer.build((None, 28, 28, 1))

    kernel = jnp.ones((3, 3, 1, 32))
    bias = jnp.zeros((32,))
    params = {"kernel": kernel, "bias": bias}

    transfer_conv2d_transpose(layer, params)

    tolerance = Tolerance()
    success, messages = verify_conv2d_transpose(layer, params, tolerance)

    assert success
    assert len(messages) == 2


def test_verify_group_norm_success() -> None:
    layer = tf.keras.layers.GroupNormalization(groups=8)
    layer.build((None, 28, 28, 32))

    scale = jnp.ones((32,))
    bias = jnp.zeros((32,))
    params = {"scale": scale, "bias": bias}

    transfer_group_norm(layer, params)

    tolerance = Tolerance()
    success, messages = verify_group_norm(layer, params, tolerance)

    assert success
    assert len(messages) == 2

