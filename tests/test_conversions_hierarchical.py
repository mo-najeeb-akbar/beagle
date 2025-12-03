"""Tests for hierarchical parameter conversion utilities."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from beagle.conversions.hierarchical import transfer_hierarchical_params


def test_transfer_hierarchical_params_conv_only():
    """Test transferring Conv layers from hierarchical JAX to flat Keras."""
    jax_params = {
        'backbone': {
            'Conv_0': {'kernel': jnp.ones((3, 3, 1, 16))},
            'Conv_1': {'kernel': jnp.ones((3, 3, 16, 32))},
        },
        'Conv_0': {'kernel': jnp.ones((1, 1, 32, 1))},
    }
    
    inputs = tf.keras.Input(shape=(32, 32, 1))
    x = tf.keras.layers.Conv2D(16, 3, padding='same', use_bias=False, name='conv2d')(inputs)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', use_bias=False, name='conv2d_1')(x)
    outputs = tf.keras.layers.Conv2D(1, 1, padding='same', use_bias=False, name='conv2d_2')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    stats = transfer_hierarchical_params(
        target_model=model,
        source_params=jax_params,
        hierarchy_keys=['backbone'],
        layer_type_patterns={'Conv_': 'conv2d'},
    )
    
    assert stats['conv2d'] == 3
    
    # Verify weights were transferred
    conv_0 = model.get_layer('conv2d')
    assert conv_0.get_weights()[0].shape == (3, 3, 1, 16)
    assert np.allclose(conv_0.get_weights()[0], np.ones((3, 3, 1, 16)))


def test_transfer_hierarchical_params_with_batch_norm():
    """Test transferring Conv+BatchNorm from hierarchical JAX to flat Keras."""
    jax_params = {
        'backbone': {
            'Conv_0': {'kernel': jnp.ones((3, 3, 1, 16)), 'bias': jnp.zeros(16)},
            'BatchNorm_0': {'scale': jnp.ones(16), 'bias': jnp.zeros(16)},
        },
        'Conv_0': {'kernel': jnp.ones((1, 1, 16, 1))},
    }
    
    jax_batch_stats = {
        'backbone': {
            'BatchNorm_0': {'mean': jnp.zeros(16), 'var': jnp.ones(16)},
        },
    }
    
    inputs = tf.keras.Input(shape=(32, 32, 1))
    x = tf.keras.layers.Conv2D(16, 3, padding='same', name='conv2d')(inputs)
    x = tf.keras.layers.BatchNormalization(name='batch_normalization')(x)
    outputs = tf.keras.layers.Conv2D(1, 1, padding='same', use_bias=False, name='conv2d_1')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    stats = transfer_hierarchical_params(
        target_model=model,
        source_params=jax_params,
        batch_stats=jax_batch_stats,
        hierarchy_keys=['backbone'],
        layer_type_patterns={
            'Conv_': 'conv2d',
            'BatchNorm_': 'batch_normalization',
        },
    )
    
    assert stats['conv2d'] == 2
    assert stats['batch_normalization'] == 1
    
    # Verify BatchNorm weights (gamma, beta, mean, var)
    bn = model.get_layer('batch_normalization')
    weights = bn.get_weights()
    assert len(weights) == 4
    assert np.allclose(weights[0], np.ones(16))  # gamma
    assert np.allclose(weights[1], np.zeros(16))  # beta
    assert np.allclose(weights[2], np.zeros(16))  # mean
    assert np.allclose(weights[3], np.ones(16))  # var


def test_transfer_hierarchical_params_multiple_hierarchy_levels():
    """Test transferring from multiple hierarchy levels."""
    jax_params = {
        'encoder': {
            'Conv_0': {'kernel': jnp.ones((3, 3, 1, 16))},
        },
        'decoder': {
            'Conv_0': {'kernel': jnp.ones((3, 3, 16, 32))},
        },
        'Conv_0': {'kernel': jnp.ones((1, 1, 32, 1))},
    }
    
    inputs = tf.keras.Input(shape=(32, 32, 1))
    x = tf.keras.layers.Conv2D(16, 3, padding='same', use_bias=False, name='conv2d')(inputs)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', use_bias=False, name='conv2d_1')(x)
    outputs = tf.keras.layers.Conv2D(1, 1, padding='same', use_bias=False, name='conv2d_2')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    stats = transfer_hierarchical_params(
        target_model=model,
        source_params=jax_params,
        hierarchy_keys=['encoder', 'decoder'],
        layer_type_patterns={'Conv_': 'conv2d'},
    )
    
    assert stats['conv2d'] == 3


def test_transfer_hierarchical_params_no_bias():
    """Test transferring Conv layers without bias."""
    jax_params = {
        'backbone': {
            'Conv_0': {'kernel': jnp.ones((3, 3, 1, 16))},
        },
    }
    
    inputs = tf.keras.Input(shape=(32, 32, 1))
    outputs = tf.keras.layers.Conv2D(16, 3, padding='same', use_bias=False, name='conv2d')(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    stats = transfer_hierarchical_params(
        target_model=model,
        source_params=jax_params,
        hierarchy_keys=['backbone'],
        layer_type_patterns={'Conv_': 'conv2d'},
    )
    
    assert stats['conv2d'] == 1
    
    conv = model.get_layer('conv2d')
    weights = conv.get_weights()
    assert len(weights) == 1  # Only kernel, no bias
    assert weights[0].shape == (3, 3, 1, 16)


def test_transfer_hierarchical_params_dense():
    """Test transferring Dense layers from hierarchical JAX to flat Keras."""
    jax_params = {
        'backbone': {
            'Dense_0': {'kernel': jnp.ones((64, 32)), 'bias': jnp.zeros(32)},
        },
        'Dense_0': {'kernel': jnp.ones((32, 10)), 'bias': jnp.zeros(10)},
    }
    
    inputs = tf.keras.Input(shape=(64,))
    x = tf.keras.layers.Dense(32, name='dense')(inputs)
    outputs = tf.keras.layers.Dense(10, name='dense_1')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    stats = transfer_hierarchical_params(
        target_model=model,
        source_params=jax_params,
        hierarchy_keys=['backbone'],
        layer_type_patterns={'Dense_': 'dense'},
    )
    
    assert stats['dense'] == 2
    
    # Verify shapes
    dense_0 = model.get_layer('dense')
    assert dense_0.get_weights()[0].shape == (64, 32)
    assert dense_0.get_weights()[1].shape == (32,)
    
    dense_1 = model.get_layer('dense_1')
    assert dense_1.get_weights()[0].shape == (32, 10)
    assert dense_1.get_weights()[1].shape == (10,)

