"""Tests for seed utility."""
from __future__ import annotations

import random
import numpy as np
import tensorflow as tf

from beagle.dataset.seed import set_global_seed, set_tf_deterministic


def test_set_global_seed_python_random():
    """Test that set_global_seed affects Python's random."""
    set_global_seed(42)
    val1 = random.random()
    
    set_global_seed(42)
    val2 = random.random()
    
    assert val1 == val2


def test_set_global_seed_numpy_random():
    """Test that set_global_seed affects NumPy random."""
    set_global_seed(42)
    arr1 = np.random.rand(5)
    
    set_global_seed(42)
    arr2 = np.random.rand(5)
    
    assert np.allclose(arr1, arr2)


def test_set_global_seed_tensorflow_random():
    """Test that set_global_seed affects TensorFlow random."""
    set_global_seed(42)
    val1 = tf.random.uniform([5]).numpy()
    
    set_global_seed(42)
    val2 = tf.random.uniform([5]).numpy()
    
    assert np.allclose(val1, val2)


def test_set_global_seed_different_seeds():
    """Test that different seeds produce different results."""
    set_global_seed(42)
    val1 = tf.random.uniform([5]).numpy()
    
    set_global_seed(123)
    val2 = tf.random.uniform([5]).numpy()
    
    # Should be different (with extremely high probability)
    assert not np.allclose(val1, val2)


def test_set_tf_deterministic_does_not_crash():
    """Test that set_tf_deterministic can be called without error."""
    # Just verify it doesn't crash - actual determinism is hard to test
    set_tf_deterministic(True)
    set_tf_deterministic(False)


def test_seed_reproducible_sequence():
    """Test that seeding produces reproducible sequence of random values."""
    set_global_seed(42)
    sequence1 = [random.random() for _ in range(10)]
    
    set_global_seed(42)
    sequence2 = [random.random() for _ in range(10)]
    
    assert sequence1 == sequence2


def test_seed_affects_tf_image_operations():
    """Test that seed affects TensorFlow image random operations."""
    img = tf.random.uniform([32, 32, 3])
    
    set_global_seed(42)
    flipped1 = tf.image.random_flip_left_right(img).numpy()
    
    set_global_seed(42)
    flipped2 = tf.image.random_flip_left_right(img).numpy()
    
    assert np.allclose(flipped1, flipped2)

