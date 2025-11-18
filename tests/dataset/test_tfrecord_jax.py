"""Tests for TFRecord to JAX utilities."""
from __future__ import annotations

import pytest
import numpy as np
import jax.numpy as jnp
import tensorflow as tf
from pathlib import Path

from beagle.dataset.iterator import (
    to_jax,
    count_tfrecord_samples,
)


def test_to_jax_converts_tensors():
    """Test conversion from TF tensors to JAX arrays."""
    tf_dict = {
        'image': tf.constant([[1.0, 2.0], [3.0, 4.0]]),
        'label': tf.constant([0, 1]),
    }
    
    jax_dict = to_jax(tf_dict, dtype=jnp.float32)
    
    assert isinstance(jax_dict['image'], jnp.ndarray)
    assert isinstance(jax_dict['label'], jnp.ndarray)
    assert jax_dict['image'].dtype == jnp.float32


def test_to_jax_preserves_structure():
    """Test that to_jax preserves dictionary structure."""
    tf_dict = {
        'a': {'b': tf.constant([1, 2, 3])},
        'c': tf.constant([4, 5, 6]),
    }
    
    jax_dict = to_jax(tf_dict)
    
    assert 'a' in jax_dict
    assert 'b' in jax_dict['a']
    assert 'c' in jax_dict


@pytest.fixture
def sample_tfrecord(tmp_path: Path) -> str:
    """Create a sample TFRecord file for testing."""
    tfrecord_path = tmp_path / "test.tfrecord"
    
    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        for _ in range(10):
            image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img_str = tf.io.encode_png(image).numpy()
            
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_str]))}
                )
            )
            writer.write(example.SerializeToString())
    
    return str(tfrecord_path)


def test_count_tfrecord_samples(sample_tfrecord: str):
    """Test counting samples in TFRecord."""
    count = count_tfrecord_samples([sample_tfrecord])
    assert count == 10


def test_count_tfrecord_samples_multiple_files(tmp_path: Path):
    """Test counting samples across multiple TFRecord files."""
    files = []
    for i in range(3):
        tfrecord_path = tmp_path / f"test_{i}.tfrecord"
        
        with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
            for _ in range(5):
                image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                img_str = tf.io.encode_png(image).numpy()
                
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_str]))}
                    )
                )
                writer.write(example.SerializeToString())
        
        files.append(str(tfrecord_path))
    
    count = count_tfrecord_samples(files)
    assert count == 15  # 3 files * 5 samples each

