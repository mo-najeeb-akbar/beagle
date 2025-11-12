"""
TensorFlow CPU-only configuration utilities.

Use this module to ensure TensorFlow uses CPU only, allowing JAX to use GPU.
"""
from __future__ import annotations
import os

# Set CUDA_VISIBLE_DEVICES to empty to hide GPUs from TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf

# Explicitly set TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')


def configure_tf_cpu() -> None:
    """
    Configure TensorFlow to use CPU only (has side effects: modifies TF config).
    
    Call this before importing other TensorFlow modules if you want to ensure
    TensorFlow doesn't use GPU, allowing JAX to have exclusive GPU access.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.config.set_visible_devices([], 'GPU')
    # Prevent TensorFlow from allocating GPU memory
    try:
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    except (ValueError, IndexError):
        pass  # No GPU devices available, which is what we want


# Auto-configure on import
configure_tf_cpu()

