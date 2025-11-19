from __future__ import annotations

import random
import numpy as np
import tensorflow as tf


def set_global_seed(seed: int) -> None:
    """
    Set random seeds for all libraries used in dataset/augmentation pipeline.
    
    This ensures reproducibility across:
    - Python's random module (used in train/val splits)
    - NumPy random (used in data processing and albumentations)
    - TensorFlow random (used in dataset shuffling and augmentations)
    - TensorFlow dataset operations (shuffling, sampling)
    
    Note: JAX uses explicit random keys (jax.random.PRNGKey), so seed 
    those separately in your training code.
    
    Args:
        seed: Random seed value
    
    Example:
        >>> from beagle.dataset import set_global_seed, create_iterator
        >>> 
        >>> # Set seed before creating iterators
        >>> set_global_seed(42)
        >>> 
        >>> # Now all randomness is reproducible
        >>> (train_iter, _), (val_iter, _) = create_iterator(
        ...     "data/*.tfrecord",
        ...     batch_size=32,
        ...     val_split=0.2,
        ...     augment_fn=my_augment_fn,
        ...     shuffle=True,
        ... )
    """
    # Python's random module (used in split_files_train_val)
    random.seed(seed)
    
    # NumPy random (used in data processing and albumentations)
    np.random.seed(seed)
    
    # TensorFlow random ops (used in augmentations)
    tf.random.set_seed(seed)
    
    # TensorFlow dataset operations (ensures shuffling is deterministic)
    # Note: TF 2.x uses global seed from set_seed above


def set_tf_deterministic(enable: bool = True) -> None:
    """
    Enable deterministic operations in TensorFlow.
    
    This makes TensorFlow operations deterministic at the cost of performance.
    Use this if you need fully reproducible results and seeding alone isn't enough.
    
    Args:
        enable: Whether to enable deterministic mode
    
    Warning:
        This can slow down training significantly. Only use for debugging
        or when absolute reproducibility is required.
    
    Example:
        >>> from beagle.dataset import set_global_seed, set_tf_deterministic
        >>> 
        >>> set_global_seed(42)
        >>> set_tf_deterministic(True)  # Force deterministic ops
    """
    if enable:
        # TF 2.9+ API
        try:
            tf.config.experimental.enable_op_determinism()
        except AttributeError:
            # Fallback for older TF versions
            import os
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

