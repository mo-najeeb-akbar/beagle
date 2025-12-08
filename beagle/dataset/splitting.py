from __future__ import annotations

from typing import Iterator, Callable
from functools import partial

import tensorflow as tf
import jax.numpy as jnp

from .iterator import to_jax, count_tfrecord_samples, compute_num_crops
from .crops import create_overlapping_crops


def create_train_val_iterators(
    files: list[str],
    parser: Callable[[tf.Tensor], dict[str, tf.Tensor]],
    batch_size: int,
    val_fraction: float,
    *,
    shuffle: bool = True,
    seed: int = 42,
    repeat: bool = True,
    crop_size: int | None = None,
    stride: int | None = None,
    image_shape: tuple[int, int] | None = None,
    augment_fn: Callable[[dict[str, tf.Tensor]], dict[str, tf.Tensor]] | None = None,
) -> tuple[Iterator[dict], Iterator[dict], int, int]:
    """
    Create train and validation iterators by splitting TFRecord dataset at sample level.

    This function splits the dataset at the sample level (not file level), ensuring
    precise control over the train/validation split ratio. The split is deterministic
    when shuffle=True with the same seed. Optionally supports extracting overlapping
    crops from images before splitting.

    Args:
        files: List of TFRecord file paths to read
        parser: Parser function that converts serialized examples to dictionaries
        batch_size: Number of samples per batch
        val_fraction: Fraction of samples to use for validation (0.0 to 1.0)
        shuffle: Whether to shuffle samples before splitting (default: True)
        seed: Random seed for shuffling (default: 42)
        repeat: Whether to repeat datasets indefinitely (default: True)
        crop_size: Optional size for extracting overlapping crops from images
        stride: Stride for overlapping crops (required if crop_size is set)
        image_shape: (height, width) of images (required if crop_size is set)
        augment_fn: Optional augmentation function to apply to training data only

    Returns:
        Tuple of (train_iterator, val_iterator, n_train_batches, n_val_batches)
        - train_iterator: Iterator yielding training batches as JAX arrays
        - val_iterator: Iterator yielding validation batches as JAX arrays
        - n_train_batches: Number of batches in training set
        - n_val_batches: Number of batches in validation set

    Raises:
        ValueError: If val_fraction is not between 0.0 and 1.0
        ValueError: If batch_size is less than 1
        ValueError: If files list is empty
        ValueError: If crop_size is set but stride or image_shape is not

    Example:
        >>> # Create iterators with 20% validation split
        >>> train_iter, val_iter, n_train, n_val = create_train_val_iterators(
        ...     files=['data.tfrecord'],
        ...     parser=my_parser,
        ...     batch_size=32,
        ...     val_fraction=0.2,
        ...     shuffle=True,
        ...     seed=42
        ... )
        >>>
        >>> # With cropping
        >>> train_iter, val_iter, n_train, n_val = create_train_val_iterators(
        ...     files=['data.tfrecord'],
        ...     parser=my_parser,
        ...     batch_size=32,
        ...     val_fraction=0.2,
        ...     crop_size=256,
        ...     stride=192,
        ...     image_shape=(512, 512)
        ... )
        >>>
        >>> # Use the iterators
        >>> train_batch = next(train_iter)  # Dict with JAX arrays
        >>> print(f"Training batches: {n_train}, Validation batches: {n_val}")
    """
    # Input validation
    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError(f"val_fraction must be between 0.0 and 1.0, got {val_fraction}")

    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    if not files:
        raise ValueError("files list cannot be empty")

    # Validate crop parameters
    if crop_size is not None:
        if stride is None:
            raise ValueError("stride must be provided when crop_size is set")
        if image_shape is None:
            raise ValueError("image_shape must be provided when crop_size is set")

    # Count total images
    n_images = count_tfrecord_samples(files)

    if n_images == 0:
        raise ValueError("No samples found in TFRecord files")

    # Calculate total samples (accounting for crops if applicable)
    if crop_size is not None:
        crops_per_image = compute_num_crops(
            image_shape[0], image_shape[1], crop_size, stride
        )
        n_samples = n_images * crops_per_image
    else:
        n_samples = n_images

    # Compute split sizes
    n_val_samples = int(n_samples * val_fraction)
    n_train_samples = n_samples - n_val_samples

    # Compute number of batches (drop remainder)
    n_train_batches = n_train_samples // batch_size
    n_val_batches = n_val_samples // batch_size

    # Create base dataset
    n_parallel = 6
    base_dataset = tf.data.TFRecordDataset(
        files,
        num_parallel_reads=n_parallel
    ).map(parser, num_parallel_calls=n_parallel)

    # Apply cropping if requested (before caching)
    if crop_size is not None:
        crop_fn = partial(create_overlapping_crops, crop_size=crop_size, stride=stride)
        base_dataset = base_dataset.map(crop_fn, num_parallel_calls=n_parallel)
        base_dataset = base_dataset.unbatch()  # Flatten crops into individual samples

    # Cache the parsed (and possibly cropped) data before splitting
    base_dataset = base_dataset.cache()

    # Shuffle if requested (must happen before split for deterministic results)
    if shuffle:
        # Use large buffer to ensure good shuffling across entire dataset
        buffer_size = min(n_samples, 10000)
        base_dataset = base_dataset.shuffle(
            buffer_size=buffer_size,
            seed=seed,
            reshuffle_each_iteration=False  # Consistent split across epochs
        )

    # Split into train and validation
    train_dataset = base_dataset.skip(n_val_samples).take(n_train_samples)
    val_dataset = base_dataset.take(n_val_samples)

    # Apply augmentation to training data only (after cache, in TF pipeline)
    if augment_fn is not None:
        train_dataset = train_dataset.map(augment_fn, num_parallel_calls=n_parallel)

    # Configure train dataset
    if repeat:
        train_dataset = train_dataset.repeat()

    if shuffle:
        # Reshuffle training data each epoch
        # Use smaller buffer for crops to reduce memory usage
        if crop_size is not None:
            train_shuffle_buffer = min(10 * batch_size, n_train_samples // 2)
        else:
            train_shuffle_buffer = min(16 * batch_size, n_train_samples)
        train_dataset = train_dataset.shuffle(
            buffer_size=train_shuffle_buffer,
            seed=seed + 1,  # Different seed from initial shuffle
            reshuffle_each_iteration=True
        )

    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    # Use larger prefetch for crops to maintain throughput
    prefetch_size = 8 if crop_size is not None else 4
    train_dataset = train_dataset.prefetch(prefetch_size)

    # Configure validation dataset
    if repeat:
        val_dataset = val_dataset.repeat()

    # Typically don't shuffle validation, but batch it
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
    val_dataset = val_dataset.prefetch(4)

    # Convert to JAX iterators
    train_iterator = map(partial(to_jax, dtype=jnp.float32), train_dataset)
    val_iterator = map(partial(to_jax, dtype=jnp.float32), val_dataset)

    return train_iterator, val_iterator, n_train_batches, n_val_batches
