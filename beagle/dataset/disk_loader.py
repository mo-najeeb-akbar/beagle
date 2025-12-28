"""
Direct disk loading for small datasets that fit in memory.

Simple, explicit approach - you provide aligned file paths, we load them into numpy arrays.
"""
from __future__ import annotations

from typing import Iterator, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf


def to_jax(tensor_dict: dict, dtype: jnp.dtype = jnp.float32) -> dict:
    """Convert a dictionary of tensors to JAX arrays.

    Handles TensorFlow tensors by calling .numpy() first.
    """
    def convert(x):
        # Handle TensorFlow tensors
        if hasattr(x, 'numpy'):
            x = x.numpy()
        return jnp.array(x, dtype=dtype)
    return jax.tree.map(convert, tensor_dict)


def _process_single_sample(
    sample_dict: dict[str, str],
    field_names: list[str],
    field_loaders: dict[str, Callable[[str], np.ndarray]],
    field_transforms: dict[str, Callable[[np.ndarray], np.ndarray]],
    sample_transform: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None,
) -> dict[str, np.ndarray]:
    """Process a single sample (pure function for parallel execution).
    
    Args:
        sample_dict: Dict mapping field name -> file path
        field_names: List of expected field names
        field_loaders: Dict mapping field name -> load function
        field_transforms: Dict mapping field name -> transform function
        sample_transform: Optional function for paired transforms
        
    Returns:
        Dict mapping field name -> transformed array
    """
    # Verify all fields are present
    if set(sample_dict.keys()) != set(field_names):
        raise ValueError(
            f"Inconsistent fields: expected {field_names}, got {list(sample_dict.keys())}"
        )

    # Load each field
    sample_arrays = {}
    for field_name in field_names:
        path = sample_dict[field_name]
        load_fn = field_loaders[field_name]
        arr = load_fn(path)
        sample_arrays[field_name] = arr

    # Apply sample-level transform (e.g., Albumentations)
    if sample_transform is not None:
        sample_arrays = sample_transform(sample_arrays)

    # Apply field-level transforms (e.g., normalization)
    result = {}
    for field_name in field_names:
        arr = sample_arrays[field_name]
        transform_fn = field_transforms.get(field_name)
        if transform_fn is not None:
            arr = transform_fn(arr)
        result[field_name] = arr

    return result


def load_fields_from_disk(
    file_list: list[dict[str, str]],
    field_loaders: dict[str, Callable[[str], np.ndarray]],
    field_transforms: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    sample_transform: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None = None,
    num_workers: int = 1,
) -> dict[str, np.ndarray]:
    """Load multiple fields from disk into a dictionary of numpy arrays.

    Simple and explicit - you provide aligned file paths, we load them.

    Args:
        file_list: List of dicts mapping field name -> file path
            Example: [
                {'image': 'data/images/001.png', 'mask': 'data/masks/001.png'},
                {'image': 'data/images/002.png', 'mask': 'data/masks/002.png'},
            ]
        field_loaders: Dict mapping field name -> load function
        field_transforms: Optional dict mapping field name -> transform function
            (applied after sample_transform)
        sample_transform: Optional function that transforms entire sample dict
            (applied before field_transforms, useful for paired augmentations like Albumentations)
        num_workers: Number of parallel workers for loading (default: 1 = sequential)
            Use > 1 for faster loading on multi-core systems

    Returns:
        Dict mapping field name -> numpy array [N, ...]

    Example (segmentation):
        >>> file_list = [
        ...     {'image': 'data/images/001.png', 'mask': 'data/masks/001.png'},
        ...     {'image': 'data/images/002.png', 'mask': 'data/masks/002.png'},
        ... ]
        >>>
        >>> data = load_fields_from_disk(
        ...     file_list=file_list,
        ...     field_loaders={
        ...         'image': lambda p: load_image(p, grayscale=True).astype(np.float32),
        ...         'mask': lambda p: load_image(p, grayscale=True).astype(np.float32),
        ...     },
        ...     field_transforms={
        ...         'image': lambda x: (x - 127.5) / 127.5,  # Normalize
        ...         'mask': lambda x: x,  # Keep as-is
        ...     }
        ... )
        >>> # data = {'image': [N, H, W, 1], 'mask': [N, H, W, 1]}

    Example (with Albumentations):
        >>> import albumentations as A
        >>> 
        >>> albu_transform = A.Compose([
        ...     A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.7),
        ...     A.ElasticTransform(p=0.3),
        ... ])
        >>> 
        >>> def apply_albu(sample):
        ...     # Remove channel dimension for albumentations
        ...     img = sample['image'][:, :, 0] if sample['image'].ndim == 3 else sample['image']
        ...     msk = sample['mask'][:, :, 0] if sample['mask'].ndim == 3 else sample['mask']
        ...     
        ...     augmented = albu_transform(image=img, mask=msk)
        ...     
        ...     # Add channel dimension back
        ...     return {
        ...         'image': augmented['image'][:, :, np.newaxis],
        ...         'mask': augmented['mask'][:, :, np.newaxis],
        ...     }
        >>> 
        >>> data = load_fields_from_disk(
        ...     file_list=file_list,
        ...     field_loaders={'image': load_image, 'mask': load_mask},
        ...     sample_transform=apply_albu,  # Apply Albumentations
        ...     field_transforms={
        ...         'image': lambda x: (x - 127.5) / 127.5,  # Normalize after augmentation
        ...         'mask': lambda x: x.astype(np.float32),
        ...     }
        ... )

    Example (classification):
        >>> file_list = [
        ...     {'image': 'data/001.jpg', 'label': 'labels/001.npy'},
        ...     {'image': 'data/002.jpg', 'label': 'labels/002.npy'},
        ... ]
        >>>
        >>> data = load_fields_from_disk(
        ...     file_list=file_list,
        ...     field_loaders={'image': load_image_fn, 'label': np.load},
        ...     field_transforms={'image': lambda x: x / 255.0},
        ... )
    """
    if len(file_list) == 0:
        raise ValueError("file_list is empty")

    field_transforms = field_transforms or {}

    # Get all field names from first entry
    field_names = list(file_list[0].keys())

    # Load all samples
    n_samples = len(file_list)
    print(f"Loading {n_samples} samples (workers={num_workers})...")

    # Sequential execution (simpler, less overhead)
    if num_workers == 1:
        processed_samples = [
            _process_single_sample(
                sample_dict=sample_dict,
                field_names=field_names,
                field_loaders=field_loaders,
                field_transforms=field_transforms,
                sample_transform=sample_transform,
            )
            for sample_dict in file_list
        ]
    # Parallel execution
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = [
                executor.submit(
                    _process_single_sample,
                    sample_dict=sample_dict,
                    field_names=field_names,
                    field_loaders=field_loaders,
                    field_transforms=field_transforms,
                    sample_transform=sample_transform,
                )
                for sample_dict in file_list
            ]
            
            # Collect results in order
            processed_samples = [future.result() for future in futures]

    # Organize by field and stack
    field_data = {name: [] for name in field_names}
    for sample_arrays in processed_samples:
        for field_name in field_names:
            field_data[field_name].append(sample_arrays[field_name])

    # Stack into arrays
    result = {}
    for field_name in field_names:
        result[field_name] = np.stack(field_data[field_name], axis=0)
        print(f"  {field_name}: {result[field_name].shape}")

    return result


def create_disk_iterator(
    data: dict[str, np.ndarray],
    batch_size: int = 8,
    shuffle: bool = True,
    repeat: bool = True,
    augment_fn: Callable[[dict], dict] | None = None,
    seed: int = 42,
) -> Iterator[dict[str, jnp.ndarray]]:
    """Create iterator from in-memory numpy data.

    Applies augmentation and batching to data already loaded into memory.
    Perfect for small datasets where you want heavy augmentation.

    Args:
        data: Dictionary of numpy arrays (e.g., {'image': images, 'mask': masks})
        batch_size: Batch size
        shuffle: Shuffle data each epoch
        repeat: Repeat indefinitely (with different augmentations each epoch)
        augment_fn: TensorFlow augmentation function (takes dict, returns dict)
        seed: Random seed

    Yields:
        Batches as dict of JAX arrays

    Example:
        >>> # Load data
        >>> data = load_fields_from_disk(file_list, field_loaders, field_transforms)
        >>>
        >>> # Define augmentation
        >>> from beagle.augmentations import compose, random_flip_left_right, random_rotate_90
        >>> aug_fn = compose(
        ...     random_flip_left_right(fields=['image', 'mask']),
        ...     random_rotate_90(fields=['image', 'mask']),
        ... )
        >>>
        >>> # Create iterator with heavy augmentation
        >>> iterator = create_disk_iterator(
        ...     data=data,
        ...     batch_size=8,
        ...     augment_fn=aug_fn,
        ...     repeat=True,
        ... )
        >>>
        >>> for batch in iterator:
        ...     # batch: dict of JAX arrays
        ...     train_step(state, batch)
    """

    # Get dataset size
    n_samples = len(next(iter(data.values())))

    # Create TensorFlow dataset from numpy arrays
    dataset = tf.data.Dataset.from_tensor_slices(data)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=n_samples, seed=seed, reshuffle_each_iteration=True)

    if repeat:
        dataset = dataset.repeat()

    # Apply augmentation BEFORE batching (so each sample gets different augmentation)
    if augment_fn is not None:
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Convert to JAX iterator
    return map(partial(to_jax, dtype=jnp.float32), dataset)