"""Minimal data protocol for beagle training.

The DataIterator protocol defines the minimal interface needed for training.
Any object implementing this protocol can be used with beagle's training utilities.
"""

from typing import Protocol, Iterator, Any


class DataIterator(Protocol):
    """Minimal protocol for data iteration.

    Any iterable that yields batches (as dictionaries) can be used for training.

    Example implementations:
        - TensorFlow dataset wrapped with iterator
        - PyTorch DataLoader wrapped with dict conversion
        - Custom numpy-based iterators
        - Hugging Face dataset iterators
    """

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield batches as dictionaries.

        Each batch should be a dict mapping keys to arrays:
            {'image': array, 'label': array, ...}
        """
        ...

    def __len__(self) -> int:
        """Return number of batches per epoch.

        Optional but recommended for progress tracking.
        """
        ...


# Helper function to validate an iterator
def validate_iterator(iterator: Any) -> bool:
    """Check if an object implements the DataIterator protocol.

    Args:
        iterator: Object to validate

    Returns:
        is_valid: True if object can be used as DataIterator
    """
    has_iter = hasattr(iterator, '__iter__')
    has_len = hasattr(iterator, '__len__')

    return has_iter  # __len__ is optional


def wrap_tf_dataset(tf_dataset) -> Any:
    """Wrap a TensorFlow dataset to work with beagle.

    Args:
        tf_dataset: tf.data.Dataset instance

    Returns:
        iterator: Object implementing DataIterator protocol

    Example:
        >>> import tensorflow as tf
        >>> ds = tf.data.Dataset.from_tensor_slices({'x': [...], 'y': [...]})
        >>> ds = ds.batch(32)
        >>> iterator = wrap_tf_dataset(ds)
        >>> for batch in iterator:
        ...     # batch is a dict with JAX arrays
        ...     pass
    """
    import jax.numpy as jnp

    class TFDatasetWrapper:
        def __init__(self, dataset):
            self.dataset = dataset

        def __iter__(self):
            for batch in self.dataset:
                # Convert TF tensors to JAX arrays
                jax_batch = {}
                for k, v in batch.items():
                    if hasattr(v, 'numpy'):
                        v = v.numpy()
                    jax_batch[k] = jnp.array(v)
                yield jax_batch

        def __len__(self):
            # TF datasets may not have length
            try:
                return len(self.dataset)
            except TypeError:
                return None

    return TFDatasetWrapper(tf_dataset)


def wrap_pytorch_dataloader(dataloader) -> Any:
    """Wrap a PyTorch DataLoader to work with beagle.

    Args:
        dataloader: PyTorch DataLoader instance

    Returns:
        iterator: Object implementing DataIterator protocol

    Example:
        >>> from torch.utils.data import DataLoader
        >>> dataloader = DataLoader(dataset, batch_size=32)
        >>> iterator = wrap_pytorch_dataloader(dataloader)
    """
    import jax.numpy as jnp
    import numpy as np

    class PyTorchDataLoaderWrapper:
        def __init__(self, loader):
            self.loader = loader

        def __iter__(self):
            for batch in self.loader:
                # Convert PyTorch tensors to JAX arrays
                jax_batch = {}
                for k, v in batch.items():
                    if hasattr(v, 'numpy'):
                        v = v.numpy()
                    elif hasattr(v, 'cpu'):
                        v = v.cpu().numpy()
                    jax_batch[k] = jnp.array(np.array(v))
                yield jax_batch

        def __len__(self):
            return len(self.loader)

    return PyTorchDataLoaderWrapper(dataloader)


def simple_numpy_iterator(
    data: dict[str, Any],
    batch_size: int,
    shuffle: bool = True,
    repeat: bool = False
) -> Any:
    """Create a simple iterator from numpy arrays.

    Args:
        data: Dictionary mapping keys to numpy arrays
        batch_size: Batch size
        shuffle: Whether to shuffle data
        repeat: Whether to repeat indefinitely

    Returns:
        iterator: Object implementing DataIterator protocol

    Example:
        >>> data = {'x': np.random.rand(100, 28, 28), 'y': np.random.randint(0, 10, 100)}
        >>> iterator = simple_numpy_iterator(data, batch_size=32, shuffle=True)
        >>> for batch in iterator:
        ...     print(batch['x'].shape)  # (32, 28, 28)
    """
    import numpy as np
    import jax.numpy as jnp

    class NumpyIterator:
        def __init__(self, data, batch_size, shuffle, repeat):
            self.data = data
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.repeat = repeat

            # Get dataset size from first key
            first_key = list(data.keys())[0]
            self.size = len(data[first_key])
            self.num_batches = (self.size + batch_size - 1) // batch_size

        def __iter__(self):
            while True:
                # Shuffle indices if needed
                if self.shuffle:
                    indices = np.random.permutation(self.size)
                else:
                    indices = np.arange(self.size)

                # Yield batches
                for i in range(0, self.size, self.batch_size):
                    batch_indices = indices[i:i + self.batch_size]
                    batch = {
                        k: jnp.array(v[batch_indices])
                        for k, v in self.data.items()
                    }
                    yield batch

                if not self.repeat:
                    break

        def __len__(self):
            return self.num_batches

    return NumpyIterator(data, batch_size, shuffle, repeat)
