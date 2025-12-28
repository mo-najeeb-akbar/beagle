"""Data utilities for beagle.

Minimal protocol-based data loading that works with any data source.
"""

from .protocol import (
    DataIterator,
    validate_iterator,
    wrap_tf_dataset,
    wrap_pytorch_dataloader,
    simple_numpy_iterator,
)

__all__ = [
    'DataIterator',
    'validate_iterator',
    'wrap_tf_dataset',
    'wrap_pytorch_dataloader',
    'simple_numpy_iterator',
]
