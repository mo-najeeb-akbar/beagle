from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Callable, TypeVar, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import tensorflow as tf

T = TypeVar('T')

# Type aliases for better IDE support
DatumValue = Union[np.ndarray, int, float, str, tuple]

if TYPE_CHECKING:
    SerializeFn = Callable[['Datum'], 'tf.train.Feature']
else:
    SerializeFn = Callable[['Datum'], Any]

DecompressFn = Callable[['Datum'], 'Datum']


def identity(x: T) -> T:
    return x


def get_function_name(func: Callable | None) -> str:
    """Retrieve the function's name, handling cases where it may not be directly available."""
    if func is None:
        return ''
    return getattr(func, '__name__', repr(func))


@dataclass(frozen=True)
class Datum:
    """
    A named data element with serialization logic for TFRecord writing.
    
    Attributes:
        value: The actual data (numpy array, int, float, str, or tuple)
        name: Field name in the TFRecord
        serialize_fn: Function to serialize the datum to tf.train.Feature
        decompress_fn: Preprocessing function before serialization
    """
    value: DatumValue
    name: str
    serialize_fn: SerializeFn
    decompress_fn: DecompressFn

    def __post_init__(self):
        if not isinstance(self.value, (np.ndarray, int, float, str, tuple)):
            raise TypeError(
                f"Value must be an instance of np.ndarray, int, float, str, or tuple -- got {type(self.value).__name__}")

    def __str__(self):
        return (f'Name: {self.name}\n'
                f'    Value: {self.value}\n'
                f'    Decompress Fn: {get_function_name(self.decompress_fn)}\n'
                f'    Serialze Fn: {get_function_name(self.serialize_fn)}')
