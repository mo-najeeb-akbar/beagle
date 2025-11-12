from __future__ import annotations
from .types import Datum
import tensorflow as tf
from typing import TypeVar, Sequence

T = TypeVar('T')


def _bytes_feature(value: bytes) -> tf.train.Feature:
    """Returns a bytes_list from a string / byte (pure function)."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value: Sequence[float]) -> tf.train.Feature:
    """Returns a float_list from a float / double (pure function)."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value: int) -> tf.train.Feature:
    """Returns an int64_list from a bool / enum / int / uint (pure function)."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def split_list(list1: list[T], k: int) -> list[list[T]]:
    """Split a list into k roughly equal parts (pure function)."""
    n = len(list1)
    part_size = n // k
    remainder = n % k

    parts1: list[list[T]] = []
    taken = 0
    for i in range(k):
        next_taken = taken + part_size + (1 if i < remainder else 0)
        parts1.append(list1[taken:next_taken])
        taken = next_taken

    return parts1


def serialize_float_array(data: Datum) -> tf.train.Feature:
    """Serialize numpy array as flattened float feature (has side effect: calls numpy)."""
    return _float_feature(data.value.flatten())


def serialize_float_or_int(data: Datum) -> tf.train.Feature:
    """Serialize single float/int as float feature (pure function)."""
    return _float_feature([data.value])


def serialize_image(data: Datum) -> tf.train.Feature:
    """Serialize image array as PNG bytes (has side effect: TF encoding)."""
    return _bytes_feature(tf.io.encode_png(data.value).numpy())


def serialize_string(data: Datum) -> tf.train.Feature:
    """Serialize string as bytes feature (pure function)."""
    encoded_bytes = data.value.encode('utf-8')
    return _bytes_feature(encoded_bytes)
