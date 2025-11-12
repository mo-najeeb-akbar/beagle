from .types import Datum, DatumValue, SerializeFn, DecompressFn
from .writer import write_dataset, write_parser_dict
from .loader import load_tfr_dataset, load_tfr_dict
from .utility import (
    serialize_float_array,
    serialize_float_or_int,
    serialize_image,
    serialize_string,
)
from .tfrecord_jax import (
    create_tfrecord_iterator,
    compute_welford_stats,
    count_tfrecord_samples,
    to_jax,
    make_image_parser,
)

# Configure TensorFlow to use CPU only (for JAX GPU compatibility)
# This is imported automatically to ensure TF doesn't grab GPU
try:
    from . import tf_config  # noqa: F401
except ImportError:
    pass  # tf_config is optional

__all__ = [
    # Core types
    'Datum',
    'DatumValue',
    'SerializeFn',
    'DecompressFn',
    
    # Writer functions
    'write_dataset',
    'write_parser_dict',
    
    # Loader functions
    'load_tfr_dataset',
    'load_tfr_dict',
    
    # TFRecord -> JAX utilities
    'create_tfrecord_iterator',
    'compute_welford_stats',
    'count_tfrecord_samples',
    'to_jax',
    'make_image_parser',
    
    # Serialization utilities
    'serialize_float_array',
    'serialize_float_or_int',
    'serialize_image',
    'serialize_string',
]
