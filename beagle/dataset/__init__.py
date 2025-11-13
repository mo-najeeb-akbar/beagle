from .types import Datum, DatumValue, SerializeFn, DecompressFn
from .writer import write_dataset, write_parser_dict
from .loader import load_tfr_dataset, load_tfr_dict
from .utility import (
    serialize_float_array,
    serialize_float_or_int,
    serialize_image,
    serialize_string,
)
from .crops import (
    create_overlapping_crops,
    reconstruct_from_crops,
    compute_crop_stats,
)
from .iterator import (
    create_iterator,
    create_tfrecord_iterator,
    compute_num_crops,
    compute_welford_stats,
    count_tfrecord_samples,
    to_jax,
)
from .preprocessing import (
    FieldConfig,
    FieldType,
    compute_field_stats,
    compute_stats_for_fields,
    create_standardize_fn,
)

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
    'create_iterator',  # Main unified iterator (optimized for performance)
    'create_tfrecord_iterator',  # Backward compatibility (wraps create_iterator)
    'compute_welford_stats',
    'count_tfrecord_samples',
    'to_jax',
    
    # Crop utilities
    'create_overlapping_crops',
    'reconstruct_from_crops',
    'compute_crop_stats',
    'compute_num_crops',
    
    # Preprocessing utilities
    'FieldConfig',
    'FieldType',
    'compute_field_stats',
    'compute_stats_for_fields',
    'create_standardize_fn',
    
    # Serialization utilities
    'serialize_float_array',
    'serialize_float_or_int',
    'serialize_image',
    'serialize_string',
]
