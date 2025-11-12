from __future__ import annotations
from .types import Datum
from .utility import split_list
import tensorflow as tf
import numpy as np
import os
import json
import multiprocessing


def process_chunk(
    dataset: list[list[Datum]],
    output_file_pre: str,
    id_: int
) -> None:
    """Write a chunk of data to a TFRecord file (has side effects: file I/O)."""
    writer_tf = tf.io.TFRecordWriter(f'{output_file_pre}{id_}.tfrecord')
    for data_list in (dataset):
        serialized_dict = {
            datum.name: 
            datum.serialize_fn(datum.decompress_fn(datum)) for datum in data_list}
        example_proto = tf.train.Example(features=tf.train.Features(feature=serialized_dict))
        writer_tf.write(example_proto.SerializeToString())
    writer_tf.close()


def write_dataset(
        data_refs: list[list[Datum]],
        output_path: str,
        extra_identifiers: list[str] | None = None,
        num_shards: int = 1,
) -> None:
    """
    Write a dataset to TFRecord files with parallel multi-shard support (has side effects: file I/O, multiprocessing).
    
    Args:
        data_refs: List of samples, where each sample is a list of Datum objects
        output_path: Directory path where TFRecord files will be written
        extra_identifiers: Optional list of strings to append to output filenames
        num_shards: Number of separate TFRecord files to create (parallelized)
    """
    sharded_data_refs = split_list(data_refs, num_shards)
    extra_suffix = '' if extra_identifiers is None else '_' + '_'.join(extra_identifiers)
    output_file_pre = os.path.join(output_path, f'record{extra_suffix}_')

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Use pool.starmap to pass both the function and items to the worker
        pool.starmap(process_chunk, [(references, output_file_pre, shard_id)
                                               for shard_id, references in enumerate(sharded_data_refs)])


def write_parser_dict(
    data_list: list[Datum],
    output_path: str,
    output_name: str
) -> None:
    """
    Write a JSON schema file describing the structure of the dataset (has side effects: file I/O).
    
    This schema is used by load_tfr_dict() to properly parse TFRecords.
    
    Args:
        data_list: A sample list of Datum objects representing one record
        output_path: Directory path where the JSON schema will be written
        output_name: Filename for the JSON schema (e.g., 'schema.json')
    """
    res = {}

    for datum in data_list:
        k = datum.name
        v = datum.decompress_fn(datum).value
        
        if isinstance(v, np.ndarray):
            shape = v.shape
        else:
            shape = (1,)
        res[k] = {
            'type': type(v).__name__,
            'shape': f'{shape}'
        }
    with open(os.path.join(output_path, output_name), 'w') as f:
        json.dump(res, f, indent=4)
