from __future__ import annotations
from typing import Callable, Literal
import tensorflow as tf
import json
import os
import glob
from functools import partial

AUTOTUNE = tf.data.AUTOTUNE


def load_tfr_dataset(
        parser: Callable[[tf.Tensor], tf.Tensor],
        data_path: str,
        regex: str,
        shuffle: bool = True,
        cycle_length: int = 3,
        block_length: int = 3,
        verbose: Literal[0, 1, 2] = 0
) -> tf.data.Dataset:
    """
    Load TFRecord dataset with interleaved reading for performance.
    
    Args:
        parser: Function that parses a serialized example proto (typically created with functools.partial)
        data_path: Directory containing TFRecord files
        regex: Glob pattern to match TFRecord files (e.g., '*.tfrecord')
        shuffle: Whether to shuffle the file order
        cycle_length: Number of input files processed concurrently
        block_length: Number of consecutive elements to produce from each file
        verbose: Verbosity level (0: silent, 1: file count, 2: file list)
    
    Returns:
        A tf.data.Dataset ready for iteration
    """
    reg_paths = os.path.join(data_path, regex)
    files = glob.glob(reg_paths)
    if verbose == 1:
        print(f'Loader is using: {len(files)} files.')
    if verbose == 2:
        print(f'Loader is using: {files}')

    filenames_dataset = tf.data.Dataset.list_files(reg_paths, shuffle=shuffle)
    dataset = filenames_dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x).map(parser, num_parallel_calls=tf.data.AUTOTUNE),
        cycle_length=cycle_length,
        block_length=block_length,
        num_parallel_calls=AUTOTUNE
    )
    return dataset


def load_tfr_dict(
        json_path: str
) -> tuple[dict[str, tf.io.FixedLenFeature], dict[str, list[int]]]:
    """
    Load feature dictionary and shape information from a JSON schema file (has side effect: file I/O).
    
    This is used in conjunction with write_parser_dict() to parse TFRecords.
    
    Args:
        json_path: Path to the JSON schema file created by write_parser_dict()
    
    Returns:
        A tuple of (feature_dict, shape_dict):
            - feature_dict: Dictionary mapping field names to tf.io.FixedLenFeature
            - shape_dict: Dictionary mapping field names to their original shapes
    """
    with open(json_path, 'r') as f:
        js_dict = json.load(f)
        tf_dict = {}
        tf_shapes = {}
        for k, v in js_dict.items():
            if v['type'] == 'str':
                type_ = tf.string
                tf_dict[k] = tf.io.FixedLenFeature([], type_)
            else:
                shape_str = v['shape'].replace('(', '').replace(')', '').replace(' ', '')
                num_vals_list = [s for s in shape_str.split(',') if s]
                tf_shapes[k] = [int(val_) for val_ in num_vals_list]
                type_ = tf.float32

                num_vals = 1
                for val_ in tf_shapes[k]:
                    num_vals = num_vals * val_
                tf_dict[k] = tf.io.FixedLenFeature([num_vals], type_)
        return tf_dict, tf_shapes


def make_parser(
    parser: Callable,
    feature_dict: dict[str, tf.io.FixedLenFeature],
    shape_dict: dict[str, list[int]]
) -> Callable:
    # TODO: for now I will own the overhead, but need to write this later to be easier to use
    pass