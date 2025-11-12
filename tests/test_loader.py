from __future__ import annotations
import pytest
import tempfile
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from functools import partial
from beagle.dataset.loader import load_tfr_dataset, load_tfr_dict
from beagle.dataset.writer import write_dataset, write_parser_dict
from beagle.dataset.types import Datum, identity
from beagle.dataset.utility import (
    serialize_float_array,
    serialize_float_or_int,
    serialize_string,
)


class TestLoadTfrDict:
    """Test load_tfr_dict function (has side effects: file I/O)."""
    
    def test_load_tfr_dict_simple(self) -> None:
        """Test loading a simple parser dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a schema file
            schema = {
                "features": {
                    "type": "ndarray",
                    "shape": "(3,)"
                },
                "label": {
                    "type": "int",
                    "shape": "(1,)"
                }
            }
            
            schema_path = Path(tmpdir) / "schema.json"
            with open(schema_path, 'w') as f:
                json.dump(schema, f)
            
            # Load it
            feature_dict, shape_dict = load_tfr_dict(str(schema_path))
            
            # Verify feature_dict
            assert "features" in feature_dict
            assert "label" in feature_dict
            assert isinstance(feature_dict["features"], tf.io.FixedLenFeature)
            assert isinstance(feature_dict["label"], tf.io.FixedLenFeature)
            
            # Verify shape_dict
            assert shape_dict["features"] == [3]
            assert shape_dict["label"] == [1]
    
    def test_load_tfr_dict_with_string(self) -> None:
        """Test loading parser dictionary with string field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema = {
                "text": {
                    "type": "str",
                    "shape": "(1,)"
                }
            }
            
            schema_path = Path(tmpdir) / "schema.json"
            with open(schema_path, 'w') as f:
                json.dump(schema, f)
            
            feature_dict, shape_dict = load_tfr_dict(str(schema_path))
            
            assert "text" in feature_dict
            # String fields should not be in shape_dict
            assert "text" not in shape_dict
    
    def test_load_tfr_dict_multidimensional(self) -> None:
        """Test loading parser dictionary with multidimensional arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema = {
                "matrix": {
                    "type": "ndarray",
                    "shape": "(2, 3)"
                }
            }
            
            schema_path = Path(tmpdir) / "schema.json"
            with open(schema_path, 'w') as f:
                json.dump(schema, f)
            
            feature_dict, shape_dict = load_tfr_dict(str(schema_path))
            
            assert shape_dict["matrix"] == [2, 3]
            # Feature dict should expect flattened array
            assert feature_dict["matrix"].shape == [6]
    
    def test_load_tfr_dict_multiple_fields(self) -> None:
        """Test loading parser dictionary with multiple fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema = {
                "field1": {"type": "ndarray", "shape": "(10,)"},
                "field2": {"type": "int", "shape": "(1,)"},
                "field3": {"type": "str", "shape": "(1,)"},
                "field4": {"type": "ndarray", "shape": "(2, 2)"},
            }
            
            schema_path = Path(tmpdir) / "schema.json"
            with open(schema_path, 'w') as f:
                json.dump(schema, f)
            
            feature_dict, shape_dict = load_tfr_dict(str(schema_path))
            
            assert len(feature_dict) == 4
            assert len(shape_dict) == 3  # field3 is string, not in shape_dict
            assert shape_dict["field1"] == [10]
            assert shape_dict["field2"] == [1]
            assert shape_dict["field4"] == [2, 2]


class TestLoadTfrDataset:
    """Test load_tfr_dataset function (has side effects: file I/O)."""
    
    def test_load_tfr_dataset_basic(self) -> None:
        """Test basic loading of TFRecord dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First, write a dataset
            data_refs = [
                [
                    Datum(
                        value=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                        name="features",
                        serialize_fn=serialize_float_array,
                        decompress_fn=identity
                    ),
                    Datum(
                        value=0,
                        name="label",
                        serialize_fn=serialize_float_or_int,
                        decompress_fn=identity
                    ),
                ],
                [
                    Datum(
                        value=np.array([4.0, 5.0, 6.0], dtype=np.float32),
                        name="features",
                        serialize_fn=serialize_float_array,
                        decompress_fn=identity
                    ),
                    Datum(
                        value=1,
                        name="label",
                        serialize_fn=serialize_float_or_int,
                        decompress_fn=identity
                    ),
                ],
            ]
            
            write_dataset(data_refs, tmpdir, num_shards=1)
            
            # Define parser
            def parser(raw_record: tf.Tensor) -> dict[str, tf.Tensor]:
                feature_description = {
                    'features': tf.io.FixedLenFeature([3], tf.float32),
                    'label': tf.io.FixedLenFeature([1], tf.float32),
                }
                return tf.io.parse_single_example(raw_record, feature_description)
            
            # Load dataset
            dataset = load_tfr_dataset(
                parser=parser,
                data_path=tmpdir,
                regex="*.tfrecord",
                shuffle=False,
                verbose=0
            )
            
            # Verify we can iterate and get correct data
            records = list(dataset.take(2))
            assert len(records) == 2
            
            # Check first record
            assert 'features' in records[0]
            assert 'label' in records[0]
            np.testing.assert_array_almost_equal(
                records[0]['features'].numpy(),
                np.array([1.0, 2.0, 3.0])
            )
    
    def test_load_tfr_dataset_multiple_files(self) -> None:
        """Test loading dataset from multiple TFRecord files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write multiple shards
            data_refs = [
                [
                    Datum(
                        value=float(i),
                        name="value",
                        serialize_fn=serialize_float_or_int,
                        decompress_fn=identity
                    ),
                ]
                for i in range(10)
            ]
            
            write_dataset(data_refs, tmpdir, num_shards=3)
            
            # Define parser
            def parser(raw_record: tf.Tensor) -> dict[str, tf.Tensor]:
                feature_description = {
                    'value': tf.io.FixedLenFeature([1], tf.float32),
                }
                return tf.io.parse_single_example(raw_record, feature_description)
            
            # Load dataset
            dataset = load_tfr_dataset(
                parser=parser,
                data_path=tmpdir,
                regex="*.tfrecord",
                shuffle=False,
                verbose=0
            )
            
            # Should be able to read all 10 records
            records = list(dataset)
            assert len(records) == 10
    
    def test_load_tfr_dataset_with_shuffle(self) -> None:
        """Test loading dataset with shuffle enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write dataset
            data_refs = [
                [
                    Datum(
                        value=float(i),
                        name="value",
                        serialize_fn=serialize_float_or_int,
                        decompress_fn=identity
                    ),
                ]
                for i in range(5)
            ]
            
            write_dataset(data_refs, tmpdir, num_shards=1)
            
            def parser(raw_record: tf.Tensor) -> dict[str, tf.Tensor]:
                feature_description = {
                    'value': tf.io.FixedLenFeature([1], tf.float32),
                }
                return tf.io.parse_single_example(raw_record, feature_description)
            
            # Load with shuffle - should not raise error
            dataset = load_tfr_dataset(
                parser=parser,
                data_path=tmpdir,
                regex="*.tfrecord",
                shuffle=True,
                verbose=0
            )
            
            records = list(dataset)
            assert len(records) == 5
    
    def test_load_tfr_dataset_verbose_modes(self) -> None:
        """Test verbose output modes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_refs = [
                [
                    Datum(
                        value=1.0,
                        name="value",
                        serialize_fn=serialize_float_or_int,
                        decompress_fn=identity
                    ),
                ]
            ]
            
            write_dataset(data_refs, tmpdir, num_shards=1)
            
            def parser(raw_record: tf.Tensor) -> dict[str, tf.Tensor]:
                return tf.io.parse_single_example(
                    raw_record,
                    {'value': tf.io.FixedLenFeature([1], tf.float32)}
                )
            
            # Test verbose=0 (silent)
            dataset = load_tfr_dataset(
                parser=parser,
                data_path=tmpdir,
                regex="*.tfrecord",
                verbose=0
            )
            assert dataset is not None
            
            # Test verbose=1 (file count)
            dataset = load_tfr_dataset(
                parser=parser,
                data_path=tmpdir,
                regex="*.tfrecord",
                verbose=1
            )
            assert dataset is not None
            
            # Test verbose=2 (file list)
            dataset = load_tfr_dataset(
                parser=parser,
                data_path=tmpdir,
                regex="*.tfrecord",
                verbose=2
            )
            assert dataset is not None
    
    def test_load_tfr_dataset_custom_interleave_params(self) -> None:
        """Test loading with custom interleave parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_refs = [
                [
                    Datum(
                        value=float(i),
                        name="value",
                        serialize_fn=serialize_float_or_int,
                        decompress_fn=identity
                    ),
                ]
                for i in range(20)
            ]
            
            write_dataset(data_refs, tmpdir, num_shards=4)
            
            def parser(raw_record: tf.Tensor) -> dict[str, tf.Tensor]:
                return tf.io.parse_single_example(
                    raw_record,
                    {'value': tf.io.FixedLenFeature([1], tf.float32)}
                )
            
            # Load with custom cycle_length and block_length
            dataset = load_tfr_dataset(
                parser=parser,
                data_path=tmpdir,
                regex="*.tfrecord",
                shuffle=False,
                cycle_length=2,
                block_length=1,
                verbose=0
            )
            
            records = list(dataset)
            assert len(records) == 20


class TestIntegrationWriteAndLoad:
    """Integration tests for writing and loading TFRecords."""
    
    def test_roundtrip_simple_dataset(self) -> None:
        """Test full roundtrip: write, create schema, load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data
            data_refs = [
                [
                    Datum(
                        value=np.array([1.0, 2.0], dtype=np.float32),
                        name="features",
                        serialize_fn=serialize_float_array,
                        decompress_fn=identity
                    ),
                    Datum(
                        value=i,
                        name="label",
                        serialize_fn=serialize_float_or_int,
                        decompress_fn=identity
                    ),
                ]
                for i in range(5)
            ]
            
            # Write dataset
            write_dataset(data_refs, tmpdir, num_shards=1)
            
            # Write schema
            write_parser_dict(data_refs[0], tmpdir, "schema.json")
            
            # Load schema
            feature_dict, shape_dict = load_tfr_dict(str(Path(tmpdir) / "schema.json"))
            
            # Create parser using schema
            def parser(raw_record: tf.Tensor) -> dict[str, tf.Tensor]:
                parsed = tf.io.parse_single_example(raw_record, feature_dict)
                # Reshape using shape_dict
                return {
                    'features': tf.reshape(parsed['features'], shape_dict['features']),
                    'label': parsed['label'][0],
                }
            
            # Load dataset
            dataset = load_tfr_dataset(
                parser=parser,
                data_path=tmpdir,
                regex="*.tfrecord",
                shuffle=False,
                verbose=0
            )
            
            # Verify data
            records = list(dataset)
            assert len(records) == 5
            
            for i, record in enumerate(records):
                np.testing.assert_array_almost_equal(
                    record['features'].numpy(),
                    np.array([1.0, 2.0])
                )
                assert record['label'].numpy() == pytest.approx(float(i))
    
    def test_roundtrip_with_strings(self) -> None:
        """Test roundtrip with string data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_refs = [
                [
                    Datum(
                        value=f"sample_{i}",
                        name="id",
                        serialize_fn=serialize_string,
                        decompress_fn=identity
                    ),
                    Datum(
                        value=float(i),
                        name="value",
                        serialize_fn=serialize_float_or_int,
                        decompress_fn=identity
                    ),
                ]
                for i in range(3)
            ]
            
            write_dataset(data_refs, tmpdir, num_shards=1)
            write_parser_dict(data_refs[0], tmpdir, "schema.json")
            
            feature_dict, shape_dict = load_tfr_dict(str(Path(tmpdir) / "schema.json"))
            
            def parser(raw_record: tf.Tensor) -> dict[str, tf.Tensor]:
                return tf.io.parse_single_example(raw_record, feature_dict)
            
            dataset = load_tfr_dataset(
                parser=parser,
                data_path=tmpdir,
                regex="*.tfrecord",
                shuffle=False,
                verbose=0
            )
            
            records = list(dataset)
            assert len(records) == 3
            
            for i, record in enumerate(records):
                assert record['id'].numpy().decode('utf-8') == f"sample_{i}"
                assert record['value'].numpy()[0] == pytest.approx(float(i))

