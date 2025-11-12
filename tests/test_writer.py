from __future__ import annotations
import pytest
import tempfile
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from beagle.dataset.writer import write_dataset, write_parser_dict, process_chunk
from beagle.dataset.types import Datum, identity
from beagle.dataset.utility import (
    serialize_float_array,
    serialize_float_or_int,
    serialize_string,
)


# Module-level function for picklability
def double_value(d: Datum) -> Datum:
    """Double the value before serialization."""
    return Datum(
        value=d.value * 2,
        name=d.name,
        serialize_fn=d.serialize_fn,
        decompress_fn=d.decompress_fn
    )


class TestWriteDataset:
    """Test write_dataset function (has side effects: file I/O)."""
    
    def test_write_dataset_single_shard(self) -> None:
        """Test writing dataset to a single TFRecord file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample data
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
            
            # Check that file was created
            files = list(Path(tmpdir).glob("*.tfrecord"))
            assert len(files) == 1
            assert files[0].name == "record_0.tfrecord"
            
            # Verify file is not empty
            assert files[0].stat().st_size > 0
    
    def test_write_dataset_multiple_shards(self) -> None:
        """Test writing dataset to multiple TFRecord files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create more sample data
            data_refs = [
                [
                    Datum(
                        value=np.array([float(i)], dtype=np.float32),
                        name="value",
                        serialize_fn=serialize_float_array,
                        decompress_fn=identity
                    ),
                ]
                for i in range(10)
            ]
            
            write_dataset(data_refs, tmpdir, num_shards=3)
            
            # Check that 3 files were created
            files = sorted(Path(tmpdir).glob("*.tfrecord"))
            assert len(files) == 3
            assert files[0].name == "record_0.tfrecord"
            assert files[1].name == "record_1.tfrecord"
            assert files[2].name == "record_2.tfrecord"
            
            # All files should have data
            for f in files:
                assert f.stat().st_size > 0
    
    def test_write_dataset_with_extra_identifiers(self) -> None:
        """Test writing dataset with extra identifiers in filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_refs = [
                [
                    Datum(
                        value=42,
                        name="value",
                        serialize_fn=serialize_float_or_int,
                        decompress_fn=identity
                    ),
                ]
            ]
            
            write_dataset(
                data_refs,
                tmpdir,
                extra_identifiers=["train", "v1"],
                num_shards=1
            )
            
            files = list(Path(tmpdir).glob("*.tfrecord"))
            assert len(files) == 1
            assert files[0].name == "record_train_v1_0.tfrecord"
    
    def test_write_dataset_empty_list(self) -> None:
        """Test writing an empty dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_refs: list[list[Datum]] = []
            
            write_dataset(data_refs, tmpdir, num_shards=1)
            
            # File should still be created
            files = list(Path(tmpdir).glob("*.tfrecord"))
            assert len(files) == 1
    
    def test_write_dataset_with_decompress_fn(self) -> None:
        """Test writing dataset with non-identity decompress function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_refs = [
                [
                    Datum(
                        value=np.array([1.0, 2.0], dtype=np.float32),
                        name="value",
                        serialize_fn=serialize_float_array,
                        decompress_fn=double_value
                    ),
                ]
            ]
            
            write_dataset(data_refs, tmpdir, num_shards=1)
            
            # Read back and verify the value was doubled
            files = list(Path(tmpdir).glob("*.tfrecord"))
            dataset = tf.data.TFRecordDataset(str(files[0]))
            
            for raw_record in dataset.take(1):
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                values = list(example.features.feature["value"].float_list.value)
                assert values == [2.0, 4.0]  # Doubled from [1.0, 2.0]
    
    def test_write_dataset_mixed_types(self) -> None:
        """Test writing dataset with mixed data types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_refs = [
                [
                    Datum(
                        value=np.array([1.0, 2.0], dtype=np.float32),
                        name="features",
                        serialize_fn=serialize_float_array,
                        decompress_fn=identity
                    ),
                    Datum(
                        value=42,
                        name="label",
                        serialize_fn=serialize_float_or_int,
                        decompress_fn=identity
                    ),
                    Datum(
                        value="sample_id_123",
                        name="id",
                        serialize_fn=serialize_string,
                        decompress_fn=identity
                    ),
                ]
            ]
            
            write_dataset(data_refs, tmpdir, num_shards=1)
            
            # Verify all fields are present
            files = list(Path(tmpdir).glob("*.tfrecord"))
            dataset = tf.data.TFRecordDataset(str(files[0]))
            
            for raw_record in dataset.take(1):
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                
                assert "features" in example.features.feature
                assert "label" in example.features.feature
                assert "id" in example.features.feature


class TestProcessChunk:
    """Test process_chunk function (has side effects: file I/O)."""
    
    def test_process_chunk_direct(self) -> None:
        """Test process_chunk function directly (not via multiprocessing)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file_pre = str(Path(tmpdir) / "test_")
            
            dataset = [
                [
                    Datum(
                        value=np.array([1.0, 2.0], dtype=np.float32),
                        name="features",
                        serialize_fn=serialize_float_array,
                        decompress_fn=identity
                    ),
                ]
            ]
            
            process_chunk(dataset, output_file_pre, 0)
            
            # Check file was created
            files = list(Path(tmpdir).glob("test_*.tfrecord"))
            assert len(files) == 1
            assert files[0].name == "test_0.tfrecord"
            assert files[0].stat().st_size > 0
            
            # Verify content
            dataset_tf = tf.data.TFRecordDataset(str(files[0]))
            for raw_record in dataset_tf.take(1):
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                assert "features" in example.features.feature


class TestWriteParserDict:
    """Test write_parser_dict function (has side effects: file I/O)."""
    
    def test_write_parser_dict_simple(self) -> None:
        """Test writing parser dictionary to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_list = [
                Datum(
                    value=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                    name="features",
                    serialize_fn=serialize_float_array,
                    decompress_fn=identity
                ),
                Datum(
                    value=42,
                    name="label",
                    serialize_fn=serialize_float_or_int,
                    decompress_fn=identity
                ),
            ]
            
            write_parser_dict(data_list, tmpdir, "schema.json")
            
            # Check file was created
            schema_path = Path(tmpdir) / "schema.json"
            assert schema_path.exists()
            
            # Read and verify contents
            import json
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            
            assert "features" in schema
            assert "label" in schema
            assert schema["features"]["type"] == "ndarray"
            assert "(3,)" in schema["features"]["shape"]
    
    def test_write_parser_dict_with_string(self) -> None:
        """Test writing parser dictionary with string field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_list = [
                Datum(
                    value="test_string",
                    name="text",
                    serialize_fn=serialize_string,
                    decompress_fn=identity
                ),
            ]
            
            write_parser_dict(data_list, tmpdir, "schema.json")
            
            import json
            schema_path = Path(tmpdir) / "schema.json"
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            
            assert "text" in schema
            assert schema["text"]["type"] == "str"
    
    def test_write_parser_dict_multidimensional_array(self) -> None:
        """Test writing parser dictionary with multidimensional arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_list = [
                Datum(
                    value=np.array([[1, 2], [3, 4]], dtype=np.float32),
                    name="matrix",
                    serialize_fn=serialize_float_array,
                    decompress_fn=identity
                ),
            ]
            
            write_parser_dict(data_list, tmpdir, "schema.json")
            
            import json
            schema_path = Path(tmpdir) / "schema.json"
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            
            assert "matrix" in schema
            assert "(2, 2)" in schema["matrix"]["shape"]
    
    def test_write_parser_dict_with_decompress_fn(self) -> None:
        """Test that parser dict respects decompress_fn."""
        with tempfile.TemporaryDirectory() as tmpdir:
            def reshape_value(d: Datum) -> Datum:
                """Reshape array before serialization."""
                return Datum(
                    value=d.value.reshape(2, 2),
                    name=d.name,
                    serialize_fn=d.serialize_fn,
                    decompress_fn=d.decompress_fn
                )
            
            data_list = [
                Datum(
                    value=np.array([1, 2, 3, 4], dtype=np.float32),
                    name="reshaped",
                    serialize_fn=serialize_float_array,
                    decompress_fn=reshape_value
                ),
            ]
            
            write_parser_dict(data_list, tmpdir, "schema.json")
            
            import json
            schema_path = Path(tmpdir) / "schema.json"
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            
            # Should reflect the decompressed shape
            assert "(2, 2)" in schema["reshaped"]["shape"]
    
    def test_write_parser_dict_scalar_values(self) -> None:
        """Test writing parser dictionary with scalar values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_list = [
                Datum(
                    value=42,
                    name="int_value",
                    serialize_fn=serialize_float_or_int,
                    decompress_fn=identity
                ),
                Datum(
                    value=3.14,
                    name="float_value",
                    serialize_fn=serialize_float_or_int,
                    decompress_fn=identity
                ),
            ]
            
            write_parser_dict(data_list, tmpdir, "schema.json")
            
            import json
            schema_path = Path(tmpdir) / "schema.json"
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            
            assert "int_value" in schema
            assert "float_value" in schema
            assert schema["int_value"]["type"] == "int"
            assert schema["float_value"]["type"] == "float"

