from __future__ import annotations
import pytest
import numpy as np
import tensorflow as tf
from hypothesis import given, strategies as st
from beagle.dataset.utility import (
    split_list,
    serialize_float_array,
    serialize_float_or_int,
    serialize_image,
    serialize_string,
)
from beagle.dataset.types import Datum


class TestSplitList:
    """Test the split_list function (pure function)."""
    
    def test_split_list_equal_parts(self) -> None:
        """Test splitting a list into equal parts."""
        items = [1, 2, 3, 4, 5, 6]
        result = split_list(items, 3)
        
        assert len(result) == 3
        assert result[0] == [1, 2]
        assert result[1] == [3, 4]
        assert result[2] == [5, 6]
    
    def test_split_list_unequal_parts(self) -> None:
        """Test splitting when items don't divide evenly."""
        items = [1, 2, 3, 4, 5]
        result = split_list(items, 3)
        
        assert len(result) == 3
        assert result[0] == [1, 2]  # Gets extra item
        assert result[1] == [3, 4]  # Gets extra item
        assert result[2] == [5]
    
    def test_split_list_single_part(self) -> None:
        """Test splitting into a single part."""
        items = [1, 2, 3]
        result = split_list(items, 1)
        
        assert len(result) == 1
        assert result[0] == [1, 2, 3]
    
    def test_split_list_more_parts_than_items(self) -> None:
        """Test splitting when k > len(list)."""
        items = [1, 2]
        result = split_list(items, 5)
        
        assert len(result) == 5
        assert result[0] == [1]
        assert result[1] == [2]
        assert result[2] == []
        assert result[3] == []
        assert result[4] == []
    
    def test_split_list_empty(self) -> None:
        """Test splitting an empty list."""
        items: list[int] = []
        result = split_list(items, 3)
        
        assert len(result) == 3
        assert all(part == [] for part in result)
    
    def test_split_list_preserves_order(self) -> None:
        """Test that splitting preserves original order."""
        items = list(range(10))
        result = split_list(items, 3)
        
        flattened = [item for part in result for item in part]
        assert flattened == items
    
    def test_split_list_immutability(self) -> None:
        """Test that original list is not modified (pure function)."""
        items = [1, 2, 3, 4, 5]
        original = items.copy()
        split_list(items, 2)
        
        assert items == original
    
    @given(st.lists(st.integers(), min_size=1, max_size=100), st.integers(min_value=1, max_value=10))
    def test_split_list_property_all_items_preserved(self, items: list[int], k: int) -> None:
        """Property: all items should be preserved after splitting."""
        result = split_list(items, k)
        flattened = [item for part in result for item in part]
        assert sorted(flattened) == sorted(items)
    
    @given(st.lists(st.integers()), st.integers(min_value=1, max_value=10))
    def test_split_list_property_correct_number_of_parts(self, items: list[int], k: int) -> None:
        """Property: should always return exactly k parts."""
        result = split_list(items, k)
        assert len(result) == k


class TestSerializeFloatArray:
    """Test serialize_float_array function."""
    
    def test_serialize_float_array_1d(self) -> None:
        """Test serializing a 1D array."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        datum = Datum(
            value=arr,
            name="test",
            serialize_fn=serialize_float_array,
            decompress_fn=lambda d: d
        )
        
        feature = serialize_float_array(datum)
        
        assert isinstance(feature, tf.train.Feature)
        assert len(feature.float_list.value) == 3
        assert list(feature.float_list.value) == [1.0, 2.0, 3.0]
    
    def test_serialize_float_array_2d(self) -> None:
        """Test serializing a 2D array (should flatten)."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        datum = Datum(
            value=arr,
            name="test",
            serialize_fn=serialize_float_array,
            decompress_fn=lambda d: d
        )
        
        feature = serialize_float_array(datum)
        
        assert isinstance(feature, tf.train.Feature)
        assert len(feature.float_list.value) == 4
        assert list(feature.float_list.value) == [1.0, 2.0, 3.0, 4.0]
    
    def test_serialize_float_array_3d(self) -> None:
        """Test serializing a 3D array (should flatten)."""
        arr = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
        datum = Datum(
            value=arr,
            name="test",
            serialize_fn=serialize_float_array,
            decompress_fn=lambda d: d
        )
        
        feature = serialize_float_array(datum)
        
        assert isinstance(feature, tf.train.Feature)
        assert len(feature.float_list.value) == 8


class TestSerializeFloatOrInt:
    """Test serialize_float_or_int function."""
    
    def test_serialize_float(self) -> None:
        """Test serializing a single float."""
        datum = Datum(
            value=3.14,
            name="test",
            serialize_fn=serialize_float_or_int,
            decompress_fn=lambda d: d
        )
        
        feature = serialize_float_or_int(datum)
        
        assert isinstance(feature, tf.train.Feature)
        assert len(feature.float_list.value) == 1
        assert feature.float_list.value[0] == pytest.approx(3.14)
    
    def test_serialize_int(self) -> None:
        """Test serializing a single integer."""
        datum = Datum(
            value=42,
            name="test",
            serialize_fn=serialize_float_or_int,
            decompress_fn=lambda d: d
        )
        
        feature = serialize_float_or_int(datum)
        
        assert isinstance(feature, tf.train.Feature)
        assert len(feature.float_list.value) == 1
        assert feature.float_list.value[0] == 42.0
    
    def test_serialize_negative(self) -> None:
        """Test serializing negative numbers."""
        datum = Datum(
            value=-7.5,
            name="test",
            serialize_fn=serialize_float_or_int,
            decompress_fn=lambda d: d
        )
        
        feature = serialize_float_or_int(datum)
        assert feature.float_list.value[0] == pytest.approx(-7.5)


class TestSerializeImage:
    """Test serialize_image function."""
    
    def test_serialize_image_grayscale(self) -> None:
        """Test serializing a grayscale image."""
        # Create a small grayscale image (8x8)
        img = np.random.randint(0, 256, (8, 8, 1), dtype=np.uint8)
        datum = Datum(
            value=img,
            name="test",
            serialize_fn=serialize_image,
            decompress_fn=lambda d: d
        )
        
        feature = serialize_image(datum)
        
        assert isinstance(feature, tf.train.Feature)
        assert len(feature.bytes_list.value) == 1
        # Verify it's valid PNG data
        assert feature.bytes_list.value[0].startswith(b'\x89PNG')
    
    def test_serialize_image_rgb(self) -> None:
        """Test serializing an RGB image."""
        # Create a small RGB image (8x8x3)
        img = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        datum = Datum(
            value=img,
            name="test",
            serialize_fn=serialize_image,
            decompress_fn=lambda d: d
        )
        
        feature = serialize_image(datum)
        
        assert isinstance(feature, tf.train.Feature)
        assert len(feature.bytes_list.value) == 1
        assert feature.bytes_list.value[0].startswith(b'\x89PNG')
    
    def test_serialize_image_roundtrip(self) -> None:
        """Test that image can be encoded and decoded."""
        img = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        datum = Datum(
            value=img,
            name="test",
            serialize_fn=serialize_image,
            decompress_fn=lambda d: d
        )
        
        feature = serialize_image(datum)
        encoded_bytes = feature.bytes_list.value[0]
        
        # Decode and verify
        decoded = tf.io.decode_png(encoded_bytes).numpy()
        assert np.array_equal(decoded, img)


class TestSerializeString:
    """Test serialize_string function."""
    
    def test_serialize_string_simple(self) -> None:
        """Test serializing a simple string."""
        datum = Datum(
            value="hello",
            name="test",
            serialize_fn=serialize_string,
            decompress_fn=lambda d: d
        )
        
        feature = serialize_string(datum)
        
        assert isinstance(feature, tf.train.Feature)
        assert len(feature.bytes_list.value) == 1
        assert feature.bytes_list.value[0] == b'hello'
    
    def test_serialize_string_empty(self) -> None:
        """Test serializing an empty string."""
        datum = Datum(
            value="",
            name="test",
            serialize_fn=serialize_string,
            decompress_fn=lambda d: d
        )
        
        feature = serialize_string(datum)
        
        assert isinstance(feature, tf.train.Feature)
        assert len(feature.bytes_list.value) == 1
        assert feature.bytes_list.value[0] == b''
    
    def test_serialize_string_unicode(self) -> None:
        """Test serializing Unicode strings."""
        datum = Datum(
            value="hello 世界 🌍",
            name="test",
            serialize_fn=serialize_string,
            decompress_fn=lambda d: d
        )
        
        feature = serialize_string(datum)
        
        assert isinstance(feature, tf.train.Feature)
        decoded = feature.bytes_list.value[0].decode('utf-8')
        assert decoded == "hello 世界 🌍"
    
    def test_serialize_string_special_chars(self) -> None:
        """Test serializing strings with special characters."""
        datum = Datum(
            value="line1\nline2\ttab",
            name="test",
            serialize_fn=serialize_string,
            decompress_fn=lambda d: d
        )
        
        feature = serialize_string(datum)
        decoded = feature.bytes_list.value[0].decode('utf-8')
        assert decoded == "line1\nline2\ttab"
    
    @given(st.text())
    def test_serialize_string_property_roundtrip(self, text: str) -> None:
        """Property: serialized string should decode back to original."""
        datum = Datum(
            value=text,
            name="test",
            serialize_fn=serialize_string,
            decompress_fn=lambda d: d
        )
        
        feature = serialize_string(datum)
        decoded = feature.bytes_list.value[0].decode('utf-8')
        assert decoded == text

