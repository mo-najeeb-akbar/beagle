from __future__ import annotations
import pytest
import numpy as np
from hypothesis import given, strategies as st
from beagle.dataset.types import Datum, identity, get_function_name


class TestIdentity:
    """Test the identity function (pure function)."""
    
    def test_identity_returns_same_value(self) -> None:
        assert identity(5) == 5
        assert identity("test") == "test"
        assert identity([1, 2, 3]) == [1, 2, 3]
    
    @given(st.integers())
    def test_identity_property_integers(self, x: int) -> None:
        """Property: identity(x) == x for all integers."""
        assert identity(x) == x
    
    @given(st.text(min_size=0, max_size=1000))
    def test_identity_property_strings(self, x: str) -> None:
        """Property: identity(x) == x for all strings."""
        assert identity(x) == x


class TestGetFunctionName:
    """Test the get_function_name utility (pure function)."""
    
    def test_get_function_name_with_function(self) -> None:
        def sample_func() -> None:
            pass
        
        assert get_function_name(sample_func) == "sample_func"
    
    def test_get_function_name_with_lambda(self) -> None:
        result = get_function_name(lambda x: x)
        assert "<lambda>" in result
    
    def test_get_function_name_with_none(self) -> None:
        assert get_function_name(None) == ""
    
    def test_get_function_name_with_builtin(self) -> None:
        assert get_function_name(len) == "len"


class TestDatum:
    """Test the Datum dataclass."""
    
    def test_datum_creation_with_numpy_array(self) -> None:
        """Test creating Datum with numpy array."""
        arr = np.array([1.0, 2.0, 3.0])
        serialize_fn = lambda d: None
        decompress_fn = lambda d: d
        
        datum = Datum(
            value=arr,
            name="test_array",
            serialize_fn=serialize_fn,
            decompress_fn=decompress_fn
        )
        
        assert np.array_equal(datum.value, arr)
        assert datum.name == "test_array"
        assert datum.serialize_fn == serialize_fn
        assert datum.decompress_fn == decompress_fn
    
    def test_datum_creation_with_int(self) -> None:
        """Test creating Datum with integer."""
        serialize_fn = lambda d: None
        decompress_fn = lambda d: d
        
        datum = Datum(
            value=42,
            name="test_int",
            serialize_fn=serialize_fn,
            decompress_fn=decompress_fn
        )
        
        assert datum.value == 42
        assert datum.name == "test_int"
    
    def test_datum_creation_with_float(self) -> None:
        """Test creating Datum with float."""
        serialize_fn = lambda d: None
        decompress_fn = lambda d: d
        
        datum = Datum(
            value=3.14,
            name="test_float",
            serialize_fn=serialize_fn,
            decompress_fn=decompress_fn
        )
        
        assert datum.value == 3.14
        assert datum.name == "test_float"
    
    def test_datum_creation_with_string(self) -> None:
        """Test creating Datum with string."""
        serialize_fn = lambda d: None
        decompress_fn = lambda d: d
        
        datum = Datum(
            value="hello",
            name="test_str",
            serialize_fn=serialize_fn,
            decompress_fn=decompress_fn
        )
        
        assert datum.value == "hello"
        assert datum.name == "test_str"
    
    def test_datum_creation_with_tuple(self) -> None:
        """Test creating Datum with tuple."""
        serialize_fn = lambda d: None
        decompress_fn = lambda d: d
        
        datum = Datum(
            value=(1, 2, 3),
            name="test_tuple",
            serialize_fn=serialize_fn,
            decompress_fn=decompress_fn
        )
        
        assert datum.value == (1, 2, 3)
        assert datum.name == "test_tuple"
    
    def test_datum_invalid_type_raises_error(self) -> None:
        """Test that invalid types raise TypeError."""
        serialize_fn = lambda d: None
        decompress_fn = lambda d: d
        
        with pytest.raises(TypeError, match="Value must be an instance of"):
            Datum(
                value=[1, 2, 3],  # list is not allowed
                name="test_list",
                serialize_fn=serialize_fn,
                decompress_fn=decompress_fn
            )
    
    def test_datum_is_frozen(self) -> None:
        """Test that Datum is immutable (frozen dataclass)."""
        serialize_fn = lambda d: None
        decompress_fn = lambda d: d
        
        datum = Datum(
            value=42,
            name="test",
            serialize_fn=serialize_fn,
            decompress_fn=decompress_fn
        )
        
        with pytest.raises(Exception):  # FrozenInstanceError
            datum.value = 100  # type: ignore
    
    def test_datum_str_representation(self) -> None:
        """Test string representation of Datum."""
        def my_serialize(d: Datum) -> None:
            pass
        
        def my_decompress(d: Datum) -> Datum:
            return d
        
        datum = Datum(
            value=42,
            name="test",
            serialize_fn=my_serialize,
            decompress_fn=my_decompress
        )
        
        str_repr = str(datum)
        assert "test" in str_repr
        assert "42" in str_repr
        assert "my_serialize" in str_repr
        assert "my_decompress" in str_repr
    
    def test_datum_all_fields_required(self) -> None:
        """Test that all fields are required (no defaults)."""
        serialize_fn = lambda d: None
        decompress_fn = lambda d: d
        
        # Should fail without any field
        with pytest.raises(TypeError):
            Datum()  # type: ignore
        
        # Should fail without serialize_fn
        with pytest.raises(TypeError):
            Datum(  # type: ignore
                value=42,
                name="test",
                decompress_fn=decompress_fn
            )
        
        # Should fail without decompress_fn
        with pytest.raises(TypeError):
            Datum(  # type: ignore
                value=42,
                name="test",
                serialize_fn=serialize_fn
            )

