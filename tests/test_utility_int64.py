from __future__ import annotations
import tensorflow as tf
from beagle.dataset.utility import _int64_feature


def test_int64_feature() -> None:
    """Test _int64_feature helper function."""
    feature = _int64_feature(42)
    
    assert isinstance(feature, tf.train.Feature)
    assert len(feature.int64_list.value) == 1
    assert feature.int64_list.value[0] == 42


def test_int64_feature_negative() -> None:
    """Test _int64_feature with negative numbers."""
    feature = _int64_feature(-10)
    
    assert feature.int64_list.value[0] == -10


def test_int64_feature_zero() -> None:
    """Test _int64_feature with zero."""
    feature = _int64_feature(0)
    
    assert feature.int64_list.value[0] == 0

