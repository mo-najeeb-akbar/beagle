from __future__ import annotations
import os
import pytest


def test_tf_config_imports() -> None:
    """Test that tf_config module can be imported and configures TF."""
    # Save original CUDA_VISIBLE_DEVICES
    original_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    
    try:
        # Import should set CUDA_VISIBLE_DEVICES
        from beagle.dataset import tf_config  # noqa: F401
        
        # Verify environment variable is set
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
        
        # Verify function exists
        assert hasattr(tf_config, 'configure_tf_cpu')
        assert callable(tf_config.configure_tf_cpu)
        
    finally:
        # Restore original value
        if original_cuda is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda


def test_configure_tf_cpu_function() -> None:
    """Test configure_tf_cpu function can be called."""
    from beagle.dataset.tf_config import configure_tf_cpu
    
    # Should not raise
    configure_tf_cpu()
    
    # Verify environment is set
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'


def test_tf_config_imported_via_init() -> None:
    """Test that tf_config is imported when importing beagle.dataset."""
    # Import should trigger tf_config import
    import beagle.dataset  # noqa: F401
    
    # Verify tf_config was imported and configured
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'

