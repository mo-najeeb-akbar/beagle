"""Tests for visualization backend configuration."""

from __future__ import annotations

import os
import pytest


def test_configure_backend_respects_mplbackend():
    """Test that MPLBACKEND env var takes precedence."""
    from beagle.visualization.backend import configure_matplotlib_backend
    
    # Save original
    original_mplbackend = os.environ.get('MPLBACKEND')
    original_beagle = os.environ.get('BEAGLE_HEADLESS')
    
    try:
        # Set both env vars
        os.environ['MPLBACKEND'] = 'PDF'
        os.environ['BEAGLE_HEADLESS'] = 'true'
        
        backend = configure_matplotlib_backend()
        assert backend == 'PDF', "MPLBACKEND should take precedence"
        
    finally:
        # Restore
        if original_mplbackend:
            os.environ['MPLBACKEND'] = original_mplbackend
        elif 'MPLBACKEND' in os.environ:
            del os.environ['MPLBACKEND']
            
        if original_beagle:
            os.environ['BEAGLE_HEADLESS'] = original_beagle
        elif 'BEAGLE_HEADLESS' in os.environ:
            del os.environ['BEAGLE_HEADLESS']


def test_configure_backend_headless_mode():
    """Test that BEAGLE_HEADLESS=true sets Agg backend."""
    from beagle.visualization.backend import configure_matplotlib_backend
    
    # Save original
    original_mplbackend = os.environ.get('MPLBACKEND')
    original_beagle = os.environ.get('BEAGLE_HEADLESS')
    
    try:
        # Clear MPLBACKEND so BEAGLE_HEADLESS takes effect
        if 'MPLBACKEND' in os.environ:
            del os.environ['MPLBACKEND']
        
        # Test various truthy values
        for value in ['true', 'True', 'TRUE', '1', 'yes', 'YES']:
            os.environ['BEAGLE_HEADLESS'] = value
            backend = configure_matplotlib_backend()
            assert backend == 'Agg', f"BEAGLE_HEADLESS={value} should set Agg backend"
            # Clean up for next iteration
            if 'MPLBACKEND' in os.environ:
                del os.environ['MPLBACKEND']
    
    finally:
        # Restore
        if original_mplbackend:
            os.environ['MPLBACKEND'] = original_mplbackend
        elif 'MPLBACKEND' in os.environ:
            del os.environ['MPLBACKEND']
            
        if original_beagle:
            os.environ['BEAGLE_HEADLESS'] = original_beagle
        elif 'BEAGLE_HEADLESS' in os.environ:
            del os.environ['BEAGLE_HEADLESS']


def test_configure_backend_non_headless_mode():
    """Test that BEAGLE_HEADLESS=false uses default backend."""
    from beagle.visualization.backend import configure_matplotlib_backend
    
    # Save original
    original_mplbackend = os.environ.get('MPLBACKEND')
    original_beagle = os.environ.get('BEAGLE_HEADLESS')
    
    try:
        # Clear MPLBACKEND
        if 'MPLBACKEND' in os.environ:
            del os.environ['MPLBACKEND']
        
        # Test falsy values
        for value in ['false', 'False', '0', 'no', 'off', '']:
            os.environ['BEAGLE_HEADLESS'] = value
            backend = configure_matplotlib_backend()
            assert backend == 'default', f"BEAGLE_HEADLESS={value} should use default"
            # Clean up for next iteration
            if 'MPLBACKEND' in os.environ:
                del os.environ['MPLBACKEND']
    
    finally:
        # Restore
        if original_mplbackend:
            os.environ['MPLBACKEND'] = original_mplbackend
        elif 'MPLBACKEND' in os.environ:
            del os.environ['MPLBACKEND']
            
        if original_beagle:
            os.environ['BEAGLE_HEADLESS'] = original_beagle
        elif 'BEAGLE_HEADLESS' in os.environ:
            del os.environ['BEAGLE_HEADLESS']


def test_is_headless_detects_mode():
    """Test is_headless() utility function."""
    from beagle.visualization.backend import is_headless
    
    # Save original
    original_mplbackend = os.environ.get('MPLBACKEND')
    original_beagle = os.environ.get('BEAGLE_HEADLESS')
    
    try:
        # Test headless via BEAGLE_HEADLESS
        if 'MPLBACKEND' in os.environ:
            del os.environ['MPLBACKEND']
        os.environ['BEAGLE_HEADLESS'] = 'true'
        assert is_headless() is True
        
        # Test headless via MPLBACKEND
        os.environ['BEAGLE_HEADLESS'] = 'false'
        os.environ['MPLBACKEND'] = 'Agg'
        assert is_headless() is True
        
        # Test non-headless
        del os.environ['MPLBACKEND']
        os.environ['BEAGLE_HEADLESS'] = 'false'
        assert is_headless() is False
        
    finally:
        # Restore
        if original_mplbackend:
            os.environ['MPLBACKEND'] = original_mplbackend
        elif 'MPLBACKEND' in os.environ:
            del os.environ['MPLBACKEND']
            
        if original_beagle:
            os.environ['BEAGLE_HEADLESS'] = original_beagle
        elif 'BEAGLE_HEADLESS' in os.environ:
            del os.environ['BEAGLE_HEADLESS']

