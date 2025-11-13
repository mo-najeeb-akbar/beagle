"""Configure matplotlib backend for headless/non-headless operation."""

from __future__ import annotations

import os


def configure_matplotlib_backend() -> str:
    """Configure matplotlib backend based on environment.
    
    Checks BEAGLE_HEADLESS environment variable:
    - "true", "1", "yes" (case-insensitive) -> headless mode (Agg backend)
    - anything else or unset -> non-headless mode (default backend)
    
    Also respects MPLBACKEND if set (takes precedence).
    
    Returns:
        Backend name that was configured
        
    Example:
        >>> # In Docker: BEAGLE_HEADLESS=true make shell
        >>> configure_matplotlib_backend()  # Returns 'Agg'
        
        >>> # In Docker: BEAGLE_HEADLESS=false make shell  
        >>> configure_matplotlib_backend()  # Returns default backend
    """
    # MPLBACKEND takes precedence (standard matplotlib env var)
    if 'MPLBACKEND' in os.environ:
        backend = os.environ['MPLBACKEND']
        return backend
    
    # Check BEAGLE_HEADLESS
    headless_str = os.environ.get('BEAGLE_HEADLESS', '').lower()
    is_headless = headless_str in ('true', '1', 'yes')
    
    if is_headless:
        # Set before importing matplotlib.pyplot
        os.environ['MPLBACKEND'] = 'Agg'
        return 'Agg'
    
    # Let matplotlib choose default (usually TkAgg or Qt5Agg)
    return 'default'


def is_headless() -> bool:
    """Check if running in headless mode.
    
    Returns:
        True if headless mode is enabled
    """
    headless_str = os.environ.get('BEAGLE_HEADLESS', '').lower()
    is_headless_flag = headless_str in ('true', '1', 'yes')
    
    # Also check if MPLBACKEND is set to Agg
    backend = os.environ.get('MPLBACKEND', '').lower()
    is_agg = backend == 'agg'
    
    return is_headless_flag or is_agg

