"""Wavelet transforms for JAX (non-differentiable preprocessing).

Pure JAX implementation adapted from Jax-Wavelet-Toolbox:
https://github.com/v0lta/Jax-Wavelet-Toolbox

All transforms apply stop_gradient internally to prevent interference
with gradient-based training. Use these for fixed preprocessing, not
learnable wavelet representations.

Supported wavelets:
- 'haar': Haar wavelet (shortest, fastest)
- 'db2': Daubechies 2
- 'db4': Daubechies 4

Example:
    >>> import jax.numpy as jnp
    >>> from beagle.wavelets import wavedec2
    >>> 
    >>> images = jnp.ones((4, 256, 256, 1))
    >>> coeffs = wavedec2(images, wavelet='haar')
    >>> coeffs.shape
    (4, 128, 128, 4)
"""

from beagle.wavelets.conv import wavedec2_single as wavedec2, waverec2_single as waverec2
from beagle.wavelets.filters import get_filters, WAVELETS

__all__ = [
    'wavedec2',
    'waverec2',
    'get_filters',
    'WAVELETS',
]

