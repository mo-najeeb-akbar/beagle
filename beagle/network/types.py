"""Type definitions for model I/O contracts.

This module defines standard output types and naming conventions for Beagle models.
All models should return dict[str, jnp.ndarray] with semantic, domain-specific names.
"""

from typing import TypedDict
import jax.numpy as jnp


class VAEOutputs(TypedDict, total=False):
    """Standard VAE model outputs.

    Attributes:
        reconstruction: Reconstructed input [B, H, W, C]
        wavelet_coeffs: Wavelet domain coefficients [B, H/2, W/2, 4] (optional)
        mu: Latent distribution mean [B, ...]
        log_var: Latent distribution log variance [B, ...]
        logvar: Latent distribution log variance (alternative naming) [B, ...]
        latent: Sampled or deterministic latent vector [B, ...]
        z: Latent vector (alternative naming) [B, ...]
        latent_normalized: Normalized latent for embeddings [B, ...]
    """
    reconstruction: jnp.ndarray
    wavelet_coeffs: jnp.ndarray
    mu: jnp.ndarray
    log_var: jnp.ndarray
    logvar: jnp.ndarray
    latent: jnp.ndarray
    z: jnp.ndarray
    latent_normalized: jnp.ndarray


class SegmentationOutputs(TypedDict, total=False):
    """Segmentation model outputs.

    Attributes:
        logits: Unnormalized class logits [B, H, W, num_classes]
        mask: Predicted segmentation mask (class indices) [B, H, W]
        features: Backbone feature maps [B, H, W, C]
    """
    logits: jnp.ndarray
    mask: jnp.ndarray
    features: jnp.ndarray


class EncoderOutputs(TypedDict):
    """Encoder model outputs.

    Attributes:
        embedding: Normalized embedding vector [B, D]
        reconstruction_logits: Reconstruction head output [B, N, C]
        targets: Ground truth targets for reconstruction [B, N]
    """
    embedding: jnp.ndarray
    reconstruction_logits: jnp.ndarray
    targets: jnp.ndarray


# Standard output key names and their semantic meanings
STANDARD_OUTPUT_KEYS = {
    # VAE family
    'reconstruction': 'Reconstructed input (images, sequences, etc.)',
    'wavelet_coeffs': 'Wavelet domain coefficients',
    'mu': 'Latent distribution mean',
    'log_var': 'Latent distribution log variance',
    'logvar': 'Latent distribution log variance (alternative)',
    'latent': 'Sampled or deterministic latent vector',
    'z': 'Latent vector (alternative)',
    'latent_normalized': 'Normalized latent for embeddings',

    # Segmentation
    'logits': 'Unnormalized class logits',
    'mask': 'Predicted segmentation mask (class indices)',
    'features': 'Intermediate feature maps from backbone',

    # Encoders
    'embedding': 'Normalized embedding vector',
    'reconstruction_logits': 'Reconstruction logits (categorical)',
    'targets': 'Ground truth targets for reconstruction',

    # Generic
    'output': 'Primary model output (when only one)',
}
