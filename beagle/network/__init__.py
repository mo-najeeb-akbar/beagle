from beagle.network.attention import Attention, Block, LayerScale, MLP
from beagle.network.categorical_vae import CategoricalVAE
from beagle.network.compact_vae import CompactVAE
from beagle.network.dense_encoder import DenseEncoder
from beagle.network.hrnet import EmbedNet, HRNetBB, MoNet
from beagle.network.patch import PatchEmbed
from beagle.network.receptive_field import (
    LayerConfig,
    ReceptiveFieldInfo,
    compute_receptive_field,
    create_layer_from_module,
    print_receptive_field_report,
)

from beagle.network.unet import DenoisingUNet
from beagle.network.vit import MaskedViT

__all__ = [
    "Attention",
    "Block",
    "CategoricalVAE",
    "CompactVAE",
    "compute_receptive_field",
    "create_layer_from_module",
    "DenseEncoder",
    "DenoisingUNet",
    "EmbedNet",
    "HRNetBB",
    "LayerConfig",
    "LayerScale",
    "MaskedViT",
    "MLP",
    "MoNet",
    "PatchEmbed",
    "print_receptive_field_report",
    "ReceptiveFieldInfo",
]

try:
    from beagle.network.wavelet_vae import VAE

    __all__.append("VAE")
except ImportError:
    pass

