"""Simple mask prediction network using HRNet backbone."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax.numpy as jnp
import flax.linen as nn

from beagle.network.hrnet import HRNetBB


class MaskNet(nn.Module):
    """Mask prediction network with HRNet backbone.
    
    Outputs single-channel mask at target resolution.
    """
    
    num_stages: int
    features: int
    target_res: float
    
    def setup(self) -> None:
        self.backbone = HRNetBB(
            num_stages=self.num_stages,
            features=self.features,
            target_res=self.target_res,
        )
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Forward pass through backbone and output head.
        
        Args:
            x: Input images [B, H, W, C]
            train: Training mode
            
        Returns:
            Logits for binary mask [B, H//4, W//4, 1]
        """
        # Backbone features
        features = self.backbone(x, train=train)
        
        # Output head: 1x1 conv to single channel
        mask_logits = nn.Conv(
            features=1,
            kernel_size=(1, 1),
            use_bias=True,
        )(features)
        
        return mask_logits

