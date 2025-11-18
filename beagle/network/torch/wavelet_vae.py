from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, filters: int) -> None:
        super().__init__()
        self.filters = filters
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=filters, eps=1e-5)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=filters, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = F.silu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        return F.silu(out + skip)


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, features: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.features = features

        self.haar_conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        self.gn_haar = nn.GroupNorm(num_groups=4, num_channels=4, eps=1e-5)

        self.conv_layers = nn.ModuleList()
        self.gn_layers = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()

        for i in range(5):
            if i == 0:
                in_channels = 4
            else:
                in_channels = features
            self.conv_layers.append(
                nn.Conv2d(in_channels, features, 3, stride=2, padding=1, bias=False)
            )
            self.gn_layers.append(nn.GroupNorm(num_groups=8, num_channels=features, eps=1e-5))
            self.residual_blocks.append(ResidualBlock(features))

        self.dense_mu = nn.Conv2d(features, latent_dim, 1)
        self.dense_logvar = nn.Conv2d(features, latent_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.haar_conv(x)
        x = self.gn_haar(x)
        for i in range(5):
            x = self.conv_layers[i](x)
            x = self.gn_layers[i](x)
            x = F.silu(x)
            x = self.residual_blocks[i](x)

        mu = self.dense_mu(x)
        log_var = self.dense_logvar(x)

        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, bottle_neck: int, features: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.bottle_neck = bottle_neck
        self.features = features

        self.conv_layers = nn.ModuleList()
        self.gn_layers = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()

        for i in range(5):
            if i == 0:
                in_channels = latent_dim
            else:
                in_channels = features
            self.conv_layers.append(
                nn.Conv2d(in_channels, features, 3, padding=1, bias=False)
            )
            self.gn_layers.append(nn.GroupNorm(num_groups=8, num_channels=features, eps=1e-5))
            self.residual_blocks.append(ResidualBlock(features))

        self.out_conv = nn.Conv2d(features, 4, 3, padding=1)
        self.haar_conv_transpose = nn.ConvTranspose2d(
            4, 1, kernel_size=2, stride=2, padding=0, bias=False
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for i in range(5):
            batch, channels, height, width = x.shape
            x = F.interpolate(
                x, size=(height * 2, width * 2), mode="bilinear", align_corners=False
            )
            x = self.conv_layers[i](x)
            x = self.gn_layers[i](x)
            x = F.silu(x)
            x = self.residual_blocks[i](x)

        x_haar = self.out_conv(x)
        x_recon = self.haar_conv_transpose(x_haar)

        return x_recon, x_haar


class VAEPyTorch(nn.Module):
    def __init__(
        self, latent_dim: int = 128, base_features: int = 32, block_size: int = 8
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.base_features = base_features
        self.block_size = block_size

        self.encoder = Encoder(latent_dim, base_features)
        self.decoder = Decoder(latent_dim, block_size, base_features)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return mu

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon, x_haar = self.decode(z)
        return x_recon, x_haar, mu, log_var
