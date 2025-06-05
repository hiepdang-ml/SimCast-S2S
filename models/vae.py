from typing import *
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):

    def __init__(self, pixel_dim: int, latent_dim: int, n_layers: int) -> None:
        super().__init__()
        self.pixel_dim: int = pixel_dim
        self.latent_dim: int = latent_dim
        self.n_layers: int = n_layers
        div: int = 2 ** n_layers
        if 192 % div != 0 or 288 % div != 0:
            raise ValueError(f"Invalid n_layers: 192 or 288 not divisible by 2^{n_layers}")

        self.expected_H: int = 192 // div
        self.expected_W: int = 288 // div
        # Each layer should shrink the pixel space by half
        kernel_size: int = 4
        stride: int = 2
        padding: int = 1
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels=pixel_dim, out_channels=latent_dim, 
                kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.ReLU(),
        ]
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Conv2d(
                    in_channels=latent_dim, out_channels=latent_dim, 
                    kernel_size=kernel_size, stride=stride, padding=padding
                ),
                nn.ReLU(),
            ])

        self.encoder = nn.Sequential(*layers)
        self.mu_head = nn.Sequential(
            nn.Conv2d(
                in_channels=latent_dim, out_channels=latent_dim, 
                kernel_size=1, stride=1, padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=latent_dim, out_channels=latent_dim, 
                kernel_size=1, stride=1, padding=0,
            ),
        )
        self.logvar_head = nn.Sequential(
            nn.Conv2d(
                in_channels=latent_dim, out_channels=latent_dim, 
                kernel_size=1, stride=1, padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=latent_dim, out_channels=latent_dim, 
                kernel_size=1, stride=1, padding=0,
            ),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size: int = x.shape[0]
        assert x.shape == (batch_size, 192, 288, self.pixel_dim)
        x: torch.Tensor = x.permute(0, 3, 1, 2)
        assert x.shape == (batch_size, self.pixel_dim, 192, 288)
        h: torch.Tensor = self.encoder(x)
        assert h.shape == (batch_size, self.latent_dim, self.expected_H, self.expected_W)
        mu: torch.Tensor = self.mu_head(h)
        logvar: torch.Tensor = self.logvar_head(h)
        assert mu.shape == logvar.shape == (batch_size, self.latent_dim, self.expected_H, self.expected_W)
        return mu, logvar


class VAEDecoder(nn.Module):
    
    def __init__(self, pixel_dim: int, latent_dim: int, n_layers: int) -> None:
        super().__init__()
        self.pixel_dim: int = pixel_dim
        self.latent_dim: int = latent_dim
        self.n_layers: int = n_layers

        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels=latent_dim, out_channels=latent_dim, 
                kernel_size=1, stride=1, padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=latent_dim, out_channels=latent_dim, 
                kernel_size=1, stride=1, padding=0,
            ),
            nn.ReLU(),
        ]
        # Each layer should double the pixel space
        kernel_size: int = 4
        stride: int = 2
        padding: int = 1
        for _ in range(n_layers):
            layers.extend([
                nn.ConvTranspose2d(
                    in_channels=latent_dim, out_channels=latent_dim, 
                    kernel_size=kernel_size, stride=stride, padding=padding
                ),
                nn.ReLU(),
            ])

        layers.extend([
            nn.Conv2d(
                in_channels=latent_dim, out_channels=pixel_dim, 
                kernel_size=1, stride=1, padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=pixel_dim, out_channels=pixel_dim, 
                kernel_size=1, stride=1, padding=0,
            ),
        ])
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size: int = z.shape[0]
        assert z.shape == (batch_size, self.latent_dim, z.shape[2], z.shape[3])
        x: torch.Tensor = self.decoder(z)
        assert x.shape == (batch_size, self.pixel_dim, 192, 288)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape((batch_size, 192, 288, self.pixel_dim))
        return x


class VAE(nn.Module):

    def __init__(self, pixel_dim: int, latent_dim: int, n_layers: int) -> None:
        super().__init__()
        self.pixel_dim: int = pixel_dim
        self.latent_dim: int = latent_dim
        self.n_layers: int = n_layers
        self.encoder = VAEEncoder(pixel_dim=pixel_dim, latent_dim=latent_dim, n_layers=n_layers)
        self.decoder = VAEDecoder(pixel_dim=pixel_dim, latent_dim=latent_dim, n_layers=n_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size: int = x.shape[0]
        assert x.shape == (batch_size, 192, 288, self.pixel_dim)
        mu: torch.Tensor; logvar: torch.Tensor
        mu, logvar = self.encoder(x)
        z: torch.Tensor = VAE._reparameterize(mu=mu, logvar=logvar)
        reconstructed_x: torch.Tensor = self.decoder(z)
        assert reconstructed_x.shape == x.shape
        return reconstructed_x, mu, logvar

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        assert mu.shape == logvar.shape
        std: torch.Tensor = torch.exp(0.5 * logvar)
        eps: torch.Tensor = torch.randn_like(std)
        return mu + eps * std



if __name__ == "__main__":
    pixel_dim: int = 28 * 14
    latent_dim: int = 64
    n_layers: int = 3
    vae = VAE(pixel_dim=pixel_dim, latent_dim=latent_dim, n_layers=n_layers)

    x: torch.Tensor = torch.randn(size=(32, 192, 288, pixel_dim))
    x_hat, mu, logvar = vae(x)


