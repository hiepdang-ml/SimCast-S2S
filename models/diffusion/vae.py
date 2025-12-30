from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common import NamedModel


class _HasNamedModules:

    @staticmethod
    def get_downscalingblock_name(index: int) -> str:
        return f"down_sampling_block_{index}"

    @staticmethod
    def get_convstack_name(index: int) -> str:
        return f"conv_stack_{index}"

    @staticmethod
    def get_upscalingblock_name(index: int) -> str:
        return f"up_samping_block_{index}"


class _Freezable:

    @property
    def is_frozen(self) -> bool:
        return not any(param.requires_grad for param in self.parameters())

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        print(f"{self.name} has been frozen")

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
        print(f"{self.name} has been unfrozen")


class _ConvStack(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.hidden_dim: int = hidden_dim
        self.n_layers: int = n_layers

        assert n_layers >= 2
        layers: list[nn.Module] = []
        for i in range(n_layers):
            # First layer
            if i == 0:
                _in_channels: int = in_channels
                _out_channels: int = hidden_dim
                Activation: Type[nn.Module] = nn.GELU
            # Last layer
            elif i == n_layers - 1:
                _in_channels: int = hidden_dim
                _out_channels: int = out_channels
                Activation: Type[nn.Module] = nn.Identity
            # Intermediate layers
            else:
                _in_channels: int = hidden_dim
                _out_channels: int = hidden_dim
                Activation: Type[nn.Module] = nn.GELU

            layers.extend([
                nn.Conv2d(
                    in_channels=_in_channels, out_channels=_out_channels,
                    kernel_size=3, stride=1, padding=1,
                ),
                Activation(),
            ])
            del _in_channels, _out_channels

        self.block = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.block(input)


class _DownscalingBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, 
                kernel_size=4, stride=2, padding=1,
            ),
            nn.GELU(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.block(input)


class _UpscalingBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
    
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=out_channels, 
                kernel_size=4, stride=2, padding=1
            ),
            nn.GELU(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.block(input)


class VAEEncoder(_Freezable, _HasNamedModules, NamedModel, nn.Module):

    def __init__(
        self, 
        n_days: int, n_features: int, latent_dim: int, hidden_dim: int,
        n_downscaling_blocks: int, n_convstack_layers: int,
        n_convhead_layers: int,
    ) -> None:
        super().__init__()
        self.n_days: int = n_days
        self.n_features: int = n_features
        self.pixel_dim: int = n_days * n_features
        self.latent_dim: int = latent_dim
        self.hidden_dim: int = hidden_dim
        self.n_downscaling_blocks: int = n_downscaling_blocks
        self.n_convstack_layers: int = n_convstack_layers
        self.n_convhead_layers: int = n_convhead_layers
        div: int = 2 ** n_downscaling_blocks
        if 192 % div != 0 or 288 % div != 0:
            raise ValueError(f"Invalid n_layers: 192 or 288 not divisible by 2^{n_downscaling_blocks}")

        self.expected_H: int = 192 // div
        self.expected_W: int = 288 // div
        self.preprocessing = _ConvStack(
            in_channels=self.pixel_dim, out_channels=self.pixel_dim, hidden_dim=self.pixel_dim * 4, n_layers=2,
        )

        # Downsampling blocks: each block should shrink the pixel space four times
        assert n_downscaling_blocks >= 2
        for i in range(n_downscaling_blocks):
            # First layer
            if i == 0:
                _in_channels: int = self.pixel_dim
                _out_channels: int = hidden_dim
            # From 2nd layer
            else:
                _in_channels: int = hidden_dim
                _out_channels: int = hidden_dim

            setattr(
                self, 
                VAEEncoder.get_downscalingblock_name(index=i),
                _DownscalingBlock(in_channels=_in_channels, out_channels=_in_channels)
            )
            setattr(
                self, 
                VAEEncoder.get_convstack_name(index=i),
                _ConvStack(
                    in_channels=_in_channels, out_channels=_out_channels, 
                    hidden_dim=_out_channels, n_layers=n_convstack_layers,
                ),
            )
            del _in_channels, _out_channels

        # Head
        assert n_convhead_layers >= 2
        mu_layers: list[nn.Module] = []
        logvar_layers: list[nn.Module] = []

        for i in range(n_convhead_layers):
            # Last layer
            if i == n_convhead_layers - 1:
                _in_channels: int = hidden_dim
                _out_channels: int = latent_dim
                Activation: Type[nn.Module] = nn.Identity
            # Upto second-last layer
            else:
                _in_channels: int = hidden_dim
                _out_channels: int = hidden_dim
                Activation: Type[nn.Module] = nn.GELU

            mu_layers.extend([
                nn.Conv2d(
                    in_channels=_in_channels, out_channels=_out_channels,
                    kernel_size=1, stride=1, padding=0,
                ),
                Activation(),
            ])
            logvar_layers.extend([
                nn.Conv2d(
                    in_channels=_in_channels, out_channels=_out_channels,
                    kernel_size=1, stride=1, padding=0,
                ),
                Activation(),
            ])
            del _in_channels, _out_channels

        self.mu_head = nn.Sequential(*mu_layers)
        self.logvar_head = nn.Sequential(*logvar_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size: int = x.shape[0]
        assert x.shape == (batch_size, 1, 192, 288, self.n_features)
        h: torch.Tensor = x.permute(0, 1, 4, 2, 3).flatten(start_dim=1, end_dim=2)
        h = self.preprocessing(h) + h
        for i in range(self.n_downscaling_blocks):
            h = getattr(self, VAEEncoder.get_downscalingblock_name(index=i))(h)
            # first layer
            if i == 0:
                residual: torch.Tensor = torch.zeros(size=[1], dtype=h.dtype, device=h.device)
            # from 2nd layer
            else:
                residual: torch.Tensor = h
            h = getattr(self, VAEEncoder.get_convstack_name(index=i))(h) + residual
            del residual

        assert h.shape == (batch_size, self.hidden_dim, self.expected_H, self.expected_W)
        # h = h.contiguous()
        mu: torch.Tensor = self.mu_head(h)
        logvar: torch.Tensor = self.logvar_head(h)
        assert mu.shape == logvar.shape == (batch_size, self.latent_dim, self.expected_H, self.expected_W)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, scale: float = 1.) -> torch.Tensor:
        assert mu.shape == logvar.shape
        std: torch.Tensor = torch.exp(0.5 * logvar)
        eps: torch.Tensor = torch.randn_like(std)
        return mu + scale * eps * std


class VAEDecoder(_Freezable, _HasNamedModules, NamedModel, nn.Module):
    
    def __init__(
        self, 
        n_days: int, n_features: int, latent_dim: int, hidden_dim: int,
        n_upscaling_blocks: int, n_convstack_layers: int,
        n_convhead_layers: int,
    ) -> None:
        super().__init__()
        self.n_days: int = n_days
        self.n_features: int = n_features
        self.pixel_dim: int = n_days * n_features
        self.latent_dim: int = latent_dim
        self.hidden_dim: int = hidden_dim
        self.n_upscaling_blocks: int = n_upscaling_blocks
        self.n_convstack_layers: int =  n_convstack_layers
        self.n_convhead_layers: int = n_convhead_layers

        self.preprocessing = _ConvStack(
            in_channels=latent_dim, out_channels=latent_dim, hidden_dim=latent_dim * 4, n_layers=2,
        )
        # Upsampling blocks: each block should expand the latent space four times
        assert n_upscaling_blocks >= 2
        for i in range(n_upscaling_blocks):
            # First layer
            if i == 0:
                _in_channels: int = latent_dim
                _out_channels: int = hidden_dim
            # From 2nd layer
            else:
                _in_channels: int = hidden_dim
                _out_channels: int = hidden_dim

            setattr(
                self, VAEDecoder.get_upscalingblock_name(index=i), 
                _UpscalingBlock(in_channels=_in_channels, out_channels=_in_channels)
            )
            setattr(
                self, VAEDecoder.get_convstack_name(index=i),
                _ConvStack(
                    in_channels=_in_channels, out_channels=_out_channels, 
                    hidden_dim=_out_channels, n_layers=n_convstack_layers,
                )
            )
            del _in_channels, _out_channels

        # Head
        layers: list[nn.Module] = []
        assert n_convhead_layers >= 2
        for i in range(n_convhead_layers):
            # Last layer
            if i == n_convhead_layers - 1:
                _in_channels: int = hidden_dim
                _out_channels: int = self.pixel_dim
                Activation: Type[nn.Module] = nn.Identity
            # Upto second-last layer
            else:
                _in_channels: int = hidden_dim
                _out_channels: int = hidden_dim
                Activation: Type[nn.Module] = nn.GELU
                
            layers.extend([
                nn.Conv2d(
                    in_channels=_in_channels, out_channels=_out_channels,
                    kernel_size=3, stride=1, padding=1,
                ),
                Activation(),
            ])
            del _in_channels, _out_channels

        self.head = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size: int = z.shape[0]
        assert z.shape == (batch_size, self.latent_dim, z.shape[2], z.shape[3])
        x: torch.Tensor = self.preprocessing(z) + z
        for i in range(self.n_upscaling_blocks):
            x = getattr(self, VAEDecoder.get_upscalingblock_name(index=i))(x)
            # first layer
            if i == 0:
                residual: torch.Tensor = torch.zeros(size=[1], dtype=x.dtype, device=x.device)
            # from 2nd layer
            else:
                residual: torch.Tensor = x
            x = getattr(self, VAEDecoder.get_convstack_name(index=i))(x) + residual
            del residual

        assert x.shape == (batch_size, self.hidden_dim, 192, 288)
        x = self.head(x)
        assert x.shape == (batch_size, self.pixel_dim, 192, 288)
        x = x.reshape(batch_size, self.n_days, self.n_features, 192, 288).permute(0, 1, 3, 4, 2)
        assert x.shape == (batch_size, self.n_days, 192, 288, self.n_features)
        return x


class VAE(_Freezable, NamedModel, nn.Module):

    def __init__(
        self, 
        n_days: int, n_features: int, latent_dim: int, hidden_dim: int,
        n_scaling_blocks: int, n_convstack_layers: int, n_convhead_layers: int, 
    ) -> None:
        super().__init__()
        self.n_days: int = n_days
        self.n_features: int = n_features
        self.pixel_dim: int = n_days * n_features
        self.latent_dim: int = latent_dim
        self.hidden_dim: int = hidden_dim
        self.n_scaling_blocks: int = n_scaling_blocks
        self.n_convstack_layers: int = n_convstack_layers
        self.n_convhead_layers: int = n_convhead_layers
        self.encoder = VAEEncoder(
            n_days=n_days, n_features=n_features, latent_dim=latent_dim, hidden_dim=hidden_dim,
            n_downscaling_blocks=n_scaling_blocks, n_convstack_layers=n_convstack_layers,
            n_convhead_layers=n_convhead_layers,
        )
        self.decoder = VAEDecoder(
            n_days=n_days, n_features=n_features, latent_dim=latent_dim, hidden_dim=hidden_dim,
            n_upscaling_blocks=n_scaling_blocks, n_convstack_layers=n_convstack_layers,
            n_convhead_layers=n_convhead_layers,
        )
        self.apply(VAE._init_weights)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size: int = x.shape[0]
        mu: torch.Tensor; logvar: torch.Tensor
        mu, logvar = self.encoder(x)
        z: torch.Tensor = VAEEncoder.reparameterize(mu=mu, logvar=logvar)
        reconstructed_x: torch.Tensor = self.decoder(z)
        assert reconstructed_x.shape == x.shape
        return reconstructed_x, mu, logvar

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            with torch.no_grad():
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def out_features(self) -> int:
        return self.pixel_dim


class VAE_Wind(VAE):
    pass

class VAE_Mass(VAE):
    pass

class VAE_Thermal(VAE):
    pass

class VAE_Hydro(VAE):
    pass

class VAE_Precip(VAE):
    pass

class VAE_Target(VAE):
    pass



