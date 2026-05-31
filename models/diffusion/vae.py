from typing import Type, cast
import torch
import torch.nn as nn

from models.common import NamedModel
from models.adaptation.lora import LoRAConv2d, LoRATConv2d


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

    def is_all_frozen(self) -> bool:
        return not any(param.requires_grad for param in self.parameters())

    def freeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        print(f"{self.name} has been frozen")

    def unfreeze_all(self) -> None:
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
        logvar = torch.clamp(logvar, min=-10., max=10.) # avoid overflow
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

    def _decode_features(self, z: torch.Tensor) -> torch.Tensor:
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
        return x

    def _format_output(self, x: torch.Tensor) -> torch.Tensor:
        batch_size: int = x.shape[0]
        assert x.shape == (batch_size, self.pixel_dim, 192, 288)
        x = x.reshape(batch_size, self.n_days, self.n_features, 192, 288).permute(0, 1, 3, 4, 2)
        assert x.shape == (batch_size, self.n_days, 192, 288, self.n_features)
        return x

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        features = self._decode_features(z)
        return self._format_output(self.head(features))


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
        _batch_size: int = x.shape[0]
        mu: torch.Tensor; logvar: torch.Tensor  # noqa
        mu, logvar = self.encoder(x)
        z: torch.Tensor = VAEEncoder.reparameterize(mu=mu, logvar=logvar)
        reconstructed_x: torch.Tensor = self.decoder(z)
        assert reconstructed_x.shape == x.shape
        return reconstructed_x, mu, logvar

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            with torch.no_grad():
                nn.init.kaiming_normal_(cast(nn.Parameter, module.weight))
                nn.init.zeros_(cast(nn.Parameter, module.bias))

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


class _Finetunable:

    def is_backbone_frozen(self) -> bool:
        for name, param in self.named_parameters():
            is_lora_param: bool = "lora_A" in name or "lora_B" in name
            is_allowed_module: bool = (
                name.startswith(self._ALLOWED_FINETUNE_PREFIXES)
            )
            if param.requires_grad and not (is_lora_param or is_allowed_module):
                return False
        return True

    def freeze_backbone(self) -> None:
        for name, param in self.named_parameters():
            param.requires_grad = (
                "lora_A" in name
                or "lora_B" in name
                or name.startswith(self._ALLOWED_FINETUNE_PREFIXES)
            )
        print(f"{self.name} backbone has been frozen for LoRA fine-tuning")


class VAE_Precip(_Finetunable, VAE):

    _ALLOWED_FINETUNE_PREFIXES: tuple[str, ...] = (
        "decoder.head.",
        "encoder.mu_head.",
        "encoder.logvar_head.",
    )

    def __init__(
        self,
        n_days: int, n_features: int, latent_dim: int, hidden_dim: int,
        n_scaling_blocks: int, n_convstack_layers: int, n_convhead_layers: int,
        is_finetuning: bool = False, lora_rank: int = 0,
    ) -> None:
        self.is_finetuning: bool = is_finetuning
        self.lora_rank: int = lora_rank
        super().__init__(
            n_days=n_days,
            n_features=n_features,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_scaling_blocks=n_scaling_blocks,
            n_convstack_layers=n_convstack_layers,
            n_convhead_layers=n_convhead_layers,
        )
        self.n_lora_conv_layers: int = 0
        self.n_lora_tconv_layers: int = 0
        if self.is_finetuning:
            assert self.lora_rank > 0
            self.n_lora_conv_layers, self.n_lora_tconv_layers = self.enable_lora_conv2d(rank=self.lora_rank)
            print(
                f"Applied LoRA to {self.n_lora_conv_layers} Conv2d layers "
                f"and {self.n_lora_tconv_layers} ConvTranspose2d layers"
            )

    @staticmethod
    def _replace_conv2d_with_lora(module: nn.Module, rank: int) -> tuple[int, int]:
        conv_count: int = 0
        tconv_count: int = 0
        for name, child in list(module.named_children()):
            if isinstance(child, (LoRAConv2d, LoRATConv2d)):
                continue
            if isinstance(child, nn.Conv2d):
                setattr(module, name, LoRAConv2d(base=child, rank=rank))
                conv_count += 1
            elif isinstance(child, nn.ConvTranspose2d):
                setattr(module, name, LoRATConv2d(base=child, rank=rank))
                tconv_count += 1
            else:
                # DFS: Recursive further into the child module
                child_conv_count, child_tconv_count = VAE_Precip._replace_conv2d_with_lora(
                    module=child, rank=rank
                )
                conv_count += child_conv_count
                tconv_count += child_tconv_count
        return conv_count, tconv_count

    def enable_lora_conv2d(self, rank: int) -> tuple[int, int]:
        assert rank > 0
        return VAE_Precip._replace_conv2d_with_lora(module=self, rank=rank)
