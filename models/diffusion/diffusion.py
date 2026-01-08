from abc import ABC, abstractmethod
from functools import cached_property

from typing import Callable, Literal
import torch
import torch.nn as nn
from models.common import NamedModel


class _SinusoidEmbedding(nn.Module):

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim: int = embedding_dim

        # Frequency scaling
        self.register_buffer(
            "w",
            1. / torch.pow(
                input=torch.tensor(10_000., dtype=torch.float),
                exponent=torch.arange(0, embedding_dim, 2, dtype=torch.float) / embedding_dim,
            )
        )
        assert self.w.shape == (embedding_dim // 2,)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        batch_size, length = t.shape
        t = t.to(dtype=torch.float32)
        sinusoid: torch.Tensor = torch.zeros(
            (batch_size, length, self.embedding_dim), dtype=torch.float32, device=t.device,
        )
        scaled: torch.Tensor = t[:, :, None] * self.w[None, None, :]
        assert scaled.shape == (batch_size, length, self.embedding_dim // 2)
        sinusoid[:, :, 0::2] = torch.sin(scaled)
        sinusoid[:, :, 1::2] = torch.cos(scaled)
        return sinusoid


class _BaseNoiseScheduler:

    @cached_property
    def snr(self) -> torch.Tensor:
        # NOTE: first element always inf
        return torch.sqrt(self.alpha_bar_schedule) / torch.sqrt(1 - self.alpha_bar_schedule)


class LinearNoiseScheduler(_BaseNoiseScheduler, nn.Module):

    def __init__(self, n_steps: int, beta_min: float, beta_max: float) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.beta_min: float = beta_min
        self.beta_max: float = beta_max
        self.register_buffer(
            name="beta_schedule", 
            tensor=torch.linspace(start=beta_min, end=beta_max, steps=n_steps, dtype=torch.float32),
            persistent=False,
        )
        alpha_bar_values: torch.Tensor = torch.exp(torch.log(1.0 - self.beta_schedule).cumsum(dim=0))
        alpha_bar_values = torch.cat([torch.ones(1, dtype=alpha_bar_values.dtype), alpha_bar_values], dim=0)
        assert alpha_bar_values.shape[0] == self.n_steps + 1    # len = self.n_steps + 1
        self.register_buffer(name="alpha_bar_schedule", tensor=alpha_bar_values, persistent=False)


class CosineNoiseScheduler(_BaseNoiseScheduler, nn.Module):

    def __init__(self, n_steps: int, cosine_offset: float = 0.008) -> None:
        super().__init__()
        assert cosine_offset >= 0, "cosine_offset must be non-negative"
        self.n_steps: int = n_steps
        self.cosine_offset: float = cosine_offset
        timesteps: torch.Tensor = torch.arange(self.n_steps + 1, dtype=torch.float32)   # len = self.n_steps + 1
        alpha_bar_values: torch.Tensor = self._f(timesteps / self.n_steps) / self._f(torch.tensor(0.0))
        alpha_bar_values = alpha_bar_values.clamp(min=1e-6)
        self.register_buffer(name="alpha_bar_schedule", tensor=alpha_bar_values, persistent=False)
        self.register_buffer(
            name="beta_schedule",
            tensor=1. - (self.alpha_bar_schedule[1:] / (self.alpha_bar_schedule[:-1] + 1e-10)),
            persistent=False
        )

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos((x + self.cosine_offset) / (1 + self.cosine_offset) * torch.pi / 2) ** 2


class _NormActConv(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, n_heads: int):
        super().__init__()
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.n_heads: int = n_heads
        self.group_norm: nn.Module = nn.GroupNorm(num_groups=n_heads, num_channels=input_dim)
        self.activation: nn.Module = nn.SiLU()
        self.conv2d: nn.Module = nn.Conv2d(
            in_channels=input_dim, out_channels=output_dim, kernel_size=3, padding=1,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, input_dim, H, W = input.shape
        assert input_dim == self.input_dim
        output: torch.Tensor = self.conv2d(self.activation(self.group_norm(input)))
        assert output.shape == (batch_size, self.output_dim, H, W)
        return output


class _UncertaintyAwareTransformer(nn.Module):

    def __init__(
        self, 
        hidden_dim: int, 
        n_encoder_layers: int, 
        n_decoder_layers: int, 
        n_heads: int, 
        maxlength: int,
    ):
        super().__init__()
        self.hidden_dim: int = hidden_dim
        self.n_encoder_layers: int = n_encoder_layers
        self.n_decoder_layers: int = n_decoder_layers
        self.n_heads: int = n_heads
        self.maxlenght: int = maxlength
        self.condition_fuse: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim * 2, out_channels=hidden_dim * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim * 2, out_channels=hidden_dim * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=1),
        )
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=n_heads, 
            num_encoder_layers=n_encoder_layers, num_decoder_layers=n_decoder_layers,
            batch_first=True,
        )
        self.target_pos_embedding = nn.Parameter(torch.randn(1, self.maxlenght, hidden_dim) * 0.02)
        self.condition_pos_embedding = nn.Parameter(torch.randn(1, self.maxlenght, hidden_dim) * 0.02)

    def forward(self, target: torch.Tensor, condition_mu: torch.Tensor, condition_logvar: torch.Tensor) -> torch.Tensor:
        target_N, target_D, target_H, target_W = target.shape
        assert condition_mu.shape == condition_logvar.shape
        condition_N, condition_D, condition_H, condition_W = condition_mu.shape
        assert target_N == condition_N
        assert target_D == condition_D == self.hidden_dim, (
            "target, condition_mu, condition_logvar must be projected to the same hidden dim before _UncertaintyAwareCrossAttention"
        )
        target_L: int = target_H * target_W
        condition_L: int = condition_H * condition_W

        # Preprocess target        
        target_flattened: torch.Tensor = target.flatten(start_dim=2, end_dim=3)
        target_flattened = target_flattened.transpose(1, 2) + self.target_pos_embedding[:, :target_L, :]
        # Preprocess condition
        condition: torch.Tensor = self.condition_fuse(torch.cat([condition_mu, condition_logvar], dim=1))
        condition_flattened: torch.Tensor = condition.flatten(start_dim=2, end_dim=3).transpose(1, 2)
        condition_flattened = condition_flattened + self.condition_pos_embedding[:, :condition_L, :]
        assert condition_flattened.shape == (condition_N, condition_L, condition_D)
        # Fuse
        output_flattened: torch.Tensor = self.transformer(tgt=target_flattened, src=condition_flattened)
        assert output_flattened.shape == (target_N, target_L, self.hidden_dim)
        output: torch.Tensor = output_flattened.transpose(1, 2).reshape_as(target)
        return output


class _DownSample(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        assert output_dim % 2 == 0

        # Downsampling using Conv
        self.conv2d: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1),
            nn.Conv2d(in_channels=input_dim, out_channels=output_dim // 2, kernel_size=4, stride=2, padding=1),
        )
        # Downsampling using MaxPool
        self.maxpool2d: nn.Module = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=input_dim, out_channels=output_dim // 2, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, input_dim, H, W = input.shape
        assert input_dim == self.input_dim
        output: torch.Tensor = torch.cat([self.conv2d(input), self.maxpool2d(input)], dim=1)
        assert output.shape == (batch_size, self.output_dim, H // 2, W // 2)
        return output


class _UpSample(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        assert output_dim % 2 == 0

        # Upsampling using ConvTranspose
        self.convtranspose2d: nn.Module = nn.Sequential(
            nn.ConvTranspose2d(in_channels=input_dim, out_channels=input_dim, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_channels=input_dim, out_channels=output_dim // 2, kernel_size=1, stride=1, padding=0),
        )
        # Upsampling using Upsample
        self.up2d: nn.Module = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"), 
            nn.Conv2d(in_channels=input_dim, out_channels=output_dim // 2, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, input_dim, H, W = input.shape
        assert input_dim == self.input_dim
        output: torch.Tensor = torch.cat([self.convtranspose2d(input), self.up2d(input)], dim=1)
        assert output.shape == (batch_size, self.output_dim, H * 2, W * 2)
        return output


class _ScalingBlock(nn.Module):

    """
    Project (linearly) target, condition, step, and day embbeding into a common hidden_dim
    """

    def __init__(
        self,
        input_dim: int, 
        wind_condition_dim: int, 
        mass_condition_dim: int, 
        thermal_condition_dim: int, 
        hydro_condition_dim: int, 
        precip_condition_dim: int, 
        hidden_dim: int, output_dim: int, 
        n_conv_layers: int, 
        n_transformer_encoder_layers: int, 
        n_transformer_decoder_layers: int,
        n_attention_heads: int,
        max_sequence_length: int,
        type: Literal["up", "down", "mid"],
    ):
        super().__init__()
        self.input_dim: int = input_dim
        self.wind_condition_dim: int = wind_condition_dim
        self.mass_condition_dim: int = mass_condition_dim
        self.thermal_condition_dim: int = thermal_condition_dim
        self.hydro_condition_dim: int = hydro_condition_dim
        self.precip_condition_dim: int = precip_condition_dim
        self.hidden_dim: int = hidden_dim
        self.output_dim: int = output_dim
        self.n_conv_layers: int = n_conv_layers; assert n_conv_layers % 2 == 0
        self.n_transformer_encoder_layers: int = n_transformer_encoder_layers
        self.n_transformer_decoder_layers: int = n_transformer_decoder_layers
        self.n_attention_heads: int = n_attention_heads
        self.max_sequence_length: int = max_sequence_length
        self.type: Literal["up", "down", "mid"] = type
        # Condition
        self.wind_mu_projection: nn.Module = nn.Conv2d(
            in_channels=wind_condition_dim, out_channels=hidden_dim, kernel_size=1, padding=0,
        )
        self.wind_logvar_projection: nn.Module = nn.Conv2d(
            in_channels=wind_condition_dim, out_channels=hidden_dim, kernel_size=1, padding=0,
        )
        self.mass_mu_projection: nn.Module = nn.Conv2d(
            in_channels=mass_condition_dim, out_channels=hidden_dim, kernel_size=1, padding=0,
        )
        self.mass_logvar_projection: nn.Module = nn.Conv2d(
            in_channels=mass_condition_dim, out_channels=hidden_dim, kernel_size=1, padding=0,
        )
        self.thermal_mu_projection: nn.Module = nn.Conv2d(
            in_channels=thermal_condition_dim, out_channels=hidden_dim, kernel_size=1, padding=0,
        )
        self.thermal_logvar_projection: nn.Module = nn.Conv2d(
            in_channels=thermal_condition_dim, out_channels=hidden_dim, kernel_size=1, padding=0,
        )
        self.hydro_mu_projection: nn.Module = nn.Conv2d(
            in_channels=hydro_condition_dim, out_channels=hidden_dim, kernel_size=1, padding=0,
        )
        self.hydro_logvar_projection: nn.Module = nn.Conv2d(
            in_channels=hydro_condition_dim, out_channels=hidden_dim, kernel_size=1, padding=0,
        )
        self.precip_mu_projection: nn.Module = nn.Conv2d(
            in_channels=precip_condition_dim, out_channels=hidden_dim, kernel_size=1, padding=0,
        )
        self.precip_logvar_projection: nn.Module = nn.Conv2d(
            in_channels=precip_condition_dim, out_channels=hidden_dim, kernel_size=1, padding=0,
        )
        # Target
        self.target_projection: nn.Module = nn.Conv2d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=1, padding=0,
        )
        self.target_conv1_layers: nn.ModuleList = nn.ModuleList([
            _NormActConv(input_dim=hidden_dim, output_dim=hidden_dim, n_heads=n_attention_heads)
            for _ in range(n_conv_layers)
        ])
        self.target_conv2_layers: nn.ModuleList = nn.ModuleList([
            _NormActConv(input_dim=hidden_dim, output_dim=hidden_dim, n_heads=n_attention_heads)
            for _ in range(n_conv_layers)
        ])
        self.target_res_blocks: nn.ModuleList = nn.ModuleList([
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
            for _ in range(n_conv_layers)
        ])
        # Attention
        self.wind_attention_layer: nn.Module = _UncertaintyAwareTransformer(
            hidden_dim=hidden_dim, 
            n_encoder_layers=n_transformer_encoder_layers, 
            n_decoder_layers=n_transformer_decoder_layers,
            n_heads=n_attention_heads,
            maxlength=max_sequence_length,
        )
        self.mass_attention_layer: nn.Module = _UncertaintyAwareTransformer(
            hidden_dim=hidden_dim, 
            n_encoder_layers=n_transformer_encoder_layers, 
            n_decoder_layers=n_transformer_decoder_layers,
            n_heads=n_attention_heads,
            maxlength=max_sequence_length,
        )
        self.thermal_attention_layer: nn.Module = _UncertaintyAwareTransformer(
            hidden_dim=hidden_dim, 
            n_encoder_layers=n_transformer_encoder_layers, 
            n_decoder_layers=n_transformer_decoder_layers,
            n_heads=n_attention_heads,
            maxlength=max_sequence_length,
        )
        self.hydro_attention_layer: nn.Module = _UncertaintyAwareTransformer(
            hidden_dim=hidden_dim, 
            n_encoder_layers=n_transformer_encoder_layers, 
            n_decoder_layers=n_transformer_decoder_layers,
            n_heads=n_attention_heads,
            maxlength=max_sequence_length,
        )
        self.precip_attention_layer: nn.Module = _UncertaintyAwareTransformer(
            hidden_dim=hidden_dim, 
            n_encoder_layers=n_transformer_encoder_layers, 
            n_decoder_layers=n_transformer_decoder_layers,
            n_heads=n_attention_heads,
            maxlength=max_sequence_length,
        )
        # Step
        self.step_embedding_layer: nn.Module = _SinusoidEmbedding(embedding_dim=hidden_dim)
        # Day
        self.day_embedding_layer: nn.Module = _SinusoidEmbedding(embedding_dim=hidden_dim)
        # Condition weights
        self.cond_weights: nn.Parameter = nn.Parameter(torch.ones(size=(hidden_dim * 6, hidden_dim)) * 0.5)
        # Scaling block
        if type == "down":
            self.scaling_block: nn.Module = _DownSample(input_dim=hidden_dim, output_dim=output_dim)
        elif type == "up":
            self.scaling_block: nn.Module = _UpSample(input_dim=hidden_dim, output_dim=output_dim)
        elif type == "mid":
            self.scaling_block: nn.Module = nn.Conv2d(
                in_channels=hidden_dim, out_channels=output_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            raise ValueError("Invalid type for _ScalingBlock, must be either 'up' or 'down' or 'mid'")

    def forward(
        self, 
        target: torch.Tensor, 
        wind_mu: torch.Tensor, wind_logvar: torch.Tensor, 
        mass_mu: torch.Tensor, mass_logvar: torch.Tensor, 
        thermal_mu: torch.Tensor, thermal_logvar: torch.Tensor, 
        hydro_mu: torch.Tensor, hydro_logvar: torch.Tensor, 
        precip_mu: torch.Tensor, precip_logvar: torch.Tensor, 
        step: torch.Tensor, days: torch.Tensor
    ) -> torch.Tensor:
        assert wind_mu.shape == wind_logvar.shape
        assert mass_mu.shape == mass_logvar.shape
        assert thermal_mu.shape == thermal_logvar.shape
        assert hydro_mu.shape == hydro_logvar.shape
        assert precip_mu.shape == precip_logvar.shape

        target_N, target_D, target_H, target_W = target.shape
        wind_N, wind_D, wind_H, wind_W = wind_mu.shape
        mass_N, mass_D, mass_H, mass_W = mass_mu.shape
        thermal_N, thermal_D, thermal_H, thermal_W = thermal_mu.shape
        hydro_N, hydro_D, hydro_H, hydro_W = hydro_mu.shape
        precip_N, precip_D, precip_H, precip_W = precip_mu.shape
        step_N, step_L = step.shape
        days_N, days_L = days.shape
        assert target_N == wind_N == mass_N == thermal_N == hydro_N == precip_N == step_N == days_N
        assert target_D == self.input_dim
        assert wind_D == self.wind_condition_dim
        assert mass_D == self.mass_condition_dim
        assert thermal_D == self.thermal_condition_dim
        assert hydro_D == self.hydro_condition_dim
        assert precip_D == self.precip_condition_dim
        
        # Linear projection
        target = self.target_projection(target)
        wind_mu = self.wind_mu_projection(wind_mu)
        wind_logvar = self.wind_logvar_projection(wind_logvar)
        mass_mu = self.mass_mu_projection(mass_mu)
        mass_logvar = self.mass_logvar_projection(mass_logvar)
        thermal_mu = self.thermal_mu_projection(thermal_mu)
        thermal_logvar = self.thermal_logvar_projection(thermal_logvar)
        hydro_mu = self.hydro_mu_projection(hydro_mu)
        hydro_logvar = self.hydro_logvar_projection(hydro_logvar)
        precip_mu = self.precip_mu_projection(precip_mu)
        precip_logvar = self.precip_logvar_projection(precip_logvar)
        step = self.step_embedding_layer(step).mean(dim=1)
        days = self.day_embedding_layer(days).mean(dim=1)
        assert step.shape == (step_N, self.hidden_dim)
        assert days.shape == (days_N, self.hidden_dim)

        target_output: torch.Tensor = target
        for i in range(self.n_conv_layers):
            target_resnet_input: torch.Tensor = target_output
            # Resnet block (target)
            target_output = (
                self.target_conv1_layers[i](target_output) 
                + step[:, :, None, None] + days[:, :, None, None]
            )
            target_output = (
                self.target_conv2_layers[i](target_output) 
                + self.target_res_blocks[i](target_resnet_input)
            )

        # Cross Attention
        wind_condition: torch.Tensor = self.wind_attention_layer(
            target=target_output, condition_mu=wind_mu, condition_logvar=wind_logvar,
        )
        mass_condition: torch.Tensor = self.mass_attention_layer(
            target=target_output, condition_mu=mass_mu, condition_logvar=mass_logvar,
        )
        thermal_condition: torch.Tensor = self.thermal_attention_layer(
            target=target_output, condition_mu=thermal_mu, condition_logvar=thermal_logvar,
        )
        hydro_condition: torch.Tensor = self.hydro_attention_layer(
            target=target_output, condition_mu=hydro_mu, condition_logvar=hydro_logvar,
        )
        precip_condition: torch.Tensor = self.precip_attention_layer(
            target=target_output, condition_mu=precip_mu, condition_logvar=precip_logvar,
        )
        assert (
            target_output.shape == wind_condition.shape == mass_condition.shape 
            == thermal_condition.shape == hydro_condition.shape == precip_condition.shape
        )
        concat: torch.Tensor = torch.cat(
            tensors=[
                target_output, 
                wind_condition, mass_condition, thermal_condition, hydro_condition, precip_condition,
            ],
            dim=1
        )
        target_output = torch.einsum("nihw,io->nohw", [concat, self.cond_weights])
        assert target_output.shape == (target_N, self.hidden_dim, target_H, target_W)
        # Scaling
        target_output = self.scaling_block(target_output)
        if self.type == "down":
            assert target_output.shape == (target_N, self.output_dim, target_H // 2, target_W // 2)
        elif self.type == "up":
            assert target_output.shape == (target_N, self.output_dim, target_H * 2, target_W * 2)
        elif self.type == "mid":
            assert target_output.shape == (target_N, self.output_dim, target_H, target_W)
        else:
            raise ValueError("Invalid type for _ScalingBlock, must be either 'up' or 'down' or 'mid'")

        return target_output
        

class _DownBlock(_ScalingBlock):

    def __init__(
        self,
        input_dim: int, 
        wind_condition_dim: int, 
        mass_condition_dim: int, 
        thermal_condition_dim: int, 
        hydro_condition_dim: int, 
        precip_condition_dim: int, 
        hidden_dim: int, output_dim: int, 
        n_conv_layers: int, 
        n_transformer_encoder_layers: int, 
        n_transformer_decoder_layers: int, 
        n_attention_heads: int,
        max_sequence_length: int,
    ):
        super().__init__(
            input_dim=input_dim, 
            wind_condition_dim=wind_condition_dim, 
            mass_condition_dim=mass_condition_dim, 
            thermal_condition_dim=thermal_condition_dim, 
            hydro_condition_dim=hydro_condition_dim, 
            precip_condition_dim=precip_condition_dim, 
            hidden_dim=hidden_dim, output_dim=output_dim, 
            n_conv_layers=n_conv_layers, 
            n_transformer_encoder_layers=n_transformer_encoder_layers,
            n_transformer_decoder_layers=n_transformer_decoder_layers,
            n_attention_heads=n_attention_heads,
            max_sequence_length=max_sequence_length,
            type="down",
        )

class _UpBlock(_ScalingBlock):

    def __init__(
        self,
        input_dim: int, 
        down_output_dim: int,
        wind_condition_dim: int, 
        mass_condition_dim: int, 
        thermal_condition_dim: int, 
        hydro_condition_dim: int, 
        precip_condition_dim: int, 
        hidden_dim: int, output_dim: int, 
        n_conv_layers: int, 
        n_transformer_encoder_layers: int,
        n_transformer_decoder_layers: int,
        n_attention_heads: int,
        max_sequence_length: int,
    ):
        super().__init__(
            input_dim=input_dim + down_output_dim,
            wind_condition_dim=wind_condition_dim, 
            mass_condition_dim=mass_condition_dim, 
            thermal_condition_dim=thermal_condition_dim, 
            hydro_condition_dim=hydro_condition_dim, 
            precip_condition_dim=precip_condition_dim, 
            hidden_dim=hidden_dim, output_dim=output_dim, 
            n_conv_layers=n_conv_layers, 
            n_transformer_encoder_layers=n_transformer_encoder_layers,
            n_transformer_decoder_layers=n_transformer_decoder_layers,
            n_attention_heads=n_attention_heads,
            max_sequence_length=max_sequence_length,
            type="up",
        )


class _MidBlock(_ScalingBlock):

    def __init__(
        self,
        input_dim: int, 
        wind_condition_dim: int, 
        mass_condition_dim: int, 
        thermal_condition_dim: int, 
        hydro_condition_dim: int, 
        precip_condition_dim: int, 
        hidden_dim: int, output_dim: int, 
        n_conv_layers: int, 
        n_transformer_encoder_layers: int, 
        n_transformer_decoder_layers: int, 
        n_attention_heads: int,
        max_sequence_length: int,
    ):
        super().__init__(
            input_dim=input_dim, 
            wind_condition_dim=wind_condition_dim, 
            mass_condition_dim=mass_condition_dim, 
            thermal_condition_dim=thermal_condition_dim, 
            hydro_condition_dim=hydro_condition_dim, 
            precip_condition_dim=precip_condition_dim, 
            hidden_dim=hidden_dim, output_dim=output_dim, 
            n_conv_layers=n_conv_layers, 
            n_transformer_encoder_layers=n_transformer_encoder_layers,
            n_transformer_decoder_layers=n_transformer_decoder_layers,
            n_attention_heads=n_attention_heads,
            max_sequence_length=max_sequence_length,
            type="mid",
        )


class UNetDenoiser(NamedModel, nn.Module):

    def __init__(
        self,
        target_dim: int, 
        wind_condition_dim: int, 
        mass_condition_dim: int, 
        thermal_condition_dim: int, 
        hydro_condition_dim: int, 
        precip_condition_dim: int, 
        n_condition_days: int,
        down_out_dims: list[int], down_hidden_dims: list[int], 
        mid_out_dims: list[int], mid_hidden_dims: list[int], 
        up_out_dims: list[int], up_hidden_dims: list[int], 
        n_conv_layers_per_scaling_block: int, 
        n_transformer_encoder_layers_per_scaling_block: int, 
        n_transformer_decoder_layers_per_scaling_block: int, 
        n_conv_layers_per_mid_block: int, 
        n_transformer_encoder_layers_per_mid_block: int,
        n_transformer_decoder_layers_per_mid_block: int,
        n_attention_heads: int, max_sequence_length: int,
        switch_ratio: float,
    ):
        super().__init__()
        self.target_dim: int = target_dim
        self.wind_condition_dim: int = wind_condition_dim
        self.mass_condition_dim: int = mass_condition_dim
        self.thermal_condition_dim: int = thermal_condition_dim
        self.hydro_condition_dim: int = hydro_condition_dim
        self.precip_condition_dim: int = precip_condition_dim
        self.n_condition_days: int = n_condition_days
        self.down_out_dims: list[int] = down_out_dims
        self.down_hidden_dims: list[int] = down_hidden_dims
        self.mid_out_dims: list[int] = mid_out_dims
        self.mid_hidden_dims: list[int] = mid_hidden_dims
        self.up_out_dims: list[int] = up_out_dims
        self.up_hidden_dims: list[int] = up_hidden_dims
        self.n_conv_layers_per_scaling_block: int = n_conv_layers_per_scaling_block
        self.n_transformer_encoder_layers_per_scaling_block: int = n_transformer_encoder_layers_per_scaling_block
        self.n_transformer_decoder_layers_per_scaling_block: int = n_transformer_decoder_layers_per_scaling_block
        self.n_conv_layers_per_mid_block: int = n_conv_layers_per_mid_block
        self.n_transformer_encoder_layers_per_mid_block: int = n_transformer_encoder_layers_per_mid_block
        self.n_transformer_decoder_layers_per_mid_block: int = n_transformer_decoder_layers_per_mid_block
        self.n_attention_heads: int = n_attention_heads
        self.max_sequence_length: int = max_sequence_length
        self.switch_ratio: float = switch_ratio

        assert len(down_hidden_dims) == len(down_out_dims) == len(up_hidden_dims) == len(up_out_dims)
        assert len(mid_hidden_dims) == len(mid_out_dims)

        self.n_scaling_blocks: int = len(down_out_dims)
        self.n_mid_blocks: int = len(mid_out_dims)

        self.down_blocks = nn.ModuleList([
            _DownBlock(
                input_dim=target_dim if i == 0 else down_out_dims[i - 1], 
                wind_condition_dim=wind_condition_dim, 
                mass_condition_dim=mass_condition_dim, 
                thermal_condition_dim=thermal_condition_dim, 
                hydro_condition_dim=hydro_condition_dim, 
                precip_condition_dim=precip_condition_dim, 
                hidden_dim=down_hidden_dims[i], output_dim=down_out_dims[i],
                n_conv_layers=n_conv_layers_per_scaling_block, 
                n_transformer_encoder_layers=n_transformer_encoder_layers_per_scaling_block,
                n_transformer_decoder_layers=n_transformer_decoder_layers_per_scaling_block,
                n_attention_heads=n_attention_heads,
                max_sequence_length=max_sequence_length,
            )
            for i in range(self.n_scaling_blocks)
        ])
        self.up_blocks = nn.ModuleList([
            _UpBlock(
                input_dim=mid_out_dims[-1] if i == 0 else up_out_dims[i - 1], 
                down_output_dim=down_out_dims[-i - 1],
                wind_condition_dim=wind_condition_dim, 
                mass_condition_dim=mass_condition_dim, 
                thermal_condition_dim=thermal_condition_dim, 
                hydro_condition_dim=hydro_condition_dim, 
                precip_condition_dim=precip_condition_dim, 
                hidden_dim=up_hidden_dims[i], output_dim=up_out_dims[i], 
                n_conv_layers=n_conv_layers_per_scaling_block, 
                n_transformer_encoder_layers=n_transformer_encoder_layers_per_scaling_block,
                n_transformer_decoder_layers=n_transformer_decoder_layers_per_scaling_block,
                n_attention_heads=n_attention_heads, 
                max_sequence_length=max_sequence_length,
            )
            for i in range(self.n_scaling_blocks)
        ])
        self.mid_blocks = nn.ModuleList([
            _MidBlock(
                input_dim=down_out_dims[-1] if i == 0 else mid_out_dims[i - 1],
                wind_condition_dim=wind_condition_dim, 
                mass_condition_dim=mass_condition_dim, 
                thermal_condition_dim=thermal_condition_dim, 
                hydro_condition_dim=hydro_condition_dim, 
                precip_condition_dim=precip_condition_dim, 
                hidden_dim=mid_hidden_dims[i], output_dim=mid_out_dims[i],
                n_conv_layers=n_conv_layers_per_mid_block, 
                n_transformer_encoder_layers=n_transformer_encoder_layers_per_mid_block,
                n_transformer_decoder_layers=n_transformer_decoder_layers_per_mid_block,
                n_attention_heads=n_attention_heads,
                max_sequence_length=max_sequence_length,
            )
            for i in range(self.n_mid_blocks)
        ])
        self.head: nn.Module = nn.Conv2d(
            in_channels=self.up_out_dims[-1], out_channels=target_dim, kernel_size=1,
        )

    def forward(
        self, 
        target: torch.Tensor, 
        wind_mu: torch.Tensor, wind_logvar: torch.Tensor,
        mass_mu: torch.Tensor, mass_logvar: torch.Tensor,
        thermal_mu: torch.Tensor, thermal_logvar: torch.Tensor,
        hydro_mu: torch.Tensor, hydro_logvar: torch.Tensor,
        precip_mu: torch.Tensor, precip_logvar: torch.Tensor,
        integer_step: torch.Tensor, condition_days: torch.Tensor,
    ) -> torch.Tensor:
        assert wind_mu.shape == wind_logvar.shape
        assert mass_mu.shape == mass_logvar.shape
        assert thermal_mu.shape == thermal_logvar.shape
        assert hydro_mu.shape == hydro_logvar.shape
        assert precip_mu.shape == precip_logvar.shape
        target_N, target_D, target_H, target_W = target.shape
        wind_N, wind_D, wind_H, wind_W = wind_mu.shape
        mass_N, mass_D, mass_H, mass_W = mass_mu.shape
        thermal_N, thermal_D, thermal_H, thermal_W = thermal_mu.shape
        hydro_N, hydro_D, hydro_H, hydro_W = hydro_mu.shape
        precip_N, precip_D, precip_H, precip_W = precip_mu.shape
        step_N, step_L = integer_step.shape
        day_N, day_L = condition_days.shape
        assert target_N == wind_N == mass_N == thermal_N == hydro_N == precip_N == step_N == day_N
        assert target_D == self.target_dim
        assert wind_D == self.wind_condition_dim
        assert mass_D == self.mass_condition_dim
        assert thermal_D == self.thermal_condition_dim
        assert hydro_D == self.hydro_condition_dim
        assert precip_D == self.precip_condition_dim
        assert step_L == 1
        assert day_L == self.n_condition_days
        
        # Check if too many scaling blocks on low-dim inputs
        _min_dim: float = min(target_H, target_W) / (2 ** self.n_scaling_blocks)
        assert _min_dim % 2 == 0, (
            f"too many scaling blocks (self.n_scaling_blocks={self.n_scaling_blocks}) "
            f"for target dimension: {(target_H, target_W)}"
        )
        # Switch condition on/off randomly during training
        if self.training:
            switch: torch.Tensor = torch.rand(size=(target_N,), device=target.device) > self.switch_ratio
            switch = switch.float()
            # Condition switch
            wind_mu = wind_mu * switch[:, None, None, None]
            wind_logvar = wind_logvar * switch[:, None, None, None]
            mass_mu = mass_mu * switch[:, None, None, None]
            mass_logvar = mass_logvar * switch[:, None, None, None]
            thermal_mu = thermal_mu * switch[:, None, None, None]
            thermal_logvar = thermal_logvar * switch[:, None, None, None]
            hydro_mu = hydro_mu * switch[:, None, None, None]
            hydro_logvar = hydro_logvar * switch[:, None, None, None]
            precip_mu = precip_mu * switch[:, None, None, None]
            precip_logvar = precip_logvar * switch[:, None, None, None]

        # UNet
        down_input: torch.Tensor = target
        down_outputs: list[torch.Tensor] = []
        for i in range(self.n_scaling_blocks):
            down_outputs.append(
                self.down_blocks[i](
                    target=down_input,
                    wind_mu=wind_mu, wind_logvar=wind_logvar, 
                    mass_mu=mass_mu, mass_logvar=mass_logvar,
                    thermal_mu=thermal_mu, thermal_logvar=thermal_logvar, 
                    hydro_mu=hydro_mu, hydro_logvar=hydro_logvar,
                    precip_mu=precip_mu, precip_logvar=precip_logvar,
                    step=integer_step, days=condition_days,
                )
            )
            down_input = down_outputs[-1]
        
        mid_output: torch.Tensor = down_outputs[-1]
        for i in range(self.n_mid_blocks):
            mid_output = self.mid_blocks[i](
                target=mid_output,
                wind_mu=wind_mu, wind_logvar=wind_logvar, 
                mass_mu=mass_mu, mass_logvar=mass_logvar,
                thermal_mu=thermal_mu, thermal_logvar=thermal_logvar, 
                hydro_mu=hydro_mu, hydro_logvar=hydro_logvar,
                precip_mu=precip_mu, precip_logvar=precip_logvar,
                step=integer_step, days=condition_days,
            )

        up_output: torch.Tensor = mid_output
        for i in range(self.n_scaling_blocks):
            skip: torch.Tensor = down_outputs.pop()
            assert skip.shape[2:] == up_output.shape[2:], (
                "Skip connection and decoder feature maps must share spatial dimensions: "
                f"skip.shape={skip.shape}, up_output.shape={up_output.shape}"
            )
            concat: torch.Tensor = torch.cat([skip, up_output], dim=1)
            expected_channels: int = self.up_blocks[i].input_dim
            assert concat.shape[1] == expected_channels, (
                "Concatenated skip features do not match expected channel count for up block: "
                f"concat.shape[1]={concat.shape[1]}, expected_channels={expected_channels}"
            )
            up_output = self.up_blocks[i](
                target=concat,
                wind_mu=wind_mu, wind_logvar=wind_logvar,
                mass_mu=mass_mu, mass_logvar=mass_logvar,
                thermal_mu=thermal_mu, thermal_logvar=thermal_logvar,
                hydro_mu=hydro_mu, hydro_logvar=hydro_logvar,
                precip_mu=precip_mu, precip_logvar=precip_logvar,
                step=integer_step, days=condition_days,
            )

        assert len(down_outputs) == 0, f"down_outputs must exhaust, getting {len(down_outputs)} items left"
        output: torch.Tensor = self.head(up_output)
        assert output.shape == target.shape, f"Shape mismatched: output.shape={output.shape} and target.shape={target.shape}."
        return output


class _DiffusionProcess:

    def __init__(self, noise_scheduler: LinearNoiseScheduler | CosineNoiseScheduler):
        self.noise_scheduler: LinearNoiseScheduler | CosineNoiseScheduler = noise_scheduler
        self.alpha_bar_schedule: torch.Tensor = self.noise_scheduler.alpha_bar_schedule
        self.beta_schedule: torch.Tensor = self.noise_scheduler.beta_schedule

    def compute_alpha_bar(self, step: torch.Tensor) -> torch.Tensor:
        step_N, _ = step.shape  # (batch_size, 1)
        # alpha ranges from k=0,...,K
        assert torch.all(step.long() >= 0)
        assert torch.all(step.long() <= self.noise_scheduler.n_steps)
        # print(f"self.alpha_bar_schedule: {self.alpha_bar_schedule}")
        alpha_bar: torch.Tensor = self.alpha_bar_schedule.to(step.device)[step.long()]
        assert alpha_bar.shape == (step_N, 1)
        alpha_bar = alpha_bar[:, :, None, None]
        return alpha_bar

    def compute_beta(self, step: torch.Tensor) -> torch.Tensor:
        step_N, _ = step.shape  # (batch_size, 1)
        # beta ranges from k=1,...,K
        assert torch.all(step.long() - 1 >= 0)
        assert torch.all(step.long() - 1 <= self.noise_scheduler.n_steps - 1)
        beta: torch.Tensor = self.beta_schedule.to(step.device)[step.long() - 1]
        assert beta.shape == (step_N, 1)
        beta = beta[:, :, None, None]
        return beta
    
    def compute_tilde_beta(self, alpha_bar_prev: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
        return (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)

    def compute_sigma(self, tilde_beta: torch.Tensor, eta: float) -> torch.Tensor:
        return eta * torch.sqrt(tilde_beta)
    
    def compute_velocity(self, alpha_bar: torch.Tensor, target_0: torch.Tensor, gaussian: torch.Tensor) -> torch.Tensor:
        return - torch.sqrt(1 - alpha_bar) * target_0 + torch.sqrt(alpha_bar) * gaussian

    def compute_x0(self, target_k: torch.Tensor, predicted_velocity: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
        target_N, target_D, target_H, target_W = target_k.shape
        velocity_N, velocity_D, velocity_H, velocity_W = predicted_velocity.shape
        alpha_bar_N, _, _, _ = alpha_bar.shape  # (batch_size, 1)
        assert target_N == velocity_N == alpha_bar_N
        # Original target prediction at k
        target_0: torch.Tensor = torch.sqrt(alpha_bar) * target_k - torch.sqrt(1 - alpha_bar) * predicted_velocity
        assert target_0.shape == target_k.shape
        return target_0

    def compute_gaussian(self, target_k: torch.Tensor, predicted_velocity: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
        target_N, target_D, target_H, target_W = target_k.shape
        velocity_N, velocity_D, velocity_H, velocity_W = predicted_velocity.shape
        alpha_bar_N, _, _, _ = alpha_bar.shape  # (batch_size, 1)
        assert target_N == velocity_N == alpha_bar_N
        # Original target prediction at k
        gaussian: torch.Tensor = torch.sqrt(1 - alpha_bar) * target_k + torch.sqrt(alpha_bar) * predicted_velocity
        assert gaussian.shape == target_k.shape
        return gaussian


class ForwardProcess(_DiffusionProcess):

    def add_noise(self, original_latent: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_N: int = original_latent.shape[0]
        step_N, _ = k.shape  # (batch_size, 1)
        assert original_N == step_N # batch_size
        alpha_bar: torch.Tensor = self.compute_alpha_bar(step=k)
        true_gaussian: torch.Tensor = torch.randn_like(original_latent)
        true_velocity: torch.Tensor = self.compute_velocity(
            alpha_bar=alpha_bar, target_0=original_latent, gaussian=true_gaussian,
        )
        noisy_latent: torch.Tensor = original_latent * torch.sqrt(alpha_bar) + true_gaussian * torch.sqrt(1 - alpha_bar)
        assert noisy_latent.shape == original_latent.shape
        return noisy_latent, true_velocity


class ReverseProcess(_DiffusionProcess):

    def __init__(self, eta: float, noise_scheduler: LinearNoiseScheduler | CosineNoiseScheduler) -> None:
        super().__init__(noise_scheduler=noise_scheduler)
        self.eta: float = eta
        assert 0. <= self.eta <= 1.

    def sample(
        self, 
        target_k: torch.Tensor, predicted_velocity: torch.Tensor, k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        target_N, target_D, target_H, target_W = target_k.shape
        velocity_N, velocity_D, velocity_H, velocity_W = predicted_velocity.shape
        step_N, _ = k.shape  # (batch_size, 1)
        assert target_N == velocity_N == step_N  # batch_size

        # Alpha bar
        alpha_bar: torch.Tensor = self.compute_alpha_bar(step=k)
        alpha_bar_prev: torch.Tensor = self.compute_alpha_bar(step=k - 1)
        # Beta tilde
        tilde_beta: torch.Tensor = self.compute_tilde_beta(alpha_bar_prev=alpha_bar_prev, alpha_bar=alpha_bar)
        # Sigma
        sigma: torch.Tensor = self.compute_sigma(tilde_beta=tilde_beta, eta=self.eta)
        # Original target prediction at k
        target_0: torch.Tensor = self.compute_x0(
            target_k=target_k, predicted_velocity=predicted_velocity, alpha_bar=alpha_bar
        )
        assert target_0.shape == target_k.shape
        predicted_gaussian: torch.Tensor = self.compute_gaussian(
            target_k=target_k, predicted_velocity=predicted_velocity, alpha_bar=alpha_bar
        )
        mean: torch.Tensor = (
            torch.sqrt(alpha_bar_prev) * target_0 
            + torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma**2, min=0.)) * predicted_gaussian
        )
        return mean + sigma * torch.randn_like(target_k), target_0
