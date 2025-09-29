from abc import ABC, abstractmethod
from functools import cached_property

from typing import Callable, Literal
import torch
import torch.nn as nn
from ..common import NamedModel


class _StepSinusoidEmbedding(nn.Module):

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

    def forward(self, step: torch.Tensor) -> torch.Tensor:
        batch_size: int = step.shape[0]
        assert step.shape == (batch_size, 1)
        step = step.to(dtype=torch.float32)
        sinusoid: torch.Tensor = torch.zeros(
            (batch_size, self.embedding_dim), dtype=torch.float32, device=step.device,
        )
        scaled: torch.Tensor = step * self.w[None, :]
        assert scaled.shape == (batch_size, self.embedding_dim // 2)
        sinusoid[:, 0::2] = torch.sin(scaled)
        sinusoid[:, 1::2] = torch.cos(scaled)
        return sinusoid


class _NoiseScheduler(ABC):

    def __init__(self, n_steps: int, device: torch.device):
        assert n_steps > 0
        self.n_steps: int = n_steps
        self.device: torch.device = device

    @property
    @abstractmethod
    def beta_schedule(self) -> torch.Tensor:
        """
        Compute the beta schedule with beta_0 = 0.0.
        Returns array of shape (n_steps + 1,)
        """
        pass

    @property
    @abstractmethod
    def alpha_bar_schedule(self) -> torch.Tensor:
        """
        Compute the alpha_bar schedule with alpha_bar_0 = 1.0.
        Returns array of shape (n_steps + 1,)
        """
        pass


class LinearNoiseScheduler(_NoiseScheduler):

    def __init__(self, beta_min: float, beta_max: float, n_steps: int, device: torch.device) -> None:
        super().__init__(n_steps=n_steps, device=device)
        assert 0. < beta_min < beta_max
        self.beta_min: float = beta_min
        self.beta_max: float = beta_max

    @cached_property
    def beta_schedule(self) -> torch.Tensor:
        betas: torch.Tensor = torch.linspace(
            start=self.beta_min, end=self.beta_max, steps=self.n_steps, 
            dtype=torch.float32,
            device=self.device,
        )
        return torch.cat([torch.tensor([0.0], dtype=torch.float32, device=self.device), betas])

    @cached_property
    def alpha_bar_schedule(self) -> torch.Tensor:
        return torch.exp(torch.log(1.0 - self.beta_schedule).cumsum(dim=0))
    

class CosineNoiseScheduler(_NoiseScheduler):

    def __init__(self, n_steps: int, device: torch.device, cosine_offset: float = 0.008) -> None:
        super().__init__(n_steps=n_steps, device=device)
        assert cosine_offset >= 0, "cosine_offset must be non-negative"
        self.cosine_offset: float = cosine_offset

    @cached_property
    def alpha_bar_schedule(self) -> torch.Tensor:
        timesteps: torch.Tensor = torch.arange(self.n_steps + 1, dtype=torch.float32, device=self.device)
        f: Callable[[torch.Tensor], torch.Tensor] = (
            lambda x: torch.cos((x + self.cosine_offset) / (1 + self.cosine_offset) * torch.pi / 2) ** 2
        )
        return f(timesteps / self.n_steps) / f(torch.tensor(0.0))

    @cached_property
    def beta_schedule(self) -> torch.Tensor:
        betas: torch.Tensor = 1. - (self.alpha_bar_schedule[1:] / (self.alpha_bar_schedule[:-1] + 1e-10))
        return torch.cat([torch.tensor([0.0], dtype=torch.float32, device=self.device), betas])


class _LatentTransformerEncoder(nn.Module):

    def __init__(
        self, 
        in_dim: int, out_dim: int, 
        n_condition_days: int, n_heads: int, n_layers: int
    ) -> None:
        super().__init__()
        self.in_dim: int = in_dim
        self.out_dim: int = out_dim
        self.n_condition_days: int = n_condition_days
        self.n_heads: int = n_heads
        self.n_layers: int = n_layers
        self.linear_projection: nn.Module = nn.Linear(in_features=in_dim, out_features=out_dim)
        encoder_layer: nn.Module = nn.TransformerEncoderLayer(d_model=out_dim, nhead=n_heads, batch_first=True)
        self.encoder: nn.Module = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)
        self.positional_encoding: nn.Parameter = nn.Parameter(
            torch.randn(1, n_condition_days, out_dim)
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        batch_size, condition_dim, T, H, W = condition.shape
        assert self.in_dim == condition_dim
        assert self.n_condition_days == T
        output: torch.Tensor = condition.permute(0, 3, 4, 2, 1).flatten(0, 2)
        assert output.shape == (batch_size * H * W, T, condition_dim)
        output = self.linear_projection(output) + self.positional_encoding
        output = self.encoder(output)
        assert output.shape == (batch_size * H * W, T, self.out_dim)
        output = output.mean(dim=1, keepdim=False) # temporal pooling
        output = output.reshape(batch_size, H, W, self.out_dim).permute(0, 3, 1, 2)
        assert output.shape == (batch_size, self.out_dim, H, W)
        return output


class _StepEmbedding(nn.Module):

    def __init__(self, step_dim: int, step_out_dim: int):
        super().__init__()
        self.step_dim: int = step_dim
        self.step_out_dim: int = step_out_dim
        self.step_embedding_block = nn.Sequential(nn.SiLU(), nn.Linear(step_dim, step_out_dim))

    def forward(self, step: torch.Tensor) -> torch.Tensor:
        batch_size, t_in_dim = step.shape
        assert t_in_dim == self.step_dim
        output: torch.Tensor = self.step_embedding_block(step)
        assert output.shape == (batch_size, self.step_out_dim)
        return output


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


class _CrossAttention(nn.Module):

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.hidden_dim: int = hidden_dim
        self.n_heads: int = n_heads
        self.dropout: float = dropout
        self.target_group_norm: nn.Module = nn.GroupNorm(num_groups=n_heads, num_channels=hidden_dim)
        self.condition_group_norm: nn.Module = nn.GroupNorm(num_groups=n_heads, num_channels=hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True)

    def forward(self, target: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        target_N, target_D, target_H, target_W = target.shape
        condition_N, condition_D, condition_H, condition_W = condition.shape
        assert target_N == condition_N
        assert target_D == condition_D == self.hidden_dim, (
            "target and condition must be projected to the same hidden dim before _CrossAttention"
        )
        target_sequence_length: int = target_H * target_W
        condition_sequence_length: int = condition_H * condition_W

        # Preprocess target        
        target_flattened: torch.Tensor = target.reshape(target_N, self.hidden_dim, target_sequence_length)
        target_flattened = self.target_group_norm(target_flattened).transpose(1, 2)
        # Preprocess condition
        condition_flattened: torch.Tensor = condition.reshape(target_N, self.hidden_dim, condition_sequence_length)
        condition_flattened = self.condition_group_norm(condition_flattened).transpose(1, 2)
        # Fuse
        output_flattened: torch.Tensor = self.attention(
            query=target_flattened, key=condition_flattened, value=condition_flattened,
            need_weights=False, # for optimization purpose
        )[0]
        assert output_flattened.shape == (target_N, target_sequence_length, self.hidden_dim)
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
    Project (linearly) target, condition, and step embbeding into a common hidden_dim
    """

    def __init__(
        self,
        input_dim: int, condition_dim: int, step_dim: int, 
        hidden_dim: int, output_dim: int, 
        n_layers: int, n_attention_heads: int, condition_dropout: float,
        type: Literal["up", "down", "mid"],
    ):
        super().__init__()
        self.input_dim: int = input_dim
        self.condition_dim: int = condition_dim
        self.step_dim: int = step_dim
        self.hidden_dim: int = hidden_dim
        self.output_dim: int = output_dim
        self.n_layers: int = n_layers
        self.n_attention_heads: int = n_attention_heads
        self.condition_dropout: float = condition_dropout
        self.type: Literal["up", "down", "mid"] = type
        # Condition
        self.condition_projection: nn.Module = nn.Conv2d(
            in_channels=condition_dim, out_channels=hidden_dim, kernel_size=1, padding=0,
        )
        self.condition_conv_layers: nn.ModuleList = nn.ModuleList([
            _NormActConv(input_dim=hidden_dim, output_dim=hidden_dim, n_heads=n_attention_heads) for _ in range(n_layers)
        ])
        self.condition_res_blocks: nn.ModuleList = nn.ModuleList([
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
            for _ in range(n_layers)
        ])
        # Target
        self.target_projection: nn.Module = nn.Conv2d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=1, padding=0,
        )
        self.target_conv1_layers: nn.ModuleList = nn.ModuleList([
            _NormActConv(input_dim=hidden_dim, output_dim=hidden_dim, n_heads=n_attention_heads) for _ in range(n_layers)
        ])
        self.target_conv2_layers: nn.ModuleList = nn.ModuleList([
            _NormActConv(input_dim=hidden_dim, output_dim=hidden_dim, n_heads=n_attention_heads) for _ in range(n_layers)
        ])
        self.target_res_blocks: nn.ModuleList = nn.ModuleList([
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
            for _ in range(n_layers)
        ])
        # Attention
        self.cross_attention_layers: nn.ModuleList = nn.ModuleList([
            _CrossAttention(hidden_dim=hidden_dim, n_heads=n_attention_heads, dropout=condition_dropout)
            for _ in range(n_layers)
        ])
        # Step
        self.step_projection = nn.Linear(in_features=step_dim, out_features=hidden_dim)
        self.step_embedding_blocks = nn.ModuleList([
            _StepEmbedding(step_dim=hidden_dim, step_out_dim=hidden_dim)
            for _ in range(n_layers)
        ])
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

    def forward(self, target: torch.Tensor, condition: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        target_N, target_D, target_H, target_W = target.shape
        condition_N, condition_D, condition_H, condition_W = condition.shape
        step_N, step_D = step.shape
        # print(f"target.shape={target.shape}")
        # print(f"self.input_dim={self.input_dim}")
        # print(f"condition.shape={condition.shape}")
        # print(f"self.condition_dim={self.condition_dim}")
        # print("----------")
        assert target_N == condition_N == step_N
        assert target_D == self.input_dim
        assert condition_D == self.condition_dim
        assert step_D == self.step_dim
        
        # Linear projection
        target = self.target_projection(target)
        condition = self.condition_projection(condition)
        step = self.step_projection(step)

        target_output: torch.Tensor = target
        condition_output: torch.Tensor = condition
        for i in range(self.n_layers):
            target_resnet_input: torch.Tensor = target_output
            condition_resnet_input: torch.Tensor = condition_output
            # Resnet block (target)
            target_output = self.target_conv1_layers[i](target_output)
            target_output = target_output + self.step_embedding_blocks[i](step)[:, :, None, None]
            target_output = self.target_conv2_layers[i](target_output)
            target_output = target_output + self.target_res_blocks[i](target_resnet_input)
            # Resnet block (condition)
            condition_output: torch.Tensor = self.condition_conv_layers[i](condition_output)
            condition_output = condition_output + self.condition_res_blocks[i](condition_resnet_input)
            # Cross Attention
            target_output = target_output + self.cross_attention_layers[i](target=target_output, condition=condition_output)
        
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
        input_dim: int, condition_dim: int, step_dim: int, 
        hidden_dim: int, output_dim: int, 
        n_layers: int, n_attention_heads: int, condition_dropout: float,
    ):
        super().__init__(
            input_dim=input_dim, condition_dim=condition_dim, step_dim=step_dim, 
            hidden_dim=hidden_dim, output_dim=output_dim, 
            n_layers=n_layers, n_attention_heads=n_attention_heads, condition_dropout=condition_dropout,
            type="down",
        )


class _UpBlock(_ScalingBlock):

    def __init__(
        self,
        input_dim: int, down_output_dim: int, condition_dim: int, step_dim: int, 
        hidden_dim: int, output_dim: int, 
        n_layers: int, n_attention_heads: int, condition_dropout: float,
    ):
        super().__init__(
            input_dim=input_dim + down_output_dim, condition_dim=condition_dim, step_dim=step_dim, 
            hidden_dim=hidden_dim, output_dim=output_dim, 
            n_layers=n_layers, n_attention_heads=n_attention_heads, condition_dropout=condition_dropout,
            type="up",
        )


class _MidBlock(_ScalingBlock):

    def __init__(
        self,
        input_dim: int, condition_dim: int, step_dim: int, 
        hidden_dim: int, output_dim: int, 
        n_layers: int, n_attention_heads: int, condition_dropout: float,
    ):
        super().__init__(
            input_dim=input_dim, condition_dim=condition_dim, step_dim=step_dim, 
            hidden_dim=hidden_dim, output_dim=output_dim, 
            n_layers=n_layers, n_attention_heads=n_attention_heads, condition_dropout=condition_dropout,
            type="mid",
        )


class _ProjectionHead(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> None:
        super().__init__()
        self.in_dim: int = in_dim
        self.hidden_dim: int = hidden_dim
        self.out_dim: int = out_dim
        self.n_layers: int = n_layers
        assert n_layers >= 3, f"There must be at least 3 layers in the projection head"

        layers: list[nn.Module] = []
        for i in range(n_layers):
            if i == 0:
                _in_channels: int = in_dim
                _out_channels: int = hidden_dim
            elif i == n_layers - 1:
                _in_channels: int = hidden_dim
                _out_channels: int = out_dim
            else:
                _in_channels = _out_channels = hidden_dim

            layers.extend([
                nn.Conv2d(in_channels=_in_channels, out_channels=_out_channels, kernel_size=1),
                nn.ReLU(),
            ])

        self.head: nn.Module = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        assert C == self.in_dim
        return self.head(x)


class UNetDenoiser(NamedModel, nn.Module):

    def __init__(
        self,
        target_dim: int, 
        wind_dim: int, geopotential_dim: int, 
        thermaldynamic_dim: int, precipitation_dim: int, 
        step_dim: int, 
        condition_latent_embedding_dim: int,
        n_latent_embedding_layers: int,
        n_condition_days: int,
        down_out_dims: list[int], down_hidden_dims: list[int], 
        mid_out_dims: list[int], mid_hidden_dims: list[int], 
        up_out_dims: list[int], up_hidden_dims: list[int], 
        n_layers_per_scaling_block: int, n_layers_per_mid_block: int,
        n_attention_heads: int, condition_dropout: float,
        # projection_head_hidden_dim: int, n_head_layers: int
    ):
        super().__init__()
        self.target_dim: int = target_dim
        self.wind_dim: int = wind_dim
        self.geopotential_dim: int = geopotential_dim
        self.thermaldynamic_dim: int = thermaldynamic_dim
        self.precipitation_dim: int = precipitation_dim
        self.step_dim: int = step_dim
        self.condition_latent_embedding_dim: int = condition_latent_embedding_dim
        self.n_latent_embedding_layers: int = n_latent_embedding_layers
        self.n_condition_days: int = n_condition_days
        self.down_out_dims: list[int] = down_out_dims
        self.down_hidden_dims: list[int] = down_hidden_dims
        self.mid_out_dims: list[int] = mid_out_dims
        self.mid_hidden_dims: list[int] = mid_hidden_dims
        self.up_out_dims: list[int] = up_out_dims
        self.up_hidden_dims: list[int] = up_hidden_dims
        self.n_layers_per_scaling_block: int = n_layers_per_scaling_block
        self.n_layers_per_mid_block: int = n_layers_per_mid_block
        self.n_attention_heads: int = n_attention_heads
        self.condition_dropout: float = condition_dropout
        # self.projection_head_hidden_dim: int = projection_head_hidden_dim
        # self.n_head_layers: int = n_head_layers

        assert len(down_hidden_dims) == len(down_out_dims) == len(up_hidden_dims) == len(up_out_dims)
        assert len(mid_hidden_dims) == len(mid_out_dims)
        assert condition_latent_embedding_dim % 4 == 0, (
            f"condition_latent_embedding_dim must be divisible by 4 for "
            f"4 condition groups: ['wind', 'geopotential', 'thermaldynamic', 'precipitation'],"
            f"getting condition_latent_embedding_dim={condition_latent_embedding_dim}"
        )

        self.n_scaling_blocks: int = len(down_out_dims)
        self.n_mid_blocks: int = len(mid_out_dims)

        self.wind_latent_encoder: nn.Module = _LatentTransformerEncoder(
            in_dim=wind_dim, out_dim=condition_latent_embedding_dim // 4, 
            n_condition_days=n_condition_days, n_heads=n_attention_heads, n_layers=n_latent_embedding_layers,
        )
        self.geopotential_latent_encoder: nn.Module = _LatentTransformerEncoder(
            in_dim=geopotential_dim, out_dim=condition_latent_embedding_dim // 4, 
            n_condition_days=n_condition_days, n_heads=n_attention_heads, n_layers=n_latent_embedding_layers,
        )
        self.thermaldynamic_latent_encoder: nn.Module = _LatentTransformerEncoder(
            in_dim=thermaldynamic_dim, out_dim=condition_latent_embedding_dim // 4, 
            n_condition_days=n_condition_days, n_heads=n_attention_heads, n_layers=n_latent_embedding_layers,
        )
        self.precipitation_latent_encoder: nn.Module = _LatentTransformerEncoder(
            in_dim=precipitation_dim, out_dim=condition_latent_embedding_dim // 4, 
            n_condition_days=n_condition_days, n_heads=n_attention_heads, n_layers=n_latent_embedding_layers,
        )
        self.step_embedding_layer: nn.Module = _StepSinusoidEmbedding(embedding_dim=step_dim)
        self.down_blocks = nn.ModuleList([
            _DownBlock(
                input_dim=target_dim if i == 0 else down_out_dims[i - 1], 
                condition_dim=self.condition_latent_embedding_dim, 
                step_dim=step_dim,
                hidden_dim=down_hidden_dims[i], output_dim=down_out_dims[i],
                n_layers=n_layers_per_scaling_block, n_attention_heads=n_attention_heads,
                condition_dropout=condition_dropout,
            )
            for i in range(self.n_scaling_blocks)
        ])
        self.up_blocks = nn.ModuleList([
            _UpBlock(
                input_dim=mid_out_dims[-1] if i == 0 else up_out_dims[i - 1], 
                down_output_dim=down_out_dims[-i - 1],
                condition_dim=self.condition_latent_embedding_dim, step_dim=step_dim,
                hidden_dim=up_hidden_dims[i], output_dim=up_out_dims[i], 
                n_layers=n_layers_per_scaling_block, n_attention_heads=n_attention_heads, 
                condition_dropout=condition_dropout,
            )
            for i in range(self.n_scaling_blocks)
        ])
        self.mid_blocks = nn.ModuleList([
            _MidBlock(
                input_dim=down_out_dims[-1] if i == 0 else mid_out_dims[i - 1],
                condition_dim=self.condition_latent_embedding_dim, step_dim=step_dim,
                hidden_dim=mid_hidden_dims[i], output_dim=mid_out_dims[i],
                n_layers=n_layers_per_mid_block, n_attention_heads=n_attention_heads,
                condition_dropout=condition_dropout,
            )
            for i in range(self.n_mid_blocks)
        ])
        self.head: nn.Module = nn.Conv2d(
            in_channels=self.up_out_dims[-1], out_channels=target_dim, kernel_size=1,
        )
        # self.output_projection: nn.Module = _ProjectionHead(
        #     in_dim=self.up_out_dims[-1], 
        #     hidden_dim=projection_head_hidden_dim, 
        #     out_dim=target_dim, 
        #     n_layers=n_head_layers,
        # )

    def forward(
        self, 
        target: torch.Tensor, 
        wind_condition: torch.Tensor, 
        geopotential_condition: torch.Tensor, 
        thermaldynamic_condition: torch.Tensor, 
        precipitation_condition: torch.Tensor, 
        step: torch.Tensor
    ) -> torch.Tensor:
        target_N, target_D, target_H, target_W = target.shape
        wind_N, wind_D, wind_T, wind_H, wind_W = wind_condition.shape
        geopotential_N, geopotential_D, geopotential_T, geopotential_H, geopotential_W = geopotential_condition.shape
        thermaldynamic_N, thermaldynamic_D, thermaldynamic_T, thermaldynamic_H, thermaldynamic_W = thermaldynamic_condition.shape
        precipitation_N, precipitation_D, precipitation_T, precipitation_H, precipitation_W = precipitation_condition.shape
        step_N, step_D = step.shape
        assert target_N == wind_N == geopotential_N == thermaldynamic_N == precipitation_N == step_N
        assert wind_T == geopotential_T == thermaldynamic_T == precipitation_T
        assert wind_H == geopotential_H == thermaldynamic_H == precipitation_H
        assert wind_W == geopotential_W == thermaldynamic_W == precipitation_W
        assert target_D == self.target_dim
        assert step_D == 1
        
        # Check if too many scaling blocks on low-dim inputs
        _min_dim: float = min(target_H, target_W) / (2 ** self.n_scaling_blocks)
        assert _min_dim % 2 == 0, (
            f"too many scaling blocks (self.n_scaling_blocks={self.n_scaling_blocks}) "
            f"for target dimension: {(target_H, target_W)}"
        )
        # Temporal encoding
        wind_condition: torch.Tensor = self.wind_latent_encoder(condition=wind_condition)
        geopotential_condition: torch.Tensor = self.geopotential_latent_encoder(condition=geopotential_condition)
        thermaldynamic_condition: torch.Tensor = self.thermaldynamic_latent_encoder(condition=thermaldynamic_condition)
        precipitation_condition: torch.Tensor = self.precipitation_latent_encoder(condition=precipitation_condition)
        # print(f"wind_condition: {wind_condition.shape}")
        # print(f"geopotential_condition: {geopotential_condition.shape}")
        # print(f"thermaldynamic_condition: {thermaldynamic_condition.shape}")
        # print(f"precipitation_condition: {precipitation_condition.shape}")
        condition: torch.Tensor = torch.cat(
            tensors=[wind_condition, geopotential_condition, thermaldynamic_condition, precipitation_condition],
            dim=1,
        )
        assert condition.shape[1] == self.condition_latent_embedding_dim
        # Step embedding
        step_embedding: torch.Tensor = self.step_embedding_layer(step=step)
        # UNet
        down_input: torch.Tensor = target
        down_outputs: list[torch.Tensor] = []
        for i in range(self.n_scaling_blocks):
            down_outputs.append(self.down_blocks[i](target=down_input, condition=condition, step=step_embedding))
            down_input = down_outputs[-1]
        
        mid_output: torch.Tensor = down_outputs[-1]
        for i in range(self.n_mid_blocks):
            # print(f"mid_output.shape={mid_output.shape}")
            # print(f"self.mid_blocks[i].input_dim={self.mid_blocks[i].input_dim}")
            # print(f"--------")
            mid_output = self.mid_blocks[i](target=mid_output, condition=condition, step=step_embedding)

        up_output: torch.Tensor = mid_output
        # print(f"mid_output.shape={mid_output.shape}")
        for i in range(self.n_scaling_blocks):
            # print(i)
            # print(f"down_outputs[-1].shape={down_outputs[-1].shape}")
            # print(f"up_output.shape={up_output.shape}")
            assert down_outputs[-1].shape[1] == up_output.shape[1], (
                f"Wrong UNet configuration leads to dimension mismatch: "
                f"down_outputs[-1].shape={down_outputs[-1].shape}, and up_output.shape={up_output.shape}"
            )
            up_output = self.up_blocks[i](
                target=torch.cat([down_outputs.pop(), up_output], dim=1),
                condition=condition, step=step_embedding,
            )
            # print(f"up_output.shape={up_output.shape}")
            # print(f"--------")

        assert len(down_outputs) == 0, f"down_outputs must exhaust, getting {len(down_outputs)} items left"
        output: torch.Tensor = self.head(up_output)
        assert output.shape == target.shape, f"Shape mismatched: output.shape={output.shape} and target.shape={target.shape}."
        return output


class _DiffusionProcess:

    def __init__(self, noise_scheduler: _NoiseScheduler):
        self.noise_scheduler: _NoiseScheduler = noise_scheduler
        self.alpha_bar_schedule: torch.Tensor = self.noise_scheduler.alpha_bar_schedule
        self.beta_schedule: torch.Tensor = self.noise_scheduler.beta_schedule

    def compute_alpha_bar(self, step: torch.Tensor) -> torch.Tensor:
        step_N, _ = step.shape  # (batch_size, 1)
        alpha_bar: torch.Tensor = self.alpha_bar_schedule.to(step.device)[step.long()]
        assert alpha_bar.shape == (step_N, 1)
        alpha_bar = alpha_bar[:, :, None, None]
        return alpha_bar

    def compute_beta(self, step: torch.Tensor) -> torch.Tensor:
        step_N, _ = step.shape  # (batch_size, 1)
        beta: torch.Tensor = self.beta_schedule.to(step.device)[step.long()]
        assert beta.shape == (step_N, 1)
        beta = beta[:, :, None, None]
        return beta

    def compute_x0(self, target_k: torch.Tensor, predicted_noise: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
        target_N, target_D, target_H, target_W = target_k.shape
        noise_N, noise_D, noise_H, noise_W = predicted_noise.shape
        alpha_bar_N, _, _, _ = alpha_bar.shape  # (batch_size, 1)
        assert target_N == noise_N == alpha_bar_N
        # Original target prediction at k
        target_0: torch.Tensor = (target_k - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
        assert target_0.shape == target_k.shape
        return target_0

    def compute_tilde_beta(self, alpha_bar_prev: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
        return (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)

    def compute_sigma(self, tilde_beta: torch.Tensor, eta: float) -> torch.Tensor:
        return eta * torch.sqrt(tilde_beta)


class ForwardProcess(_DiffusionProcess):
    
    def add_noise(self, original_latent: torch.Tensor, step: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_N: int = original_latent.shape[0]
        step_N: int = step.shape[0]
        assert original_N == step_N # batch_size
        gaussian: torch.Tensor = torch.randn_like(original_latent)
        # Make sure schedules are on the same device before indexing with GPU tensor indices
        alpha_bar: torch.Tensor = self.compute_alpha_bar(step=step)
        noisy_latent: torch.Tensor = original_latent * (alpha_bar ** 0.5) + gaussian * ((1 - alpha_bar) ** 0.5)
        assert noisy_latent.shape == original_latent.shape
        return noisy_latent, gaussian
    

class ReverseProcess(_DiffusionProcess):

    def __init__(self, eta: float, noise_scheduler: _NoiseScheduler) -> None:
        super().__init__(noise_scheduler=noise_scheduler)
        self.eta: float = eta
        assert 0. <= self.eta <= 1.

    def sample(
        self, 
        target_k: torch.Tensor, predicted_noise: torch.Tensor, step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        target_N, target_D, target_H, target_W = target_k.shape
        noise_N, noise_D, noise_H, noise_W = predicted_noise.shape
        step_N, _ = step.shape  # (batch_size, 1)
        assert target_N == noise_N == step_N  # batch_size

        # Alpha bar
        alpha_bar: torch.Tensor = self.compute_alpha_bar(step=step)
        alpha_bar_prev: torch.Tensor = self.compute_alpha_bar(step=torch.clamp(input=step - 1, min=0))
        # Beta tilde
        tilde_beta: torch.Tensor = self.compute_tilde_beta(alpha_bar_prev=alpha_bar_prev, alpha_bar=alpha_bar)
        # Sigma
        sigma: torch.Tensor = self.compute_sigma(tilde_beta=tilde_beta, eta=self.eta)
        # Original target prediction at k
        target_0: torch.Tensor = self.compute_x0(target_k=target_k, predicted_noise=predicted_noise, alpha_bar=alpha_bar)
        assert target_0.shape == target_k.shape
        mean: torch.Tensor = torch.sqrt(alpha_bar_prev) * target_0 + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * predicted_noise
        return mean + sigma * torch.randn_like(target_k), target_0

