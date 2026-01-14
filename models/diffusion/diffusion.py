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


class _ConditionFuser(nn.Module):

    def __init__(self, condition_dim: int):
        super().__init__()
        self.condition_dim: int = condition_dim
        self.a = nn.Parameter(torch.ones((1, 1, condition_dim)) / condition_dim)
        self.b = nn.Parameter(torch.ones((1, 1, condition_dim)) / condition_dim)
        self.c = nn.Parameter(torch.ones((1, 1, condition_dim)) / condition_dim)

    def forward(self, condition_mu: torch.Tensor, condition_logvar: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = condition_mu * self.a + condition_logvar * self.b + self.c
        assert output.shape == condition_mu.shape == condition_logvar.shape
        return output


class _TransformerFeedForward(nn.Module):

    def __init__(self, model_dim: int, feedforward_dim: int):
        super().__init__()
        self.model_dim: int = model_dim
        self.feedforward_dim: int = feedforward_dim
        self.net = nn.Sequential(
            nn.Linear(in_features=model_dim, out_features=feedforward_dim),
            nn.GELU(),
            nn.Linear(in_features=feedforward_dim, out_features=model_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TransformerEncoderLayer(nn.Module):

    def __init__(self, kv_dim: int, model_dim: int, n_heads: int, feedforward_dim: int):
        super().__init__()
        self.kv_dim: int = kv_dim
        self.model_dim: int = model_dim
        self.n_heads: int = n_heads
        self.feedforward_dim: int = feedforward_dim

        if kv_dim != model_dim:
            self.kv_projection = nn.Linear(in_features=kv_dim, out_features=model_dim)
        else:
            self.kv_projection = nn.Identity()
            
        self.self_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        self.feed_forward = _TransformerFeedForward(model_dim=model_dim, feedforward_dim=feedforward_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, kv: torch.Tensor) -> torch.Tensor:
        N, L, D = kv.shape
        kv = self.kv_projection(kv)
        output: torch.Tensor = self.self_attention(
            query=kv, key=kv, value=kv, need_weights=False
        )[0]
        kv = self.norm1(kv + output)
        kv = self.norm2(kv + self.feed_forward(kv))
        assert kv.shape == (N, L, self.model_dim)
        return kv
    

class _TransformerDecoderLayer(nn.Module):

    def __init__(self, q_dim: int, model_dim: int, n_heads: int, feedforward_dim: int):
        super().__init__()
        self.q_dim: int = q_dim
        self.model_dim: int = model_dim
        self.n_heads: int = n_heads
        self.feedforward_dim: int = feedforward_dim

        if q_dim != model_dim:
            self.q_projection = nn.Linear(in_features=q_dim, out_features=model_dim)
        else:
            self.q_projection = nn.Identity()

        # Decoder self-attention (symmetric)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        # Cross-attention (Q from decoder, K/V from encoder)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=n_heads,
            kdim=model_dim,
            vdim=model_dim,
            batch_first=True,
        )
        self.ffn = _TransformerFeedForward(model_dim=model_dim, feedforward_dim=feedforward_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_N, q_L, q_D = q.shape
        kv_N, kv_L, kv_D = kv.shape
        assert q_D == self.q_dim
        assert kv_D == self.model_dim
        q = self.q_projection(q)
        output: torch.Tensor = self.self_attention(query=q, key=q, value=q, need_weights=False)[0]
        q = self.norm1(q + output)

        # Cross-attention
        output = self.cross_attention(q, kv, kv, need_weights=False)[0]
        q = self.norm2(q + output)
        q = self.norm3(q + self.ffn(q))
        assert q.shape == (q_N, q_L, self.model_dim)
        return q


class _TransformerEncoder(nn.Module):

    def __init__(self, kv_dim: int, model_dim: int, n_heads: int, feedforward_dim: int, n_layers: int):
        super().__init__()
        self.kv_dim: int = kv_dim
        self.model_dim: int = model_dim
        self.n_heads: int = n_heads
        self.feedforward_dim: int = feedforward_dim
        self.n_layers: int = n_layers
        self.layers = nn.ModuleList([
            _TransformerEncoderLayer(
                kv_dim=kv_dim if i == 0 else model_dim,
                model_dim=model_dim, n_heads=n_heads, feedforward_dim=feedforward_dim
            )
            for i in range(n_layers)
        ])

    def forward(self, kv: torch.Tensor) -> torch.Tensor:
        N, L, D = kv.shape
        assert D == self.kv_dim
        for layer in self.layers:
            kv = layer(kv)
        assert kv.shape == (N, L, self.model_dim)
        return kv


class _TransformerDecoder(nn.Module):

    def __init__(self, q_dim: int, model_dim: int, n_heads: int, feedforward_dim: int, n_layers: int):
        super().__init__()
        self.q_dim: int = q_dim
        self.model_dim: int = model_dim
        self.n_heads: int = n_heads
        self.feedforward_dim: int = feedforward_dim
        self.n_layers: int = n_layers
        self.layers = nn.ModuleList([
            _TransformerDecoderLayer(
                q_dim=q_dim if i == 0 else model_dim,
                model_dim=model_dim, n_heads=n_heads, feedforward_dim=feedforward_dim
            )
            for i in range(n_layers)
        ])
        self.out_projection = nn.Sequential(
            nn.Linear(in_features=model_dim, out_features=feedforward_dim),
            nn.GELU(),
            nn.Linear(in_features=feedforward_dim, out_features=feedforward_dim),
            nn.GELU(),
            nn.Linear(in_features=feedforward_dim, out_features=q_dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_N, q_L, q_D = q.shape
        kv_N, kv_L, kv_D = kv.shape
        assert q_D == self.q_dim
        assert kv_D == self.model_dim
        for layer in self.layers:
            q = layer(q=q, kv=kv)
        
        q = self.out_projection(q)
        assert q.shape == (q_N, q_L, q_D)
        return q


class _Transformer(nn.Module):
    def __init__(self,
        q_dim: int, kv_dim: int, model_dim: int,
        n_heads: int, feedforward_dim: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        maxlength: int,
    ):
        super().__init__()
        self.q_dim: int = q_dim
        self.kv_dim: int = kv_dim
        self.model_dim: int = model_dim
        self.n_heads: int = n_heads
        self.feedforward_dim: int = feedforward_dim
        self.n_encoder_layers: int = n_encoder_layers
        self.n_decoder_layers: int = n_decoder_layers
        self.maxlength: int = maxlength

        self.encoder = _TransformerEncoder(
            kv_dim=kv_dim, model_dim=model_dim, n_heads=n_heads, 
            feedforward_dim=feedforward_dim, n_layers=n_encoder_layers,
        )
        self.decoder = _TransformerDecoder(
            q_dim=q_dim, model_dim=model_dim, n_heads=n_heads, 
            feedforward_dim=feedforward_dim, n_layers=n_decoder_layers
        )
        self.kv_pos_embedding = nn.Parameter(torch.randn(1, maxlength, kv_dim) * 0.02)
        self.q_pos_embedding = nn.Parameter(torch.randn(1, maxlength, q_dim) * 0.02)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_N, src_L, src_D = src.shape
        tgt_N, tgt_L, tgt_D = tgt.shape
        assert src_D == self.kv_dim
        assert tgt_D == self.q_dim
        src = src + self.kv_pos_embedding[:, :src_L, :]
        tgt = tgt + self.q_pos_embedding[:, :tgt_L, :]
        memory: torch.Tensor = self.encoder(kv=src)
        output: torch.Tensor = self.decoder(q=tgt, kv=memory)
        assert output.shape == (tgt_N, tgt_L, tgt_D)
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
        assert input_dim == self.input_dim, f"input_dim={input_dim} vs. self.input_dim={self.input_dim}"
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
        assert input_dim == self.input_dim, f"input_dim={input_dim} vs. self.input_dim={self.input_dim}"
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
        output_dim: int, 
        condition_dim: int, 
        n_condition_days: int,
        n_conv_layers: int, 
        in_H: int, in_W: int,
        transformer_model_dim: int, 
        transformer_feedforward_dim: int, 
        n_transformer_encoder_layers: int, 
        n_transformer_decoder_layers: int,
        n_attention_heads: int,
        type: Literal["up", "down", "mid"],
    ):
        super().__init__()
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.condition_dim: int = condition_dim
        self.transformer_model_dim: int = transformer_model_dim
        self.transformer_feedforward_dim: int = transformer_feedforward_dim
        self.n_conv_layers: int = n_conv_layers; assert n_conv_layers % 2 == 0
        self.in_H: int = in_H
        self.in_W: int = in_W
        self.n_transformer_encoder_layers: int = n_transformer_encoder_layers
        self.n_transformer_decoder_layers: int = n_transformer_decoder_layers
        self.n_attention_heads: int = n_attention_heads
        self.n_condition_days: int = n_condition_days
        self.type: Literal["up", "down", "mid"] = type

        # Condition
        self.condition_fuser: nn.Module = _ConditionFuser(condition_dim=condition_dim)
        # Target
        self.target_conv1_layers: nn.ModuleList = nn.ModuleList([
            _NormActConv(
                input_dim=input_dim if i == 0 else output_dim,
                output_dim=output_dim, n_heads=n_attention_heads
            )
            for i in range(n_conv_layers)
        ])
        self.target_conv2_layers: nn.ModuleList = nn.ModuleList([
            _NormActConv(
                input_dim=output_dim, output_dim=output_dim, n_heads=n_attention_heads
            )
            for i in range(n_conv_layers)
        ])
        self.target_res_blocks: nn.ModuleList = nn.ModuleList([
            nn.Conv2d(
                in_channels=input_dim if i == 0 else output_dim, 
                out_channels=output_dim, kernel_size=1
            )
            for i in range(n_conv_layers)
        ])
        # Attention
        self.transfomer: nn.Module = _Transformer(
            q_dim=output_dim * in_H * in_W,
            kv_dim=condition_dim,
            model_dim=transformer_model_dim,
            n_heads=n_attention_heads,
            feedforward_dim=transformer_feedforward_dim,
            n_encoder_layers=n_transformer_encoder_layers,
            n_decoder_layers=n_transformer_decoder_layers,
            maxlength=n_condition_days,
        )
        # Step
        self.step_embedding_layer: nn.Module = _SinusoidEmbedding(embedding_dim=output_dim)
        # Day
        self.day_embedding_layer: nn.Module = _SinusoidEmbedding(embedding_dim=output_dim)
        self.cond_weights: nn.Parameter = nn.Parameter(torch.randn(size=(1, 1, output_dim * in_H * in_W)))
        # Scaling block
        if type == "down":
            self.scaling_block: nn.Module = _DownSample(input_dim=output_dim, output_dim=output_dim)
            self.out_H: int = self.in_H // 2
            self.out_W: int = self.in_W // 2
        elif type == "up":
            self.scaling_block: nn.Module = _UpSample(input_dim=output_dim, output_dim=output_dim)
            self.out_H: int = self.in_H * 2
            self.out_W: int = self.in_W * 2
        elif type == "mid":
            self.scaling_block: nn.Module = nn.Conv2d(
                in_channels=output_dim, out_channels=output_dim, kernel_size=1, stride=1, padding=0
            )
            self.out_H: int = self.in_H
            self.out_W: int = self.in_W
        else:
            raise ValueError("Invalid type for _ScalingBlock, must be either 'up' or 'down' or 'mid'")

    def forward(
        self, 
        target: torch.Tensor, 
        condition_mu: torch.Tensor, condition_logvar: torch.Tensor,
        step: torch.Tensor, days: torch.Tensor
    ) -> torch.Tensor:
        assert condition_mu.shape == condition_logvar.shape
        condition_N, condition_L, condition_D = condition_mu.shape
        assert condition_D == self.condition_dim
        target_N, target_D, target_H, target_W = target.shape
        assert target_D == self.input_dim
        assert (target_H, target_W) == (self.in_H, self.in_W)
        step_N, step_L = step.shape
        days_N, days_L = days.shape
        assert target_N == condition_N == step_N == days_N
        
        condition: torch.Tensor = self.condition_fuser(
            condition_mu=condition_mu, condition_logvar=condition_logvar
        )
        step = self.step_embedding_layer(step).mean(dim=1)
        days = self.day_embedding_layer(days).mean(dim=1)
        assert step.shape == (step_N, self.output_dim)
        assert days.shape == (days_N, self.output_dim)

        for i in range(self.n_conv_layers):
            target_resnet_input: torch.Tensor = target
            # Resnet block (target)
            target = self.target_conv1_layers[i](target) + step[:, :, None, None] + days[:, :, None, None]
            target = self.target_conv2_layers[i](target) + self.target_res_blocks[i](target_resnet_input)

        assert target.shape == (target_N, self.output_dim, target_H, target_W)

        # Conditioning
        target = target.reshape(target_N, 1, self.output_dim * target_H * target_W)
        conditioned_target: torch.Tensor = self.transfomer(src=condition, tgt=target)
        assert conditioned_target.shape == target.shape
        weight: torch.Tensor = torch.sigmoid(self.cond_weights)
        # DEBUG
        # print(f"weight: {weight.mean()}")
        # print(f"weight: {weight.flatten()}")
        output: torch.Tensor = weight * conditioned_target + (1 - weight) * target
        assert output.shape == target.shape
        output = output.reshape(target_N, self.output_dim, target_H, target_W)

        # Scaling
        output = self.scaling_block(output)
        assert output.shape == (target_N, self.output_dim, self.out_H, self.out_W)
        return output
        

class _DownBlock(_ScalingBlock):


    def __init__(
        self,
        input_dim: int, 
        output_dim: int, 
        condition_dim: int, 
        n_condition_days: int,
        n_conv_layers: int, 
        in_H: int, in_W: int,
        transformer_model_dim: int,
        transformer_feedforward_dim: int,
        n_transformer_encoder_layers: int, 
        n_transformer_decoder_layers: int, 
        n_attention_heads: int,
    ):
        super().__init__(
            input_dim=input_dim, 
            output_dim=output_dim, 
            condition_dim=condition_dim, 
            n_condition_days=n_condition_days,
            n_conv_layers=n_conv_layers, 
            in_H=in_H, in_W=in_W,
            transformer_model_dim=transformer_model_dim, 
            transformer_feedforward_dim=transformer_feedforward_dim,
            n_transformer_encoder_layers=n_transformer_encoder_layers,
            n_transformer_decoder_layers=n_transformer_decoder_layers,
            n_attention_heads=n_attention_heads,
            type="down",
        )


class _UpBlock(_ScalingBlock):

    def __init__(
        self,
        input_dim: int, 
        output_dim: int, 
        down_output_dim: int,
        condition_dim: int, 
        n_condition_days: int,
        n_conv_layers: int, 
        in_H: int, in_W: int,
        transformer_model_dim: int, 
        transformer_feedforward_dim: int,
        n_transformer_encoder_layers: int,
        n_transformer_decoder_layers: int,
        n_attention_heads: int,
    ):
        super().__init__(
            input_dim=input_dim + down_output_dim,
            output_dim=output_dim, 
            condition_dim=condition_dim, 
            n_condition_days=n_condition_days, 
            n_conv_layers=n_conv_layers, 
            in_H=in_H, in_W=in_W,
            transformer_model_dim=transformer_model_dim,
            transformer_feedforward_dim=transformer_feedforward_dim,
            n_transformer_encoder_layers=n_transformer_encoder_layers,
            n_transformer_decoder_layers=n_transformer_decoder_layers,
            n_attention_heads=n_attention_heads,
            type="up",
        )


class _MidBlock(_ScalingBlock):

    def __init__(
        self,
        input_dim: int, 
        output_dim: int, 
        condition_dim: int, 
        n_condition_days: int,
        n_conv_layers: int, 
        in_H: int, in_W: int,
        transformer_model_dim: int,
        transformer_feedforward_dim: int,
        n_transformer_encoder_layers: int, 
        n_transformer_decoder_layers: int, 
        n_attention_heads: int,
    ):
        super().__init__(
            input_dim=input_dim, 
            output_dim=output_dim, 
            condition_dim=condition_dim, 
            n_conv_layers=n_conv_layers, 
            n_condition_days=n_condition_days,
            in_H=in_H, in_W=in_W,
            transformer_model_dim=transformer_model_dim, 
            transformer_feedforward_dim=transformer_feedforward_dim,
            n_transformer_encoder_layers=n_transformer_encoder_layers,
            n_transformer_decoder_layers=n_transformer_decoder_layers,
            n_attention_heads=n_attention_heads,
            type="mid",
        )


class UNetDenoiser(NamedModel, nn.Module):

    def __init__(
        self,
        target_dim: int, 
        condition_dim: int, 
        n_condition_days: int,
        in_H: int, in_W: int,
        down_out_dims: list[int], 
        mid_out_dims: list[int], 
        up_out_dims: list[int], 
        down_transformer_model_dims: list[int], 
        mid_transformer_model_dims: list[int], 
        up_transformer_model_dims: list[int], 
        transformer_feedforward_dim: int,
        n_conv_layers_per_scaling_block: int, 
        n_transformer_encoder_layers_per_scaling_block: int, 
        n_transformer_decoder_layers_per_scaling_block: int, 
        n_conv_layers_per_mid_block: int, 
        n_transformer_encoder_layers_per_mid_block: int,
        n_transformer_decoder_layers_per_mid_block: int,
        n_attention_heads: int,
        switch_ratio: float,
    ):
        super().__init__()
        self.target_dim: int = target_dim
        self.condition_dim: int = condition_dim
        self.n_condition_days: int = n_condition_days
        self.in_H: int = in_H
        self.in_W: int = in_W
        self.down_out_dims: list[int] = down_out_dims
        self.down_transformer_model_dims: list[int] = down_transformer_model_dims
        self.mid_out_dims: list[int] = mid_out_dims
        self.mid_transformer_model_dims: list[int] = mid_transformer_model_dims
        self.up_out_dims: list[int] = up_out_dims
        self.up_transformer_model_dims: list[int] = up_transformer_model_dims
        self.transformer_feedforward_dim: int = transformer_feedforward_dim
        self.n_conv_layers_per_scaling_block: int = n_conv_layers_per_scaling_block
        self.n_transformer_encoder_layers_per_scaling_block: int = n_transformer_encoder_layers_per_scaling_block
        self.n_transformer_decoder_layers_per_scaling_block: int = n_transformer_decoder_layers_per_scaling_block
        self.n_conv_layers_per_mid_block: int = n_conv_layers_per_mid_block
        self.n_transformer_encoder_layers_per_mid_block: int = n_transformer_encoder_layers_per_mid_block
        self.n_transformer_decoder_layers_per_mid_block: int = n_transformer_decoder_layers_per_mid_block
        self.n_attention_heads: int = n_attention_heads
        self.switch_ratio: float = switch_ratio

        assert len(down_transformer_model_dims) == len(down_out_dims) == len(up_transformer_model_dims) == len(up_out_dims)
        assert len(down_out_dims) >= 1
        assert len(mid_transformer_model_dims) == len(mid_out_dims)

        self.n_scaling_blocks: int = len(down_out_dims)
        self.n_mid_blocks: int = len(mid_out_dims)

        self.down_blocks = nn.ModuleList([
            _DownBlock(
                input_dim=target_dim if i == 0 else down_out_dims[i - 1], 
                output_dim=down_out_dims[i],
                condition_dim=condition_dim, 
                n_condition_days=n_condition_days,
                n_conv_layers=n_conv_layers_per_scaling_block, 
                in_H=in_H // (2 ** i), in_W=in_W // (2 ** i),
                transformer_model_dim=down_transformer_model_dims[i],
                transformer_feedforward_dim=transformer_feedforward_dim,
                n_transformer_encoder_layers=n_transformer_encoder_layers_per_scaling_block,
                n_transformer_decoder_layers=n_transformer_decoder_layers_per_scaling_block,
                n_attention_heads=n_attention_heads,
            )
            for i in range(self.n_scaling_blocks)
        ])
        self.up_blocks = nn.ModuleList([
            _UpBlock(
                input_dim=mid_out_dims[-1] if i == 0 else up_out_dims[i - 1], 
                output_dim=up_out_dims[i], 
                down_output_dim=down_out_dims[-i - 1],
                condition_dim=condition_dim, 
                n_condition_days=n_condition_days,
                n_conv_layers=n_conv_layers_per_scaling_block, 
                in_H=in_H // (2 ** (self.n_scaling_blocks - i)), in_W=in_W // (2 ** (self.n_scaling_blocks - i)),
                transformer_model_dim=up_transformer_model_dims[i], 
                transformer_feedforward_dim=transformer_feedforward_dim,
                n_transformer_encoder_layers=n_transformer_encoder_layers_per_scaling_block,
                n_transformer_decoder_layers=n_transformer_decoder_layers_per_scaling_block,
                n_attention_heads=n_attention_heads, 
            )
            for i in range(self.n_scaling_blocks)
        ])
        self.mid_blocks = nn.ModuleList([
            _MidBlock(
                input_dim=down_out_dims[-1] if i == 0 else mid_out_dims[i - 1],
                output_dim=mid_out_dims[i],
                condition_dim=condition_dim, 
                n_condition_days=n_condition_days,
                n_conv_layers=n_conv_layers_per_mid_block, 
                in_H=in_H // (2 **self.n_scaling_blocks), in_W=in_W // (2 ** self.n_scaling_blocks),
                transformer_model_dim=mid_transformer_model_dims[i], 
                transformer_feedforward_dim=transformer_feedforward_dim,
                n_transformer_encoder_layers=n_transformer_encoder_layers_per_mid_block,
                n_transformer_decoder_layers=n_transformer_decoder_layers_per_mid_block,
                n_attention_heads=n_attention_heads,
            )
            for i in range(self.n_mid_blocks)
        ])
        self.head: nn.Module = nn.Conv2d(
            in_channels=self.up_out_dims[-1], out_channels=target_dim, kernel_size=1,
        )

    def forward(
        self, 
        target: torch.Tensor, 
        condition_mu: torch.Tensor, condition_logvar: torch.Tensor,
        integer_step: torch.Tensor, condition_days: torch.Tensor,
    ) -> torch.Tensor:
        target_N, target_D, target_H, target_W = target.shape
        condition_N, condition_L, condition_D = condition_mu.shape
        assert condition_mu.shape == condition_logvar.shape
        step_N, step_L = integer_step.shape
        day_N, day_L = condition_days.shape
        assert target_N == condition_N == step_N == day_N
        assert target_D == self.target_dim
        assert step_L == 1
        
        # Check if too many scaling blocks
        _min_dim: float = min(target_H, target_W) / (2 ** self.n_scaling_blocks)
        assert _min_dim % 2 == 0, (
            f"too many scaling blocks (self.n_scaling_blocks={self.n_scaling_blocks}) "
            f"for target dimension: {(target_H, target_W)}"
        )
        # Switch condition on/off randomly during training
        if self.training:
            switch: torch.Tensor = torch.rand(size=(target_N,), device=target.device) > self.switch_ratio
            switch = switch[:, None, None].float()
            condition_mu = condition_mu * switch
            condition_logvar = condition_logvar * switch

        # UNet
        down_input: torch.Tensor = target
        down_outputs: list[torch.Tensor] = []
        for i in range(self.n_scaling_blocks):
            down_outputs.append(
                self.down_blocks[i](
                    target=down_input,
                    condition_mu=condition_mu, condition_logvar=condition_logvar, 
                    step=integer_step, days=condition_days,
                )
            )
            down_input = down_outputs[-1]
        
        mid_output: torch.Tensor = down_outputs[-1]
        for i in range(self.n_mid_blocks):
            mid_output = self.mid_blocks[i](
                target=mid_output,
                condition_mu=condition_mu, condition_logvar=condition_logvar, 
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
                condition_mu=condition_mu, condition_logvar=condition_logvar, 
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
