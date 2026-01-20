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
        assert embedding_dim % 2 == 0

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


class _LinearActLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        if in_features == out_features:
            self.layer0: nn.Module = nn.Identity()
        else:
            self.layer0: nn.Module = nn.Linear(in_features=in_features, out_features=out_features)
        
        self.activation: nn.Module = nn.ReLU()
        self.layer1: nn.Module = nn.Linear(in_features=out_features, out_features=out_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        N, T, E = input.shape
        assert E == self.in_features
        output: torch.Tensor = self.layer1(self.activation(self.layer0(input)))
        assert output.shape == (N, T, self.out_features)
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

        self.skips = nn.ModuleList(
            [nn.Linear(in_features=kv_dim, out_features=model_dim)]
            + [nn.Identity() for i in range(n_layers - 1)]
        )
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
        for skip, layer in zip(self.skips, self.layers):
            kv = skip(kv) + layer(kv)

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
        
        self.skips = nn.ModuleList(
            [nn.Linear(in_features=q_dim, out_features=model_dim)]
            + [nn.Identity() for i in range(n_layers - 1)]
        )
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
        for skip, layer in zip(self.skips, self.layers):
            q = skip(q) + layer(q=q, kv=kv)
        
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
        assert src_L <= self.maxlength and tgt_L <= self.maxlength
        src = src + self.kv_pos_embedding[:, :src_L, :]
        tgt = tgt + self.q_pos_embedding[:, :tgt_L, :]
        memory: torch.Tensor = self.encoder(kv=src)
        output: torch.Tensor = self.decoder(q=tgt, kv=memory)
        assert output.shape == (tgt_N, tgt_L, tgt_D)
        return output


class _ScalingBlock(nn.Module):

    def __init__(
        self,
        input_dim: int, 
        output_dim: int, 
        condition_dim: int, 
        n_nonlinear_layers: int, 
        transformer_model_dim: int, 
        transformer_feedforward_dim: int, 
        n_transformer_encoder_layers: int, 
        n_transformer_decoder_layers: int,
        n_attention_heads: int,
        transformer_maxlength: int,
    ):
        super().__init__()
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.condition_dim: int = condition_dim
        self.transformer_model_dim: int = transformer_model_dim
        self.transformer_feedforward_dim: int = transformer_feedforward_dim
        self.n_nonlinear_layers: int = n_nonlinear_layers
        self.n_transformer_encoder_layers: int = n_transformer_encoder_layers
        self.n_transformer_decoder_layers: int = n_transformer_decoder_layers
        self.n_attention_heads: int = n_attention_heads
        self.transformer_maxlength: int = transformer_maxlength

        # Condition
        self.condition_fuser: nn.Module = _ConditionFuser(condition_dim=condition_dim)
        # Target
        self.target_nonlinear_layers: nn.ModuleList = nn.ModuleList([
            _LinearActLinear(in_features=input_dim if i == 0 else output_dim, out_features=output_dim)
            for i in range(n_nonlinear_layers)
        ])
        self.target_residual_layers: nn.ModuleList = nn.ModuleList([
            nn.Linear(in_features=input_dim if i == 0 else output_dim, out_features=output_dim)
            for i in range(n_nonlinear_layers)
        ])
        # Attention
        self.transfomer: nn.Module = _Transformer(
            q_dim=output_dim,
            kv_dim=condition_dim,
            model_dim=transformer_model_dim,
            n_heads=n_attention_heads,
            feedforward_dim=transformer_feedforward_dim,
            n_encoder_layers=n_transformer_encoder_layers,
            n_decoder_layers=n_transformer_decoder_layers,
            maxlength=transformer_maxlength,
        )
        # Step
        self.step_embedding_layer: nn.Module = _SinusoidEmbedding(embedding_dim=output_dim)
        # Day
        self.day_embedding_layer: nn.Module = _SinusoidEmbedding(embedding_dim=output_dim)
        self.cond_weights: nn.Parameter = nn.Parameter(torch.randn(size=(1, transformer_maxlength, output_dim)))

    def forward(
        self, 
        target: torch.Tensor, 
        condition_mu: torch.Tensor, condition_logvar: torch.Tensor,
        step: torch.Tensor, days: torch.Tensor
    ) -> torch.Tensor:
        assert condition_mu.shape == condition_logvar.shape
        condition_N, condition_L, condition_D = condition_mu.shape
        assert condition_D == self.condition_dim
        target_N, target_L, target_D = target.shape
        assert target_D == self.input_dim
        step_N, step_L = step.shape
        days_N, days_L = days.shape
        assert target_N == condition_N == step_N == days_N
        assert step_L == 1
        
        condition: torch.Tensor = self.condition_fuser(
            condition_mu=condition_mu, condition_logvar=condition_logvar
        )
        step = self.step_embedding_layer(step).mean(dim=1, keepdim=True)
        days = self.day_embedding_layer(days).mean(dim=1, keepdim=True)
        assert step.shape == (step_N, 1, self.output_dim)
        assert days.shape == (days_N, 1, self.output_dim)

        for i in range(self.n_nonlinear_layers):
            target = self.target_residual_layers[i](target) + self.target_nonlinear_layers[i](target) + step + days

        assert target.shape == (target_N, target_L, self.output_dim)

        # Conditioning
        conditioned_target: torch.Tensor = self.transfomer(src=condition, tgt=target)
        assert conditioned_target.shape == target.shape
        weight: torch.Tensor = torch.sigmoid(self.cond_weights[:, :target_L, :])
        output: torch.Tensor = weight * conditioned_target + (1 - weight) * target
        assert output.shape == (target_N, target_L, self.output_dim)
        return output


class UNetDenoiser(NamedModel, nn.Module):

    def __init__(
        self,
        target_dim: int, 
        condition_dim: int, 
        target_H: int, target_W: int,
        n_scaling_blocks: int, n_mid_blocks: int,
        scale_factor: int,
        n_nonlinear_layers_per_scaling_block: int, 
        n_nonlinear_layers_per_mid_block: int, 
        transformer_dim: int,
        transformer_feedforward_dim: int,
        n_transformer_encoder_layers_per_scaling_block: int, 
        n_transformer_decoder_layers_per_scaling_block: int, 
        n_transformer_encoder_layers_per_mid_block: int,
        n_transformer_decoder_layers_per_mid_block: int,
        n_attention_heads: int,
        transformer_maxlength: int,
        switch_ratio: float,
    ):
        super().__init__()
        self.target_dim: int = target_dim
        self.condition_dim: int = condition_dim
        self.target_H: int = target_H
        self.target_W: int = target_W
        self.n_scaling_blocks: int = n_scaling_blocks
        self.n_mid_blocks: int = n_mid_blocks
        self.scale_factor: int = scale_factor
        self.n_nonlinear_layers_per_scaling_block: int = n_nonlinear_layers_per_scaling_block
        self.n_nonlinear_layers_per_mid_block: int = n_nonlinear_layers_per_mid_block
        self.transformer_dim: int = transformer_dim
        self.transformer_feedforward_dim: int = transformer_feedforward_dim
        self.n_transformer_encoder_layers_per_scaling_block: int = n_transformer_encoder_layers_per_scaling_block
        self.n_transformer_decoder_layers_per_scaling_block: int = n_transformer_decoder_layers_per_scaling_block
        self.n_transformer_encoder_layers_per_mid_block: int = n_transformer_encoder_layers_per_mid_block
        self.n_transformer_decoder_layers_per_mid_block: int = n_transformer_decoder_layers_per_mid_block
        self.n_attention_heads: int = n_attention_heads
        self.transformer_maxlength: int = transformer_maxlength
        self.switch_ratio: float = switch_ratio

        assert n_scaling_blocks > 0 and n_mid_blocks > 0

        self.down_dims: list[int] = [target_dim // (scale_factor ** i) for i in range(self.n_scaling_blocks + 1)]
        self.up_dims: list[int] = self.down_dims[::-1]
        self.mid_dims: list[int] = [self.down_dims[-1] for _ in range(self.n_mid_blocks + 1)]

        self.down_blocks = nn.ModuleList([
            _ScalingBlock(
                input_dim=self.down_dims[i],
                output_dim=self.down_dims[i + 1],
                condition_dim=condition_dim, 
                n_nonlinear_layers=n_nonlinear_layers_per_scaling_block, 
                transformer_model_dim=transformer_dim,
                transformer_feedforward_dim=transformer_feedforward_dim,
                n_transformer_encoder_layers=n_transformer_encoder_layers_per_scaling_block,
                n_transformer_decoder_layers=n_transformer_decoder_layers_per_scaling_block,
                n_attention_heads=n_attention_heads,
                transformer_maxlength=transformer_maxlength,
            )
            for i in range(self.n_scaling_blocks)
        ])
        self.up_blocks = nn.ModuleList([
            _ScalingBlock(
                input_dim=self.up_dims[i], 
                output_dim=self.up_dims[i + 1], 
                condition_dim=condition_dim, 
                n_nonlinear_layers=n_nonlinear_layers_per_scaling_block, 
                transformer_model_dim=transformer_dim, 
                transformer_feedforward_dim=transformer_feedforward_dim,
                n_transformer_encoder_layers=n_transformer_encoder_layers_per_scaling_block,
                n_transformer_decoder_layers=n_transformer_decoder_layers_per_scaling_block,
                n_attention_heads=n_attention_heads, 
                transformer_maxlength=transformer_maxlength,
            )
            for i in range(self.n_scaling_blocks)
        ])
        self.mid_blocks = nn.ModuleList([
            _ScalingBlock(
                input_dim=self.mid_dims[i], 
                output_dim=self.mid_dims[i + 1], 
                condition_dim=condition_dim, 
                n_nonlinear_layers=n_nonlinear_layers_per_mid_block, 
                transformer_model_dim=transformer_dim, 
                transformer_feedforward_dim=transformer_feedforward_dim,
                n_transformer_encoder_layers=n_transformer_encoder_layers_per_mid_block,
                n_transformer_decoder_layers=n_transformer_decoder_layers_per_mid_block,
                n_attention_heads=n_attention_heads, 
                transformer_maxlength=transformer_maxlength,
            )
            for i in range(self.n_mid_blocks)
        ])
        self.head: nn.Module = nn.Linear(in_features=target_dim, out_features=target_dim)

    def forward(
        self, 
        target: torch.Tensor, 
        condition_mu: torch.Tensor, condition_logvar: torch.Tensor,
        integer_step: torch.Tensor, condition_days: torch.Tensor,
    ) -> torch.Tensor:
        target_N, target_L, target_D = target.shape
        condition_N, condition_L, condition_D = condition_mu.shape
        assert condition_mu.shape == condition_logvar.shape
        step_N, step_L = integer_step.shape
        day_N, day_L = condition_days.shape
        assert target_N == condition_N == step_N == day_N
        assert target_D == self.target_dim
        assert condition_D == self.condition_dim
        assert step_L == 1
        
        # Switch condition on/off randomly during training
        if self.training:
            switch: torch.Tensor = torch.rand(size=(target_N,), device=target.device) > self.switch_ratio
            condition_mu = condition_mu * switch[:, None, None].float()
            condition_logvar = condition_logvar * switch[:, None, None].float()
            condition_days = condition_days * switch[:, None].float()

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
            assert skip.shape == up_output.shape, (
                f"skip.shape={skip.shape}, up_output.shape={up_output.shape}"
            )
            up_output = self.up_blocks[i](
                target=up_output + skip,
                condition_mu=condition_mu, condition_logvar=condition_logvar, 
                step=integer_step, days=condition_days,
            )

        assert len(down_outputs) == 0, f"down_outputs must exhaust, getting {len(down_outputs)} items left"
        output: torch.Tensor = self.head(up_output)
        assert output.shape == target.shape == (target_N, target_L, target_D)
        return output


class _DiffusionProcess:

    def __init__(self, noise_scheduler: LinearNoiseScheduler | CosineNoiseScheduler):
        self.noise_scheduler: LinearNoiseScheduler | CosineNoiseScheduler = noise_scheduler
        self.alpha_bar_schedule: torch.Tensor = self.noise_scheduler.alpha_bar_schedule
        self.beta_schedule: torch.Tensor = self.noise_scheduler.beta_schedule

    # forward & reverse process
    def compute_alpha_bar(self, step: torch.Tensor) -> torch.Tensor:
        step_N, _ = step.shape  # (batch_size, 1)
        # alpha ranges from k=0,...,K
        assert torch.all(step.long() >= 0)
        assert torch.all(step.long() <= self.noise_scheduler.n_steps)
        alpha_bar: torch.Tensor = self.alpha_bar_schedule.to(step.device)[step.long()]
        assert alpha_bar.shape == (step_N, 1)
        alpha_bar = alpha_bar[:, :, None]
        return alpha_bar

    def compute_beta(self, step: torch.Tensor) -> torch.Tensor:
        step_N, _ = step.shape  # (batch_size, 1)
        # beta ranges from k=1,...,K
        assert torch.all(step.long() - 1 >= 0)
        assert torch.all(step.long() - 1 <= self.noise_scheduler.n_steps - 1)
        beta: torch.Tensor = self.beta_schedule.to(step.device)[step.long() - 1]
        assert beta.shape == (step_N, 1)
        beta = beta[:, :, None]
        return beta
    
    # reverse process
    def compute_tilde_beta(self, alpha_bar_prev: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
        return (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)

    # reverse process
    def compute_sigma(self, tilde_beta: torch.Tensor, eta: float) -> torch.Tensor:
        return eta * torch.sqrt(tilde_beta)
    
    # forward process
    def compute_velocity(self, alpha_bar: torch.Tensor, target_0: torch.Tensor, gaussian: torch.Tensor) -> torch.Tensor:
        return - torch.sqrt(1 - alpha_bar) * target_0 + torch.sqrt(alpha_bar) * gaussian

    # reverse process
    def compute_x0(self, target_k: torch.Tensor, predicted_velocity: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
        target_N, target_T, target_D = target_k.shape
        velocity_N, velocity_T, velocity_D = predicted_velocity.shape
        alpha_bar_N, _, _ = alpha_bar.shape  # (batch_size, 1, 1)
        assert target_k.shape == predicted_velocity.shape
        assert target_N == velocity_N == alpha_bar_N
        # Original target prediction at k
        target_0: torch.Tensor = torch.sqrt(alpha_bar) * target_k - torch.sqrt(1 - alpha_bar) * predicted_velocity
        assert target_0.shape == target_k.shape
        return target_0

    # reverse process
    def compute_gaussian(self, target_k: torch.Tensor, predicted_velocity: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
        target_N, target_T, target_D = target_k.shape
        velocity_N, velocity_T, velocity_D = predicted_velocity.shape
        alpha_bar_N, _, _ = alpha_bar.shape  # (batch_size, 1)
        assert target_k.shape == predicted_velocity.shape
        assert target_N == velocity_N == alpha_bar_N
        # Original target prediction at k
        gaussian: torch.Tensor = torch.sqrt(1 - alpha_bar) * target_k + torch.sqrt(alpha_bar) * predicted_velocity
        assert gaussian.shape == target_k.shape
        return gaussian


class ForwardProcess(_DiffusionProcess):

    def add_noise(self, original_latent: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_N: int = original_latent.shape[0]
        step_N, _ = k.shape  # (batch_size, 1)
        assert original_N == step_N   # batch_size
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
        
        target_N, target_T, target_D, target_H, target_W = target_k.shape
        velocity_N, velocity_T, velocity_D, velocity_H, velocity_W = predicted_velocity.shape
        step_N, _ = k.shape  # (batch_size, 1)
        assert target_k.shape == predicted_velocity.shape
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
