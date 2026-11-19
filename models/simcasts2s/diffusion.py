from typing import Any, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import NamedModel
from models.adaptation.lora import LoRAConv2d, LoRALinear, LoRATConv2d


class CosineNoiseScheduler(nn.Module):

    alpha_bar_schedule: torch.Tensor
    beta_schedule: torch.Tensor

    def __init__(self, n_steps: int, cosine_offset: float = 0.008) -> None:
        super().__init__()
        if type(n_steps) is not int or n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        assert cosine_offset >= 0, "cosine_offset must be non-negative"
        self.n_steps = n_steps
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

    def to_config(self) -> dict[str, Any]:
        return {
            "train_steps": self.n_steps,
            "cosine_offset": self.cosine_offset,
        }


def build_noise_scheduler(config: dict[str, Any]) -> CosineNoiseScheduler:
    """Build the cosine schedule from configuration or saved schedule metadata."""
    return CosineNoiseScheduler(
        n_steps=config["train_steps"], cosine_offset=config["cosine_offset"],
    )


class _Finetunable:

    down_blocks: nn.ModuleList
    mid_blocks: nn.ModuleList
    up_blocks: nn.ModuleList
    head: nn.Module

    def _adaptation_parameter_ids(self) -> set[int]:
        """Parameters trained during LoRA adaptation."""
        self._validate_architecture()
        assert isinstance(self, nn.Module)

        parameter_ids = {
            id(param)
            for name, param in self.named_parameters()
            if "lora_A" in name or "lora_B" in name
        }
        for module in self.modules():
            if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                parameter_ids.update(
                    id(param) for param in module.parameters(recurse=False)
                )
        parameter_ids.update(id(param) for param in self.head.parameters())
        return parameter_ids

    def is_backbone_frozen(self) -> bool:
        assert isinstance(self, nn.Module)
        adaptation_parameter_ids = self._adaptation_parameter_ids()
        return all(
            param.requires_grad == (id(param) in adaptation_parameter_ids)
            for param in self.parameters()
        )

    def freeze_backbone(self) -> None:
        assert isinstance(self, nn.Module)
        adaptation_parameter_ids = self._adaptation_parameter_ids()
        for param in self.parameters():
            param.requires_grad = id(param) in adaptation_parameter_ids
        print(
            f"{self.name} backbone has been frozen; LoRA adapters, "
            "normalization affine parameters, and velocity head remain trainable"
        )

    def unfreeze_all(self) -> None:
        assert isinstance(self, nn.Module)
        for param in self.parameters():
            param.requires_grad = True
        print(f"{self.name} has been unfrozen")

    def _validate_architecture(self) -> None:
        assert hasattr(self, "down_blocks")
        assert hasattr(self, "mid_blocks")
        assert hasattr(self, "up_blocks")
        assert hasattr(self, "head")
        assert isinstance(self.down_blocks, nn.Module)
        assert isinstance(self.mid_blocks, nn.Module)
        assert isinstance(self.up_blocks, nn.Module)
        assert isinstance(self.head, nn.Module)

class _SinusoidEmbedding(nn.Module):

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim: int = embedding_dim
        assert embedding_dim % 2 == 0

        # Frequency scaling
        w = 1.0 / torch.pow(
            input=torch.tensor(10_000.0, dtype=torch.float),
            exponent=torch.arange(0, embedding_dim, 2, dtype=torch.float) / embedding_dim,
        )
        self.register_buffer("w", w)
        self.w: torch.Tensor
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


def _group_count(channels: int, maximum: int = 32) -> int:
    """Return the largest useful GroupNorm group count for ``channels``."""
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    raise RuntimeError(f"Unable to choose GroupNorm groups for {channels=}")


class _ResidualConvBlock(nn.Module):
    """Normalized residual block with diffusion-step FiLM conditioning."""

    def __init__(self, input_dim: int, output_dim: int, time_embedding_dim: int) -> None:
        super().__init__()
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.time_embedding_dim: int = time_embedding_dim

        self.norm1 = nn.GroupNorm(
            num_groups=_group_count(input_dim), num_channels=input_dim,
        )
        self.conv1 = nn.Conv2d(
            in_channels=input_dim, out_channels=output_dim,
            kernel_size=3, padding=1, padding_mode="reflect",
        )
        self.norm2 = nn.GroupNorm(
            num_groups=_group_count(output_dim), num_channels=output_dim,
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, 2 * output_dim),
        )
        self.conv2 = nn.Conv2d(
            in_channels=output_dim, out_channels=output_dim,
            kernel_size=3, padding=1, padding_mode="reflect",
        )
        self.residual_projection: nn.Module
        if input_dim == output_dim:
            self.residual_projection = nn.Identity()
        else:
            self.residual_projection = nn.Conv2d(
                in_channels=input_dim, out_channels=output_dim, kernel_size=1,
            )

    def forward(self, input: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        N, T, D, H, W = input.shape
        assert D == self.input_dim
        assert time_embedding.shape == (N, self.time_embedding_dim)

        flat_input = input.flatten(0, 1)
        residual = self.residual_projection(flat_input)
        output = self.conv1(F.silu(self.norm1(flat_input)))
        output = self.norm2(output)

        scale, shift = self.time_projection(time_embedding).chunk(2, dim=1)
        scale = scale[:, None, :, None, None].expand(N, T, self.output_dim, 1, 1)
        shift = shift[:, None, :, None, None].expand(N, T, self.output_dim, 1, 1)
        output = output.reshape(N, T, self.output_dim, H, W)
        output = output * (1 + scale) + shift
        output = self.conv2(F.silu(output.flatten(0, 1)))
        output = output + residual
        assert output.shape == (N * T, self.output_dim, H, W)
        return output.reshape(N, T, self.output_dim, H, W)

    def enable_lora(self, rank: int) -> int:
        count = 0
        for i, layer in enumerate(self.time_projection):
            if type(layer) is nn.Linear:
                self.time_projection[i] = LoRALinear(base=layer, rank=rank)
                count += 1
        for name in ("conv1", "conv2", "residual_projection"):
            layer = getattr(self, name)
            if type(layer) is nn.Conv2d:
                setattr(self, name, LoRAConv2d(base=layer, rank=rank))
                count += 1
        return count


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

    def enable_lora(self, rank: int) -> int:
        count: int = 0
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                self.net[i] = LoRALinear(base=layer, rank=rank)
                count += 1
        return count


class _MultiheadAttention(nn.Module):

    def __init__(self, embed_dim: int, n_heads: int) -> None:
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError(f"embed_dim must be divisible by n_heads, got {embed_dim=} and {n_heads=}")
        self.embed_dim: int = embed_dim
        self.n_heads: int = n_heads
        self.head_dim: int = embed_dim // n_heads

        self.q_projection: nn.Module = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.k_projection: nn.Module = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.v_projection: nn.Module = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.out_projection: nn.Module = nn.Linear(in_features=embed_dim, out_features=embed_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, length, embed_dim = x.shape
        assert embed_dim == self.embed_dim
        x = x.reshape(batch_size, length, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_heads, length, head_dim = x.shape
        assert n_heads == self.n_heads
        assert head_dim == self.head_dim
        x = x.transpose(1, 2).contiguous()
        return x.reshape(batch_size, length, self.embed_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        q_N, q_L, q_D = query.shape
        k_N, k_L, k_D = key.shape
        v_N, v_L, v_D = value.shape
        assert q_N == k_N == v_N
        assert k_L == v_L
        assert q_D == k_D == v_D == self.embed_dim

        q: torch.Tensor = self._split_heads(self.q_projection(query))
        k: torch.Tensor = self._split_heads(self.k_projection(key))
        v: torch.Tensor = self._split_heads(self.v_projection(value))

        output: torch.Tensor = F.scaled_dot_product_attention(query=q, key=k, value=v)
        output = self._merge_heads(output)
        output = self.out_projection(output)
        assert output.shape == (q_N, q_L, self.embed_dim)
        return output

    def enable_lora(self, rank: int) -> int:
        count: int = 0
        for name in ("q_projection", "k_projection", "v_projection", "out_projection"):
            layer = getattr(self, name)
            if isinstance(layer, nn.Linear):
                setattr(self, name, LoRALinear(base=layer, rank=rank))
                count += 1
        return count


class _TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        kv_dim: int,
        model_dim: int,
        n_heads: int,
        feedforward_dim: int,
    ):
        super().__init__()
        self.kv_dim: int = kv_dim
        self.model_dim: int = model_dim
        self.n_heads: int = n_heads
        self.feedforward_dim: int = feedforward_dim

        if kv_dim != model_dim:
            self.kv_projection = nn.Linear(in_features=kv_dim, out_features=model_dim)
        else:
            self.kv_projection = nn.Identity()

        self.self_attention = _MultiheadAttention(embed_dim=model_dim, n_heads=n_heads)
        self.feed_forward = _TransformerFeedForward(model_dim=model_dim, feedforward_dim=feedforward_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, kv: torch.Tensor) -> torch.Tensor:
        N, L, D = kv.shape
        kv = self.kv_projection(kv)
        normalized = self.norm1(kv)
        kv = kv + self.self_attention(
            query=normalized, key=normalized, value=normalized,
        )
        kv = kv + self.feed_forward(self.norm2(kv))
        assert kv.shape == (N, L, self.model_dim)
        return kv

    def enable_lora(self, rank: int) -> int:
        count: int = 0
        if isinstance(self.kv_projection, nn.Linear):
            self.kv_projection = LoRALinear(base=self.kv_projection, rank=rank)
            count += 1
        count += self.self_attention.enable_lora(rank=rank)
        count += self.feed_forward.enable_lora(rank=rank)
        return count


class _TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        q_dim: int,
        model_dim: int,
        n_heads: int,
        feedforward_dim: int,
    ):
        super().__init__()
        self.q_dim: int = q_dim
        self.model_dim: int = model_dim
        self.n_heads: int = n_heads
        self.feedforward_dim: int = feedforward_dim

        if q_dim != model_dim:
            self.q_projection = nn.Linear(in_features=q_dim, out_features=model_dim)
        else:
            self.q_projection = nn.Identity()

        self.self_attention = _MultiheadAttention(embed_dim=model_dim, n_heads=n_heads)
        self.cross_attention = _MultiheadAttention(embed_dim=model_dim, n_heads=n_heads)
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
        normalized_q = self.norm1(q)
        q = q + self.self_attention(
            query=normalized_q, key=normalized_q, value=normalized_q,
        )
        q = q + self.cross_attention(
            query=self.norm2(q), key=kv, value=kv,
        )
        q = q + self.ffn(self.norm3(q))
        assert q.shape == (q_N, q_L, self.model_dim)
        return q

    def enable_lora(self, rank: int) -> int:
        count: int = 0
        if isinstance(self.q_projection, nn.Linear):
            self.q_projection = LoRALinear(base=self.q_projection, rank=rank)
            count += 1
        count += self.self_attention.enable_lora(rank=rank)
        count += self.cross_attention.enable_lora(rank=rank)
        count += self.ffn.enable_lora(rank=rank)
        return count


class _TransformerEncoder(nn.Module):

    def __init__(
        self,
        kv_dim: int,
        model_dim: int,
        n_heads: int,
        feedforward_dim: int,
        n_layers: int,
    ):
        super().__init__()
        self.kv_dim: int = kv_dim
        self.model_dim: int = model_dim
        self.n_heads: int = n_heads
        self.feedforward_dim: int = feedforward_dim
        self.n_layers: int = n_layers
        self.layers = nn.ModuleList([
            _TransformerEncoderLayer(
                kv_dim=kv_dim if i == 0 else model_dim,
                model_dim=model_dim, n_heads=n_heads, feedforward_dim=feedforward_dim,
            )
            for i in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(model_dim)

    def forward(self, kv: torch.Tensor) -> torch.Tensor:
        N, L, D = kv.shape
        assert D == self.kv_dim
        for layer in self.layers:
            kv = layer(kv)
        kv = self.final_norm(kv)

        assert kv.shape == (N, L, self.model_dim)
        return kv

    def enable_lora(self, rank: int) -> int:
        count: int = 0
        for layer in self.layers:
            layer = cast(_TransformerEncoderLayer, layer)
            count += layer.enable_lora(rank=rank)
        return count


class _TransformerDecoder(nn.Module):

    def __init__(
        self,
        q_dim: int,
        model_dim: int,
        n_heads: int,
        feedforward_dim: int,
        n_layers: int,
    ):
        super().__init__()
        self.q_dim: int = q_dim
        self.model_dim: int = model_dim
        self.n_heads: int = n_heads
        self.feedforward_dim: int = feedforward_dim
        self.n_layers: int = n_layers
        self.layers = nn.ModuleList([
            _TransformerDecoderLayer(
                q_dim=q_dim if i == 0 else model_dim,
                model_dim=model_dim, n_heads=n_heads, feedforward_dim=feedforward_dim,
            )
            for i in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(model_dim)
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
        residual = q
        for layer in self.layers:
            q = layer(q=q, kv=kv)
        q = residual + self.out_projection(self.final_norm(q))
        assert q.shape == (q_N, q_L, q_D)
        return q

    def enable_lora(self, rank: int) -> int:
        count: int = 0
        for layer in self.layers:
            layer = cast(_TransformerDecoderLayer, layer)
            count += layer.enable_lora(rank=rank)
        for i, layer in enumerate(self.out_projection):
            if isinstance(layer, nn.Linear):
                self.out_projection[i] = LoRALinear(base=layer, rank=rank)
                count += 1
        return count


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
            feedforward_dim=feedforward_dim, n_layers=n_decoder_layers,
        )
        self.kv_pos_embedding = nn.Parameter(torch.randn(1, maxlength, kv_dim) * 0.02)
        self.q_pos_embedding = nn.Parameter(torch.randn(1, maxlength, q_dim) * 0.02)
        self.kv_day_embedding = _SinusoidEmbedding(embedding_dim=kv_dim)
        self.q_day_embedding = _SinusoidEmbedding(embedding_dim=q_dim)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_days: torch.Tensor,
        tgt_days: torch.Tensor,
    ) -> torch.Tensor:
        src_N, src_L, src_D = src.shape
        tgt_N, tgt_L, tgt_D = tgt.shape
        src_day_N, src_day_L = src_days.shape
        tgt_day_N, tgt_day_L = tgt_days.shape
        assert src_D == self.kv_dim
        assert tgt_D == self.q_dim
        assert src_N == tgt_N == src_day_N == tgt_day_N
        assert src_L == src_day_L
        assert tgt_L == tgt_day_L
        assert src_L <= self.maxlength and tgt_L <= self.maxlength
        src = src + self.kv_pos_embedding[:, :src_L, :] + self.kv_day_embedding(src_days)
        tgt = tgt + self.q_pos_embedding[:, :tgt_L, :] + self.q_day_embedding(tgt_days)
        memory: torch.Tensor = self.encoder(kv=src)
        output: torch.Tensor = self.decoder(q=tgt, kv=memory)
        assert output.shape == (tgt_N, tgt_L, tgt_D)
        return output

    def enable_lora(self, rank: int) -> int:
        count: int = 0
        count += self.encoder.enable_lora(rank=rank)
        count += self.decoder.enable_lora(rank=rank)
        return count


class _DownSample(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        assert output_dim % 2 == 0

        # Downsampling using Conv
        self.conv2d: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1),
            nn.Conv2d(
                in_channels=input_dim, out_channels=output_dim // 2,
                kernel_size=4, stride=2, padding=1, padding_mode="reflect",
            ),
        )
        # Downsampling using MaxPool
        self.maxpool2d: nn.Module = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=input_dim, out_channels=output_dim // 2,
                kernel_size=1, stride=1, padding=0,
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        N, T, E, H, W = input.shape
        assert E == self.input_dim, f"input_dim={E} vs. self.input_dim={self.input_dim}"
        output: torch.Tensor = input.flatten(0, 1)
        output = torch.cat([self.conv2d(output), self.maxpool2d(output)], dim=1)
        assert output.shape == (N * T, self.output_dim, H // 2, W // 2)
        return output.reshape(N, T, self.output_dim, H // 2, W // 2)

    def enable_lora(self, rank: int) -> int:
        count = 0
        for sequence in (self.conv2d, self.maxpool2d):
            assert isinstance(sequence, nn.Sequential)
            for i, layer in enumerate(sequence):
                if type(layer) is nn.Conv2d:
                    sequence[i] = LoRAConv2d(base=layer, rank=rank)
                    count += 1
        return count


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
        N, T, E, H, W = input.shape
        assert E == self.input_dim, f"input_dim={E} vs. self.input_dim={self.input_dim}"
        output: torch.Tensor = input.flatten(0, 1)
        output = torch.cat([self.convtranspose2d(output), self.up2d(output)], dim=1)
        assert output.shape == (N * T, self.output_dim, H * 2, W * 2)
        return output.reshape(N, T, self.output_dim, H * 2, W * 2)

    def enable_lora(self, rank: int) -> int:
        count = 0
        for sequence in (self.convtranspose2d, self.up2d):
            assert isinstance(sequence, nn.Sequential)
            for i, layer in enumerate(sequence):
                if type(layer) is nn.Conv2d:
                    sequence[i] = LoRAConv2d(base=layer, rank=rank)
                    count += 1
                elif type(layer) is nn.ConvTranspose2d:
                    sequence[i] = LoRATConv2d(layer, rank=rank)
                    count += 1
        return count


class _MidSample(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.mid: nn.Module = nn.Conv2d(
            in_channels=input_dim, out_channels=output_dim, kernel_size=1, stride=1, padding=0
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        N, T, E, H, W = input.shape
        assert E == self.input_dim, f"input_dim={E} vs. self.input_dim={self.input_dim}"
        output: torch.Tensor = input.flatten(0, 1)
        output = self.mid(output)
        assert output.shape == (N * T, self.output_dim, H, W)
        return output.reshape(N, T, self.output_dim, H, W)


class _SpatialContext(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim: int = dim
        self.gate: nn.Module = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        N, T, D, H, W = input.shape
        assert D == self.dim
        output: torch.Tensor = input.flatten(0, 1)
        output = output + output * self.gate(output)
        assert output.shape == (N * T, D, H, W)
        return output.reshape(N, T, D, H, W)


class _ScalingBlock(nn.Module):

    """
    Project (linearly) target, condition, step, and day embbeding into a common hidden_dim
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        condition_dim: int,
        n_conv_layers: int,
        in_H: int, in_W: int,
        transformer_model_dim: int,
        transformer_feedforward_dim: int,
        n_transformer_encoder_layers: int,
        n_transformer_decoder_layers: int,
        n_attention_heads: int,
        transformer_maxlength: int,
        kind: Literal["up", "down", "mid"],
        time_embedding_dim: int,
    ):
        super().__init__()
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.condition_dim: int = condition_dim
        self.transformer_model_dim: int = transformer_model_dim
        self.transformer_feedforward_dim: int = transformer_feedforward_dim
        self.n_conv_layers: int = n_conv_layers
        self.in_H: int = in_H
        self.in_W: int = in_W
        self.n_transformer_encoder_layers: int = n_transformer_encoder_layers
        self.n_transformer_decoder_layers: int = n_transformer_decoder_layers
        self.n_attention_heads: int = n_attention_heads
        self.transformer_maxlength: int = transformer_maxlength
        self.kind: Literal["up", "down", "mid"] = kind
        self.time_embedding_dim: int = time_embedding_dim

        # Target
        self.target_residual_blocks: nn.ModuleList = nn.ModuleList([
            _ResidualConvBlock(
                input_dim=input_dim if i == 0 else output_dim,
                output_dim=output_dim,
                time_embedding_dim=time_embedding_dim,
            )
            for i in range(n_conv_layers)
        ])
        # Attention
        self.transfomer = _Transformer(
            q_dim=output_dim,
            kv_dim=condition_dim,
            model_dim=transformer_model_dim,
            n_heads=n_attention_heads,
            feedforward_dim=transformer_feedforward_dim,
            n_encoder_layers=n_transformer_encoder_layers,
            n_decoder_layers=n_transformer_decoder_layers,
            maxlength=transformer_maxlength,
        )
        self.spatial_context: nn.Module = _SpatialContext(dim=output_dim) if kind == "mid" else nn.Identity()
        # Scaling block
        self.scaling_block: nn.Module
        self.out_H: int
        self.out_W: int
        if kind == "down":
            self.scaling_block = _DownSample(input_dim=output_dim, output_dim=output_dim)
            self.out_H = self.in_H // 2
            self.out_W = self.in_W // 2
        elif kind == "up":
            self.scaling_block = _UpSample(input_dim=output_dim, output_dim=output_dim)
            self.out_H = self.in_H * 2
            self.out_W = self.in_W * 2
        elif kind == "mid":
            self.scaling_block = _MidSample(input_dim=output_dim, output_dim=output_dim)
            self.out_H = self.in_H
            self.out_W = self.in_W
        else:
            raise ValueError("Invalid type for _ScalingBlock, must be either 'up' or 'down' or 'mid'")

    def forward(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
        step: torch.Tensor,
        condition_days: torch.Tensor,
        target_days: torch.Tensor,
    ) -> torch.Tensor:
        condition_N, condition_L, condition_D, condition_H, condition_W = condition.shape
        assert condition_D == self.condition_dim
        target_N, target_T, target_D, target_H, target_W = target.shape
        assert target_D == self.input_dim
        assert (target_H, target_W) == (self.in_H, self.in_W)
        step_N, step_D = step.shape
        condition_days_N, condition_days_L = condition_days.shape
        target_days_N, target_days_L = target_days.shape
        assert target_N == condition_N == step_N == condition_days_N == target_days_N
        assert condition_days_L == condition_L
        assert target_days_L == target_T
        if (condition_H, condition_W) != (target_H, target_W):
            condition_flat: torch.Tensor = condition.flatten(0, 1)
            condition_flat = F.interpolate(condition_flat, size=(target_H, target_W), mode="bilinear")
            condition = condition_flat.reshape(condition_N, condition_L, self.condition_dim, target_H, target_W)

        assert step_D == self.time_embedding_dim
        for block in self.target_residual_blocks:
            target = block(target, step)

        assert target.shape == (target_N, target_T, self.output_dim, target_H, target_W)

        # Conditioning
        # Time-only transformer: each spatial location has its own sequence.
        target = target.permute(0, 3, 4, 1, 2).contiguous()
        target = target.reshape(target_N * target_H * target_W, target_T, self.output_dim)
        condition = condition.permute(0, 3, 4, 1, 2).contiguous()
        condition = condition.reshape(target_N * target_H * target_W, condition_L, self.condition_dim)
        expanded_condition_days = condition_days[:, None, None, :].expand(
            condition_N, target_H, target_W, condition_L
        ).reshape(condition_N * target_H * target_W, condition_L)
        expanded_target_days = target_days[:, None, None, :].expand(
            target_N, target_H, target_W, target_T
        ).reshape(target_N * target_H * target_W, target_T)
        conditioned_target: torch.Tensor = self.transfomer(
            src=condition, tgt=target,
            src_days=expanded_condition_days, tgt_days=expanded_target_days,
        )
        assert conditioned_target.shape == target.shape == (target_N * target_H * target_W, target_T, self.output_dim)
        output = conditioned_target
        assert output.shape == target.shape
        output = output.reshape(target_N, target_H, target_W, target_T, self.output_dim)
        output = output.permute(0, 3, 4, 1, 2).contiguous()
        output = self.spatial_context(output)

        # Scaling
        output = self.scaling_block(output)
        return output.reshape(target_N, target_T, self.output_dim, self.out_H, self.out_W)

    def enable_lora(self, rank: int) -> int:
        count = self.transfomer.enable_lora(rank=rank)
        for module in self.target_residual_blocks:
            block = cast(_ResidualConvBlock, module)
            count += block.enable_lora(rank=rank)
        if isinstance(self.scaling_block, (_DownSample, _UpSample)):
            count += self.scaling_block.enable_lora(rank=rank)
        return count


class _DownBlock(_ScalingBlock):


    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        condition_dim: int,
        n_conv_layers: int,
        in_H: int, in_W: int,
        transformer_model_dim: int,
        transformer_feedforward_dim: int,
        n_transformer_encoder_layers: int,
        n_transformer_decoder_layers: int,
        n_attention_heads: int,
        transformer_maxlength: int,
        time_embedding_dim: int,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            condition_dim=condition_dim,
            n_conv_layers=n_conv_layers,
            in_H=in_H, in_W=in_W,
            transformer_model_dim=transformer_model_dim,
            transformer_feedforward_dim=transformer_feedforward_dim,
            n_transformer_encoder_layers=n_transformer_encoder_layers,
            n_transformer_decoder_layers=n_transformer_decoder_layers,
            n_attention_heads=n_attention_heads,
            transformer_maxlength=transformer_maxlength,
            kind="down",
            time_embedding_dim=time_embedding_dim,
        )


class _UpBlock(_ScalingBlock):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        down_output_dim: int,
        condition_dim: int,
        n_conv_layers: int,
        in_H: int, in_W: int,
        transformer_model_dim: int,
        transformer_feedforward_dim: int,
        n_transformer_encoder_layers: int,
        n_transformer_decoder_layers: int,
        n_attention_heads: int,
        transformer_maxlength: int,
        time_embedding_dim: int,
    ):
        super().__init__(
            input_dim=input_dim + down_output_dim,
            output_dim=output_dim,
            condition_dim=condition_dim,
            n_conv_layers=n_conv_layers,
            in_H=in_H, in_W=in_W,
            transformer_model_dim=transformer_model_dim,
            transformer_feedforward_dim=transformer_feedforward_dim,
            n_transformer_encoder_layers=n_transformer_encoder_layers,
            n_transformer_decoder_layers=n_transformer_decoder_layers,
            n_attention_heads=n_attention_heads,
            transformer_maxlength=transformer_maxlength,
            kind="up",
            time_embedding_dim=time_embedding_dim,
        )


class _MidBlock(_ScalingBlock):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        condition_dim: int,
        n_conv_layers: int,
        in_H: int, in_W: int,
        transformer_model_dim: int,
        transformer_feedforward_dim: int,
        n_transformer_encoder_layers: int,
        n_transformer_decoder_layers: int,
        n_attention_heads: int,
        transformer_maxlength: int,
        time_embedding_dim: int,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            condition_dim=condition_dim,
            n_conv_layers=n_conv_layers,
            in_H=in_H, in_W=in_W,
            transformer_model_dim=transformer_model_dim,
            transformer_feedforward_dim=transformer_feedforward_dim,
            n_transformer_encoder_layers=n_transformer_encoder_layers,
            n_transformer_decoder_layers=n_transformer_decoder_layers,
            n_attention_heads=n_attention_heads,
            transformer_maxlength=transformer_maxlength,
            kind="mid",
            time_embedding_dim=time_embedding_dim,
        )


class Denoiser(_Finetunable, NamedModel, nn.Module):

    def __init__(
        self,
        target_dim: int,
        condition_dim: int,
        in_H: int, in_W: int,
        down_out_dims: list[int],
        mid_out_dims: list[int],
        up_out_dims: list[int],
        down_transformer_model_dims: list[int],
        mid_transformer_model_dims: list[int],
        up_transformer_model_dims: list[int],
        transformer_feedforward_dim: int,
        n_conv_layers_per_scaling_block: int,
        n_transformer_encoder_layers_per_scaling_block: list[int],
        n_transformer_decoder_layers_per_scaling_block: list[int],
        n_conv_layers_per_mid_block: int,
        n_transformer_encoder_layers_per_mid_block: list[int],
        n_transformer_decoder_layers_per_mid_block: list[int],
        n_attention_heads: int,
        transformer_maxlength: int,
        is_finetuning: bool,
        lora_rank: int = 0,
    ):
        super().__init__()
        self.target_dim: int = target_dim
        self.condition_dim: int = condition_dim
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
        self.n_transformer_encoder_layers_per_scaling_block: list[int] = n_transformer_encoder_layers_per_scaling_block
        self.n_transformer_decoder_layers_per_scaling_block: list[int] = n_transformer_decoder_layers_per_scaling_block
        self.n_conv_layers_per_mid_block: int = n_conv_layers_per_mid_block
        self.n_transformer_encoder_layers_per_mid_block: list[int] = n_transformer_encoder_layers_per_mid_block
        self.n_transformer_decoder_layers_per_mid_block: list[int] = n_transformer_decoder_layers_per_mid_block
        self.n_attention_heads: int = n_attention_heads
        self.transformer_maxlength: int = transformer_maxlength
        self.is_finetuning: bool = is_finetuning
        self.lora_rank: int = lora_rank

        assert (
            len(down_transformer_model_dims)
            == len(down_out_dims)
            == len(up_transformer_model_dims)
            == len(up_out_dims)
            == len(n_transformer_encoder_layers_per_scaling_block)
            == len(n_transformer_decoder_layers_per_scaling_block)
        )
        assert len(down_out_dims) >= 1
        assert (
            len(mid_transformer_model_dims)
            == len(mid_out_dims)
            == len(n_transformer_encoder_layers_per_mid_block)
            == len(n_transformer_decoder_layers_per_mid_block)
        )
        assert len(mid_out_dims) >= 1

        self.n_scaling_blocks: int = len(down_out_dims)
        self.n_mid_blocks: int = len(mid_out_dims)
        self.time_embedding_dim: int = max(
            down_out_dims + mid_out_dims + up_out_dims
        )
        if self.time_embedding_dim % 2 != 0:
            self.time_embedding_dim += 1
        self.time_embedding_layer = _SinusoidEmbedding(
            embedding_dim=self.time_embedding_dim,
        )
        self.time_embedding_mlp = nn.Sequential(
            nn.Linear(self.time_embedding_dim, 4 * self.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(4 * self.time_embedding_dim, self.time_embedding_dim),
        )

        self.down_blocks = nn.ModuleList([
            _DownBlock(
                input_dim=target_dim if i == 0 else down_out_dims[i - 1],
                output_dim=down_out_dims[i],
                condition_dim=condition_dim,
                n_conv_layers=n_conv_layers_per_scaling_block,
                in_H=in_H // (2 ** i), in_W=in_W // (2 ** i),
                transformer_model_dim=down_transformer_model_dims[i],
                transformer_feedforward_dim=transformer_feedforward_dim,
                n_transformer_encoder_layers=n_transformer_encoder_layers_per_scaling_block[i],
                n_transformer_decoder_layers=n_transformer_decoder_layers_per_scaling_block[i],
                n_attention_heads=n_attention_heads,
                transformer_maxlength=transformer_maxlength,
                time_embedding_dim=self.time_embedding_dim,
            )
            for i in range(self.n_scaling_blocks)
        ])
        self.up_blocks = nn.ModuleList([
            _UpBlock(
                input_dim=mid_out_dims[-1] if i == 0 else up_out_dims[i - 1],
                output_dim=up_out_dims[i],
                down_output_dim=down_out_dims[-i - 1],
                condition_dim=condition_dim,
                n_conv_layers=n_conv_layers_per_scaling_block,
                in_H=in_H // (2 ** (self.n_scaling_blocks - i)), in_W=in_W // (2 ** (self.n_scaling_blocks - i)),
                transformer_model_dim=up_transformer_model_dims[i],
                transformer_feedforward_dim=transformer_feedforward_dim,
                n_transformer_encoder_layers=n_transformer_encoder_layers_per_scaling_block[-i - 1],
                n_transformer_decoder_layers=n_transformer_decoder_layers_per_scaling_block[-i - 1],
                n_attention_heads=n_attention_heads,
                transformer_maxlength=transformer_maxlength,
                time_embedding_dim=self.time_embedding_dim,
            )
            for i in range(self.n_scaling_blocks)
        ])
        self.mid_blocks = nn.ModuleList([
            _MidBlock(
                input_dim=down_out_dims[-1] if i == 0 else mid_out_dims[i - 1],
                output_dim=mid_out_dims[i],
                condition_dim=condition_dim,
                n_conv_layers=n_conv_layers_per_mid_block,
                in_H=in_H // (2 **self.n_scaling_blocks), in_W=in_W // (2 ** self.n_scaling_blocks),
                transformer_model_dim=mid_transformer_model_dims[i],
                transformer_feedforward_dim=transformer_feedforward_dim,
                n_transformer_encoder_layers=n_transformer_encoder_layers_per_mid_block[i],
                n_transformer_decoder_layers=n_transformer_decoder_layers_per_mid_block[i],
                n_attention_heads=n_attention_heads,
                transformer_maxlength=transformer_maxlength,
                time_embedding_dim=self.time_embedding_dim,
            )
            for i in range(self.n_mid_blocks)
        ])
        self.head = nn.Sequential(
            nn.GroupNorm(
                num_groups=_group_count(self.up_out_dims[-1]),
                num_channels=self.up_out_dims[-1],
            ),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=self.up_out_dims[-1],
                out_channels=target_dim,
                kernel_size=1,
            ),
        )
        self.n_lora_adapters: int = 0
        if is_finetuning:
            assert self.lora_rank > 0
            for i, layer in enumerate(self.time_embedding_mlp):
                if type(layer) is nn.Linear:
                    self.time_embedding_mlp[i] = LoRALinear(base=layer, rank=self.lora_rank)
                    self.n_lora_adapters += 1
            for block in self.down_blocks:
                assert isinstance(block, _DownBlock)
                self.n_lora_adapters += block.enable_lora(rank=self.lora_rank)
            for block in self.mid_blocks:
                assert isinstance(block, _MidBlock)
                block = cast(_MidBlock, block)
                self.n_lora_adapters += block.enable_lora(rank=self.lora_rank)
            for block in self.up_blocks:
                assert isinstance(block, _UpBlock)
                self.n_lora_adapters += block.enable_lora(rank=self.lora_rank)
            print(
                f"Applied {self.n_lora_adapters} LoRA adapters to transformer, "
                "timestep, residual, and sampling layers"
            )

    def forward(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
        integer_step: torch.Tensor,
        condition_days: torch.Tensor,
        target_days: torch.Tensor,
    ) -> torch.Tensor:
        target_N, target_T, target_D, target_H, target_W = target.shape
        condition_N, condition_L, condition_D, condition_H, condition_W = condition.shape
        step_N, step_L = integer_step.shape
        condition_day_N, condition_day_L = condition_days.shape
        target_day_N, target_day_L = target_days.shape
        assert target_N == condition_N == step_N == condition_day_N == target_day_N
        assert condition_day_L == condition_L
        assert target_day_L == target_T
        assert target_D == self.target_dim
        assert condition_D == self.condition_dim
        assert (condition_H, condition_W) == (target_H, target_W)
        assert step_L == 1

        block_step = self.time_embedding_layer(integer_step).squeeze(1)
        block_step = self.time_embedding_mlp(block_step)
        assert block_step.shape == (target_N, self.time_embedding_dim)

        # Check if too many scaling blocks
        _min_dim: float = min(target_H, target_W) / (2 ** self.n_scaling_blocks)
        assert _min_dim % 2 == 0, (
            f"too many scaling blocks (self.n_scaling_blocks={self.n_scaling_blocks}) "
            f"for target dimension: {(target_H, target_W)}"
        )
        # UNet
        down_input: torch.Tensor = target
        down_outputs: list[torch.Tensor] = []
        for i in range(self.n_scaling_blocks):
            down_outputs.append(
                self.down_blocks[i](
                    target=down_input,
                    condition=condition,
                    step=block_step, condition_days=condition_days, target_days=target_days,
                )
            )
            down_input = down_outputs[-1]

        mid_output: torch.Tensor = down_outputs[-1]
        for i in range(self.n_mid_blocks):
            mid_output = self.mid_blocks[i](
                target=mid_output,
                condition=condition,
                step=block_step, condition_days=condition_days, target_days=target_days,
            )

        up_output: torch.Tensor = mid_output
        for i in range(self.n_scaling_blocks):
            skip: torch.Tensor = down_outputs.pop()
            assert skip.shape[3:] == up_output.shape[3:], (
                "Skip connection and decoder feature maps must share spatial dimensions: "
                f"skip.shape={skip.shape}, up_output.shape={up_output.shape}"
            )
            concat: torch.Tensor = torch.cat([skip, up_output], dim=2)
            expected_channels: int = cast(_UpBlock, self.up_blocks[i]).input_dim
            assert concat.shape[2] == expected_channels, (
                "Concatenated skip features do not match expected channel count for up block: "
                f"concat.shape[2]={concat.shape[2]}, expected_channels={expected_channels}"
            )
            up_output = self.up_blocks[i](
                target=concat,
                condition=condition,
                step=block_step, condition_days=condition_days, target_days=target_days,
            )

        assert len(down_outputs) == 0, f"down_outputs must exhaust, getting {len(down_outputs)} items left"
        output: torch.Tensor = self.head(up_output.flatten(0, 1))
        output = output.reshape_as(target)
        return output


class _DiffusionProcess:

    def __init__(self, noise_scheduler: CosineNoiseScheduler):
        self.noise_scheduler: CosineNoiseScheduler = noise_scheduler
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
        alpha_bar = alpha_bar[:, :, None, None, None]
        return alpha_bar

    def compute_beta(self, step: torch.Tensor) -> torch.Tensor:
        step_N, _ = step.shape  # (batch_size, 1)
        # beta ranges from k=1,...,K
        assert torch.all(step.long() - 1 >= 0)
        assert torch.all(step.long() - 1 <= self.noise_scheduler.n_steps - 1)
        beta: torch.Tensor = self.beta_schedule.to(step.device)[step.long() - 1]
        assert beta.shape == (step_N, 1)
        beta = beta[:, :, None, None, None]
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
        target_N, target_T, target_D, target_H, target_W = target_k.shape
        velocity_N, velocity_T, velocity_D, velocity_H, velocity_W = predicted_velocity.shape
        alpha_bar_N, _, _, _, _ = alpha_bar.shape  # (batch_size, 1, 1, 1, 1)
        assert target_k.shape == predicted_velocity.shape
        assert target_N == velocity_N == alpha_bar_N
        # Original target prediction at k
        target_0: torch.Tensor = torch.sqrt(alpha_bar) * target_k - torch.sqrt(1 - alpha_bar) * predicted_velocity
        assert target_0.shape == target_k.shape
        return target_0

    # reverse process
    def compute_gaussian(self, target_k: torch.Tensor, predicted_velocity: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
        target_N, target_T, target_D, target_H, target_W = target_k.shape
        velocity_N, velocity_T, velocity_D, velocity_H, velocity_W = predicted_velocity.shape
        alpha_bar_N, _, _, _, _ = alpha_bar.shape  # (batch_size, 1)
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

    def __init__(self, eta: float, noise_scheduler: CosineNoiseScheduler) -> None:
        super().__init__(noise_scheduler=noise_scheduler)
        self.eta: float = eta
        assert 0. <= self.eta <= 1.

    def sampling_timesteps(self, inference_steps: int) -> list[int]:
        """Original training indices, descending from K to 0"""
        train_steps = self.noise_scheduler.n_steps
        if type(inference_steps) is not int or not 1 <= inference_steps <= train_steps:
            raise ValueError(
                f"inference_steps must be an integer in [1, {train_steps}], "
                f"got {inference_steps!r}"
            )
        return [i * train_steps // inference_steps for i in range(inference_steps, -1, -1)]

    def sample(
        self,
        target_k: torch.Tensor, predicted_velocity: torch.Tensor, k: torch.Tensor,
        k_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        target_N, target_T, target_D, target_H, target_W = target_k.shape
        velocity_N, velocity_T, velocity_D, velocity_H, velocity_W = predicted_velocity.shape
        step_N, _ = k.shape  # (batch_size, 1)
        assert target_k.shape == predicted_velocity.shape
        assert target_N == velocity_N == step_N  # batch_size

        if k_prev is None:
            k_prev = k - 1
        if k_prev.shape != k.shape:
            raise ValueError("k_prev must have the same shape as k")
        if not torch.all((k == k.long()) & (k_prev == k_prev.long())):
            raise ValueError("Sampling timesteps must be integer training indices")
        if not torch.all((0 <= k_prev) & (k_prev < k) & (k <= self.noise_scheduler.n_steps)):
            raise ValueError("Expected 0 <= k_prev < k <= training n_steps")

        # Alpha bar
        alpha_bar: torch.Tensor = self.compute_alpha_bar(step=k)
        alpha_bar_prev: torch.Tensor = self.compute_alpha_bar(step=k_prev)
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
