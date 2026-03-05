from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Linear):

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__(
            in_features=base.in_features,
            out_features=base.out_features,
            bias=(base.bias is not None),
        )
        with torch.no_grad():
            self.weight.copy_(base.weight)
            if base.bias is not None:
                assert self.bias is not None
                self.bias.copy_(base.bias)

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        assert rank > 0
        self.rank = rank
        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)  # no-op at init
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(input=x, weight=self.weight, bias=self.bias)
        lora_out = self.lora_B(self.lora_A(x))
        return base_out + lora_out.to(dtype=base_out.dtype)


class LoRAConv2d(nn.Conv2d):

    def __init__(self, base: nn.Conv2d, rank: int) -> None:

        kernel_size = cast(tuple[int, int], base.kernel_size)
        stride = cast(tuple[int, int], base.stride)
        padding = cast(tuple[int, int], base.padding)
        dilation = cast(tuple[int, int], base.dilation)
        super().__init__(
            in_channels=base.in_channels,
            out_channels=base.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=base.groups,
            bias=(base.bias is not None),
            padding_mode=base.padding_mode,
        )
        assert rank > 0
        with torch.no_grad():
            self.weight.copy_(base.weight)
            if base.bias is not None:
                assert self.bias is not None
                self.bias.copy_(base.bias)

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        assert rank > 0
        self.rank: int = rank
        in_features: int = (self.in_channels // self.groups) * kernel_size[0] * kernel_size[1]
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_channels, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)  # no-op at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_w: torch.Tensor = self.lora_B @ self.lora_A
        delta_w = delta_w.reshape_as(self.weight)
        w = self.weight + delta_w.to(dtype=self.weight.dtype)
        return F.conv2d(
            input=x, weight=w, bias=self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups,
        )
