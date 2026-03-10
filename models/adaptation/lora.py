from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Linear):

    def __init__(self, base: nn.Linear, rank: int) -> None:
        super().__init__(
            in_features=base.in_features,
            out_features=base.out_features,
            bias=(base.bias is not None),
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

        self.rank: int = rank
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.xavier_normal_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out: torch.Tensor = F.linear(x, self.weight, self.bias)
        lora_out: torch.Tensor = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base_out + lora_out.to(base_out.dtype)


class LoRAConv2d(nn.Conv2d):

    def __init__(self, base: nn.Conv2d, rank: int) -> None:
        assert rank > 0
        if base.groups != 1:
            raise RuntimeError(
                f"{self.__class__.__name__} is only able to wrap "
                f"Conv2d with groups=1, getting groups={self.groups}"
            )
        assert base.groups == 1
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
        with torch.no_grad():
            self.weight.copy_(base.weight)
            if base.bias is not None:
                assert self.bias is not None
                self.bias.copy_(base.bias)

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.rank: int = rank
        k_h, k_w = kernel_size
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_channels * k_h))
        self.lora_B = nn.Parameter(torch.zeros(self.out_channels * k_w, rank))
        nn.init.xavier_normal_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out: torch.Tensor = self._conv_forward(x, self.weight, self.bias)
        # Conv2d LoRA via equivalent delta kernel: B @ A -> reshape to weight shape
        delta_w: torch.Tensor = (self.lora_B @ self.lora_A).reshape_as(self.weight)
        lora_out: torch.Tensor = self._conv_forward(x, delta_w, None)
        return base_out + lora_out


class LoRATConv2d(nn.ConvTranspose2d):

    def __init__(self, base: nn.ConvTranspose2d, rank: int) -> None:
        assert rank > 0
        if base.groups != 1:
            raise RuntimeError(
                f"{self.__class__.__name__} is only able to wrap "
                f"ConvTranspose2d with groups=1, getting groups={base.groups}"
            )

        kernel_size = cast(tuple[int, int], base.kernel_size)
        stride = cast(tuple[int, int], base.stride)
        padding = cast(tuple[int, int], base.padding)
        dilation = cast(tuple[int, int], base.dilation)
        output_padding = cast(tuple[int, int], base.output_padding)

        super().__init__(
            in_channels=base.in_channels,
            out_channels=base.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=base.groups,
            bias=(base.bias is not None),
            dilation=dilation,
            padding_mode=base.padding_mode,
        )
        with torch.no_grad():
            self.weight.copy_(base.weight)
            if base.bias is not None:
                assert self.bias is not None
                self.bias.copy_(base.bias)

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.rank: int = rank
        k_h, k_w = kernel_size
        self.lora_A = nn.Parameter(torch.empty(rank, self.out_channels * k_w))
        self.lora_B = nn.Parameter(torch.zeros(self.in_channels * k_h, rank))
        nn.init.xavier_normal_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stride = cast(tuple[int, int], self.stride)
        padding = cast(tuple[int, int], self.padding)
        output_padding = cast(tuple[int, int], self.output_padding)
        dilation = cast(tuple[int, int], self.dilation)
        base_out: torch.Tensor = F.conv_transpose2d(
            x, self.weight, self.bias,
            stride, padding, output_padding,
            self.groups, dilation,
        )
        delta_w: torch.Tensor = (self.lora_B @ self.lora_A).reshape_as(self.weight)
        lora_out = F.conv_transpose2d(
            x, delta_w, None,
            stride, padding, output_padding,
            self.groups, dilation,
        )
        return base_out + lora_out.to(base_out.dtype)
