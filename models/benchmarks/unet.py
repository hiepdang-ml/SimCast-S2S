import torch
import torch.nn as nn
from models.common import NamedModel
from models.adaptation.lora import LoRAConv2d


class DoubleConv(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.double_conv: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 4
        output: torch.Tensor = self.double_conv(input)
        return output


class _Finetunable:

    def is_backbone_frozen(self) -> bool:
        return not any(
            param.requires_grad and ("lora_A" not in name and "lora_B" not in name)
            for name, param in self.named_parameters()
        )

    def freeze_backbone(self) -> None:
        for name, param in self.named_parameters():
            param.requires_grad = "lora_A" in name or "lora_B" in name
        print(f"{self.name} backbone has been frozen for LoRA fine-tuning")


class UNet(_Finetunable, NamedModel, nn.Module):

    def __init__(
        self,
        n_input_days: int,
        n_output_days: int,
        in_features: int,
        out_features: int,
        embedding_dim: int,
        is_finetuning: bool = False,
        lora_rank: int = 0,
    ) -> None:
        super().__init__()
        self.n_input_days: int = n_input_days
        self.n_output_days: int = n_output_days
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.embedding_dim: int = embedding_dim
        self.is_finetuning: bool = is_finetuning
        self.lora_rank: int = lora_rank
        self.n_lora_conv_layers: int = 0

        in_channels: int = n_input_days * in_features
        out_channels: int = n_output_days * out_features

        self.inc = DoubleConv(in_features=in_channels, out_features=embedding_dim)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_features=embedding_dim, out_features=embedding_dim * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_features=embedding_dim * 2, out_features=embedding_dim * 4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_features=embedding_dim * 4, out_features=embedding_dim * 8)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_features=embedding_dim * 8, out_features=embedding_dim * 16)
        )

        self.up1 = nn.ConvTranspose2d(in_channels=embedding_dim * 16, out_channels=embedding_dim * 8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(in_features=embedding_dim * 16, out_features=embedding_dim * 8)
        self.up2 = nn.ConvTranspose2d(in_channels=embedding_dim * 8, out_channels=embedding_dim * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(in_features=embedding_dim * 8, out_features=embedding_dim * 4)
        self.up3 = nn.ConvTranspose2d(in_channels=embedding_dim * 4, out_channels=embedding_dim * 2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(in_features=embedding_dim * 4, out_features=embedding_dim * 2)
        self.up4 = nn.ConvTranspose2d(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(in_features=embedding_dim * 2, out_features=embedding_dim)

        self.outc = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=embedding_dim, out_channels=out_channels, kernel_size=1),
        )

        if self.is_finetuning:
            assert self.lora_rank > 0
            self.n_lora_conv_layers = self.enable_lora_conv2d(rank=self.lora_rank)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, n_input_days, H, W, in_features = input.shape
        assert n_input_days == self.n_input_days
        assert in_features == self.in_features
        assert (H, W) == (192, 288)

        x: torch.Tensor = input.permute(0, 1, 4, 2, 3).flatten(1, 2)

        x1 = self.inc(x)       # (N, E, H, W)
        x2 = self.down1(x1)    # (N, 2E, H/2, W/2)
        x3 = self.down2(x2)    # (N, 4E, H/4, W/4)
        x4 = self.down3(x3)    # (N, 8E, H/8, W/8)
        x5 = self.down4(x4)    # (N, 16E, H/16, W/16)

        x = self.up1(x5)       # (N, 8E, H/8, W/8)
        x = self.conv1(torch.cat(tensors=[x, x4], dim=1))  # (N, 8E + 8E = 16E, H/8, W/8)
        x = self.up2(x)        # (N, 4E, H/4, W/4)
        x = self.conv2(torch.cat(tensors=[x, x3], dim=1))  # (N, 4E + 4E = 8E, H/4, W/4)
        x = self.up3(x)        # (N, 2E, H/2, W/2)
        x = self.conv3(torch.cat(tensors=[x, x2], dim=1))  # (N, 2E + 2E = 4E, H/2, W/2)
        x = self.up4(x)        # (N, E, H, W)
        x = self.conv4(torch.cat(tensors=[x, x1], dim=1))  # (N, E + E = 2E, H, W)

        output: torch.Tensor = self.outc(x)       # (N, Cout, H, W)
        assert output.shape == (batch_size, self.n_output_days * self.out_features, H, W)
        output: torch.Tensor = output.reshape(batch_size, self.n_output_days, self.out_features, H, W).permute(0, 1, 3, 4, 2)
        return output

    @staticmethod
    def _replace_conv2d_with_lora(module: nn.Module, rank: int) -> int:
        count: int = 0
        for name, child in list(module.named_children()):
            if isinstance(child, LoRAConv2d):
                continue
            if isinstance(child, nn.Conv2d):
                setattr(module, name, LoRAConv2d(base=child, rank=rank))
                count += 1
            else:
                # DFS: Recursive further into the child module
                count += UNet._replace_conv2d_with_lora(module=child, rank=rank)
        return count

    def enable_lora_conv2d(self, rank: int) -> int:
        assert rank > 0
        return UNet._replace_conv2d_with_lora(module=self, rank=rank)
