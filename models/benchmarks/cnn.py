import torch
import torch.nn as nn
from models.common import NamedModel
from models.adaptation.lora import LoRAConv2d


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


class CNN(_Finetunable, NamedModel, nn.Module):

    def __init__(
        self,
        n_input_days: int,
        n_output_days: int,
        in_features: int,
        out_features: int,
        embedding_dim: int,
        n_hidden_layers: int,
        is_finetuning: bool = False,
        lora_rank: int = 0,
    ) -> None:
        super().__init__()
        self.n_input_days: int = n_input_days
        self.n_output_days: int = n_output_days
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.embedding_dim: int = embedding_dim
        self.n_hidden_layers: int = n_hidden_layers
        self.is_finetuning: bool = is_finetuning
        self.lora_rank: int = lora_rank
        self.n_lora_conv_layers: int = 0

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels=n_input_days * in_features, out_channels=embedding_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.25),
        ]
        for _ in range(n_hidden_layers):
            layers += [
                nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(p=0.25),
            ]
        # Equivalent to MLPs
        for _ in range(3):
            layers += [
                nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1),
                nn.ReLU(),
            ]
        layers += [nn.Conv2d(in_channels=embedding_dim, out_channels=n_output_days * out_features, kernel_size=1)]
        self.cnn = nn.Sequential(*layers)

        if self.is_finetuning:
            assert self.lora_rank > 0
            self.n_lora_conv_layers = self.enable_lora_conv2d(rank=self.lora_rank)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, n_input_days, H, W, in_features = input.shape
        assert n_input_days == self.n_input_days
        assert (H, W) == (192, 288)
        assert in_features == self.in_features

        input_tensor_reshaped: torch.Tensor = input.permute(0, 1, 4, 2, 3).flatten(start_dim=1, end_dim=2)
        assert input_tensor_reshaped.shape == (batch_size, self.n_input_days * self.in_features , 192, 288)
        output_tensor: torch.Tensor = self.cnn(input=input_tensor_reshaped)
        assert output_tensor.shape == (batch_size, self.n_output_days * self.out_features, 192, 288)
        output_tensor = output_tensor.reshape(batch_size, self.n_output_days, self.out_features, 192, 288)
        output_tensor = output_tensor.permute(0, 1, 3, 4, 2)
        assert output_tensor.shape == (batch_size, self.n_output_days, 192, 288, self.out_features)
        return output_tensor

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
                count += CNN._replace_conv2d_with_lora(module=child, rank=rank)
        return count

    def enable_lora_conv2d(self, rank: int) -> int:
        assert rank > 0
        return CNN._replace_conv2d_with_lora(module=self, rank=rank)
