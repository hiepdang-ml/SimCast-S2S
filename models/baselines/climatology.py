import torch
import torch.nn as nn

from models.common import NamedModel


class Climatology(NamedModel, nn.Module):

    def __init__(
        self,
        n_input_days: int,
        n_output_days: int,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.n_input_days: int = n_input_days
        self.n_output_days: int = n_output_days
        self.in_features: int = in_features
        self.out_features: int = out_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, n_input_days, H, W, in_features = input.shape
        assert n_input_days == self.n_input_days
        assert in_features == self.in_features
        return input.new_zeros(
            (batch_size, self.n_output_days, H, W, self.out_features)
        )
