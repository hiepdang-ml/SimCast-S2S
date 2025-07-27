from typing import *

import torch
import torch.nn as nn
from ..common import NamedModel


class CNN(NamedModel, nn.Module):

    def __init__(
        self, 
        n_input_days: int, 
        in_features: int, 
        out_features: int, 
        embedding_dim: int,
        n_hidden_layers: int,
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.n_input_days: int = n_input_days
        self.embedding_dim: int = embedding_dim
        self.n_hidden_layers: int = n_hidden_layers

        layers: List[nn.Module] = [
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
        layers += [nn.Conv2d(in_channels=embedding_dim, out_channels=out_features, kernel_size=1)]
        self.cnn = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, n_input_days, H, W, in_features = input.shape
        assert n_input_days == self.n_input_days
        assert (H, W) == (192, 288)
        assert in_features == self.in_features

        input_tensor_reshaped: torch.Tensor = input.permute(0, 1, 4, 2, 3).flatten(start_dim=1, end_dim=2)
        assert input_tensor_reshaped.shape == (batch_size, self.n_input_days * self.in_features , 192, 288)
        output_tensor: torch.Tensor = self.cnn(input=input_tensor_reshaped)
        assert output_tensor.shape == (batch_size, self.out_features , 192, 288)
        output_tensor = output_tensor.reshape(batch_size, 1, self.out_features, 192, 288)
        output_tensor = output_tensor.permute(0, 1, 3, 4, 2)
        assert output_tensor.shape == (batch_size, 1, 192, 288, self.out_features)
        return output_tensor
    

