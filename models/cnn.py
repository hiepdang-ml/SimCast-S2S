import torch
import torch.nn as nn

from models.modules import SinusoidEmbedding


class CNN(nn.Module):

    def __init__(
        self, 
        n_input_days: int, 
        n_output_days, 
        in_features: int, 
        out_features: int, 
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.n_input_days: int = n_input_days
        self.n_output_days: int = n_output_days
        self.embedding_dim: int = embedding_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=embedding_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=embedding_dim, out_channels=out_features, kernel_size=3, padding=1),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size: int = input_tensor.shape[0]
        assert input_tensor.shape == (batch_size, self.n_input_days, 192, 288, self.in_features)

        input_tensor_reshaped: torch.Tensor = input_tensor.flatten(start_dim=0, end_dim=1).permute(0, 3, 1, 2)
        assert input_tensor_reshaped.shape == (batch_size * self.n_input_days, self.in_features , 192, 288)
        output_tensor: torch.Tensor = self.cnn(input=input_tensor_reshaped)
        output_tensor = output_tensor.reshape(batch_size, self.n_input_days, self.in_features, 192, 288).mean(dim=1, keepdim=False)
        output_tensor = output_tensor.permute(0, 2, 3, 1)
        assert output_tensor.shape == (batch_size, 192, 288, self.out_features)
        return output_tensor
    
