import torch
import torch.nn as nn
from ..common import NamedModel


class PatchEmbedding(nn.Module):

    def __init__(self, in_features: int, embedding_dim: int, patch_size: int) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.embedding_dim: int = embedding_dim
        self.patch_size: int = patch_size
        assert 192 % self.patch_size == 0 and 288 % self.patch_size == 0
        self.n_patches: int = (192 // self.patch_size) * (288 // self.patch_size)
        self.embedder = nn.Conv2d(
            in_channels=in_features, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, n_input_days, H, W, n_input_vars = input.shape
        assert (H, W) == (192, 288)
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        output: torch.Tensor = input.reshape(batch_size * n_input_days, H, W, n_input_vars).permute(0, 3, 1, 2)
        output = self.embedder(output)
        assert output.shape == (batch_size * n_input_days, self.embedding_dim, H // self.patch_size, W // self.patch_size)
        output = output.flatten(2, 3).transpose(1, 2).reshape(batch_size, n_input_days * self.n_patches, self.embedding_dim)
        return output


class PositionalEmbedding(nn.Module):
    
    def __init__(self, n_patches: int, embedding_dim: int) -> None:
        super().__init__()
        self.n_patches: int = n_patches
        self.embedding_dim: int = embedding_dim
        self.pos_embedding = nn.Parameter(data=torch.randn(365, n_patches, embedding_dim))

    def forward(self, time_indices: torch.Tensor) -> torch.Tensor:
        batch_size, n_input_days = time_indices.shape
        assert time_indices.dtype in (torch.int32, torch.int64)
        output: torch.Tensor = self.pos_embedding[time_indices]
        assert output.shape == (batch_size, n_input_days, self.n_patches, self.embedding_dim)
        return output.flatten(1, 2)


class Encoder(nn.Module):

    def __init__(self, embedding_dim: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.n_heads: int = n_heads
        self.n_layers: int = n_layers
        self.dropout: float = dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads, batch_first=True, dropout=dropout,
            dim_feedforward=4096    # use default
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = input.shape
        assert embedding_dim == self.embedding_dim
        output: torch.Tensor = self.temporal_encoder(input)
        assert output.shape == (batch_size, sequence_length, embedding_dim)
        return output


class Decoder(nn.Module):

    def __init__(
        self, n_input_days: int, embedding_dim: int, patch_size: int, out_features: int, dropout: float,
    ) -> None:
        super().__init__()
        self.n_input_days: int = n_input_days
        self.embedding_dim: int = embedding_dim
        self.patch_size: int = patch_size
        self.out_features: int = out_features
        self.dropout: float = dropout
        assert 192 % self.patch_size == 0 and 288 % self.patch_size == 0
        self.n_hpatches: int = 192 // self.patch_size
        self.n_wpatches: int = 288 // self.patch_size
        self.n_patches: int = self.n_hpatches * self.n_wpatches
        self.sequence_length: int = self.n_input_days * self.n_patches

        assert self.patch_size & (self.patch_size - 1) == 0, "patch_size must be a power of 2"
        n_layers: int = self.patch_size.bit_length() - 1
        
        self.hidden_dim: int = 4096
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                in_channels=embedding_dim * n_input_days, out_channels=self.hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        ]
        for _ in range(n_layers - 1):
            layers.extend([
                nn.ConvTranspose2d(
                    in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ])

        self.upscale = nn.Sequential(*layers)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=self.hidden_dim, out_features=self.out_features),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = input.shape
        assert sequence_length == self.sequence_length
        assert embedding_dim == self.embedding_dim
        output: torch.Tensor = input.reshape(batch_size, self.n_input_days, self.n_hpatches, self.n_wpatches, embedding_dim)
        
        output = output.permute(0, 4, 1, 2, 3).flatten(start_dim=1, end_dim=2)
        assert output.shape == (batch_size, embedding_dim * self.n_input_days, self.n_hpatches, self.n_wpatches)
        output = self.upscale(output)
        assert output.shape == (batch_size, self.hidden_dim, 192, 288)
        output = output.permute(0, 2, 3, 1)
        output = output.unsqueeze(dim=1)
        
        assert output.shape == (batch_size, 1, 192, 288, self.hidden_dim)
        output = self.mlp(output)
        assert output.shape == (batch_size, 1, 192, 288, self.out_features)
        return output


class ViT(NamedModel, nn.Module):

    def __init__(
        self, 
        n_input_days: int,
        in_features: int, out_features: int, embedding_dim: int,
        patch_size: int, n_heads: int, n_transformer_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_input_days: int = n_input_days
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.embedding_dim: int = embedding_dim
        self.patch_size: int = patch_size
        self.n_heads: int = n_heads
        self.n_transformer_layers: int = n_transformer_layers
        self.dropout: float = dropout

        self.n_hpatches: int = 192 // patch_size
        self.n_wpatches: int = 288 // patch_size
        self.n_patches: int = self.n_hpatches * self.n_wpatches
        self.sequence_length: int = self.n_input_days * self.n_patches

        self.patch_embedding_layer = PatchEmbedding(
            in_features=in_features, embedding_dim=embedding_dim, patch_size=patch_size
        )
        self.pos_embedding_layer = PositionalEmbedding(
            n_patches=self.n_patches, embedding_dim=embedding_dim
        )
        self.encoder = Encoder(
            embedding_dim=embedding_dim, n_heads=n_heads, n_layers=n_transformer_layers, dropout=dropout,
        )
        self.decoder = Decoder(
            n_input_days=n_input_days, 
            embedding_dim=embedding_dim, patch_size=patch_size,
            out_features=out_features, dropout=dropout,
        )

    def forward(self, input: torch.Tensor, input_indices: torch.Tensor) -> torch.Tensor:
        batch_size, n_input_days, H, W, in_features = input.shape
        assert n_input_days == self.n_input_days
        assert in_features == self.in_features
        assert (H, W) == (192, 288)
        assert input_indices.shape == (batch_size, self.n_input_days)

        patch_embedding: torch.Tensor = self.patch_embedding_layer(input)
        assert patch_embedding.shape == (batch_size, self.sequence_length, self.embedding_dim)

        pos_embedding: torch.Tensor = self.pos_embedding_layer(input_indices)
        assert pos_embedding.shape == (batch_size, self.sequence_length, self.embedding_dim)

        output: torch.Tensor = self.encoder(input=patch_embedding + pos_embedding)
        assert output.shape == (batch_size, self.sequence_length, self.embedding_dim)
        output = self.decoder(input=output)
        assert output.shape == (batch_size, 1, 192, 288, self.out_features)
        return output


