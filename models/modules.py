import torch
import torch.nn as nn


class SinusoidEmbedding(nn.Module):

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim: int = embedding_dim

        # Frequency scaling
        self.w = 1. / torch.pow(
            input=torch.tensor(10_000., dtype=torch.float),
            exponent=torch.arange(0, embedding_dim, 2, dtype=torch.float) / embedding_dim,
        )
        assert self.w.shape == (embedding_dim // 2,)
        self.register

    def forward(self, timeframes: torch.Tensor) -> torch.Tensor:
        assert timeframes.ndim == 2
        batch_size, n_timeframes = timeframes.shape
        timeframes = timeframes.unsqueeze(-1)  # (batch_size, n_timeframes, 1)
        sinusoid = torch.zeros(*timeframes.shape[:-1], self.embedding_dim, device=timeframes.device)
        sinusoid[:, :, 0::2] = torch.sin(timeframes * self.w)
        sinusoid[:, :, 1::2] = torch.cos(timeframes * self.w)
        assert sinusoid.shape == (batch_size, n_timeframes, self.embedding_dim)
        return sinusoid