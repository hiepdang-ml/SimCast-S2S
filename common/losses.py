from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):

    def __init__(self, lambda_: float):
        super().__init__()
        self.lambda_: float = lambda_
        assert 0. <= lambda_ <= 1.

    def forward(
        self, x_hat: torch.Tensor, true_x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
    ) -> torch.Tensor:
        assert x_hat.shape == true_x.shape
        assert mu.shape == logvar.shape
        weight: torch.Tensor = 1. / (true_x.abs().mean(dim=(0, 1, 2, 3), keepdim=True) + 1e-6)   # (1, 1, 1, 1, n_features)
        reconstruction_loss = (((x_hat - true_x) ** 2) * weight).sum() / weight.sum()
        kl_divergence: torch.Tensor = 0.5 * torch.mean(mu.pow(2) + logvar.exp() - 1 - logvar)
        negative_elbo: torch.Tensor = self.lambda_ * reconstruction_loss + (1 - self.lambda_) * kl_divergence
        return negative_elbo

