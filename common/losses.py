import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):

    def __init__(self, lambda_: float):
        super().__init__()
        self.lambda_: float = lambda_
        assert 0. <= lambda_ <= 1.

    def forward(
        self, 
        x_hat: torch.Tensor, true_x: torch.Tensor, 
        mu: torch.Tensor, logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x_hat.shape == true_x.shape
        assert mu.shape == logvar.shape
        reconstruction_mae: torch.Tensor = F.l1_loss(input=x_hat, target=true_x, reduction="mean")
        reconstruction_loss: torch.Tensor = F.mse_loss(input=x_hat, target=true_x, reduction="mean")
        kl_divergence: torch.Tensor = 0.5 * torch.mean(mu.pow(2) + logvar.exp() - 1 - logvar)
        negative_elbo: torch.Tensor = self.lambda_ * reconstruction_loss + (1 - self.lambda_) * kl_divergence
        return reconstruction_loss, kl_divergence, negative_elbo, reconstruction_mae


class DiffusionLoss(nn.Module):

    def __init__(self, lambda_: float):
        super().__init__()
        self.lambda_: float = lambda_
        assert 0. <= lambda_ <= 1.

    def forward(
        self, 
        gaussian_hat: torch.Tensor, gaussian_true: torch.Tensor, 
        x_hat: torch.Tensor, x_true: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # gaussian_mae, prediction_mae, loss
        assert gaussian_hat.shape == gaussian_true.shape
        assert x_hat.shape == x_true.shape
        gaussian_mae: torch.Tensor = F.l1_loss(input=gaussian_hat, target=gaussian_true)
        sampling_mae: torch.Tensor = F.l1_loss(input=x_hat, target=x_true)
        gaussian_loss: torch.Tensor = F.mse_loss(input=gaussian_hat, target=gaussian_true)
        sampling_loss: torch.Tensor = F.mse_loss(input=x_hat, target=x_true)
        loss: torch.Tensor = self.lambda_ * gaussian_loss + (1 - self.lambda_) * sampling_loss
        return gaussian_loss, sampling_loss, loss, gaussian_mae, sampling_mae
    

