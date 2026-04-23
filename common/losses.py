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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x_hat.shape == true_x.shape
        assert mu.shape == logvar.shape
        # reconstruction loss
        reconstruction_mae: torch.Tensor = F.l1_loss(input=x_hat, target=true_x, reduction="mean")
        reconstruction_loss: torch.Tensor = F.mse_loss(input=x_hat, target=true_x, reduction="mean")
        # KL divergence
        kl_divergence: torch.Tensor = 0.5 * torch.mean(mu.pow(2) + logvar.exp() - 1 - logvar)
        mu: torch.Tensor = torch.mean(mu)
        sigma: torch.Tensor = torch.mean(logvar.exp().sqrt())
        negative_elbo: torch.Tensor = self.lambda_ * reconstruction_loss + (1 - self.lambda_) * kl_divergence
        assert reconstruction_mae.shape == reconstruction_loss.shape == negative_elbo.shape == kl_divergence.shape == mu.shape == sigma.shape  == ()
        return reconstruction_loss, kl_divergence, negative_elbo, reconstruction_mae, mu, sigma


class DiffusionLoss(nn.Module):

    def forward(
        self,
        velocity_hat: torch.Tensor,
        velocity_true: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert velocity_hat.shape == velocity_true.shape
        velocity_mae: torch.Tensor = F.l1_loss(input=velocity_hat, target=velocity_true, reduction="mean")
        velocity_loss: torch.Tensor = F.mse_loss(input=velocity_hat, target=velocity_true, reduction="mean")
        return velocity_loss, velocity_mae
