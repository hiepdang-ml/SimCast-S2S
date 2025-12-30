import torch
from models.diffusion.vae import VAEEncoder

class RequireVAEEncoders:

    def vae_encode(self, condition: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encode condition
        wind_indices: list[int] = self.indices_by_context_group["wind"]
        wind_condition: torch.Tensor = condition[..., wind_indices]
        mass_indices: list[int] = self.indices_by_context_group["mass"]
        mass_condition: torch.Tensor = condition[..., mass_indices]
        thermal_indices: list[int] = self.indices_by_context_group["thermal"]
        thermal_condition: torch.Tensor = condition[..., thermal_indices]
        hydro_indices: list[int] = self.indices_by_context_group["hydro"]
        hydro_condition: torch.Tensor = condition[..., hydro_indices]
        precip_indices: list[int] = self.indices_by_context_group["precip"]
        precip_condition: torch.Tensor = condition[..., precip_indices]

        wind_mu, wind_logvar = self._encode(self.wind_encoder, wind_condition)
        mass_mu, mass_logvar = self._encode(self.mass_encoder, mass_condition)
        thermal_mu, thermal_logvar = self._encode(self.thermal_encoder, thermal_condition)
        hydro_mu, hydro_logvar = self._encode(self.hydro_encoder, hydro_condition)
        precip_mu, precip_logvar = self._encode(self.precip_encoder, precip_condition)

        # Encode target
        assert target.shape[1] == 1
        target_latent: torch.Tensor = VAEEncoder.reparameterize(*self.target_encoder(target), scale=1.)
        return (
            wind_mu, wind_logvar, 
            mass_mu, mass_logvar, 
            thermal_mu, thermal_logvar, 
            hydro_mu, hydro_logvar, 
            precip_mu, precip_logvar,
            target_latent,
        )
    
    @staticmethod
    def _encode(
        encoder: VAEEncoder, 
        input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        N, T, H, W, E = input.shape
        input = input.flatten(start_dim=0, end_dim=1).unsqueeze(dim=1) # (N * T, 1, H, W, E)
        mu, logvar = encoder(input)
        assert mu.shape == logvar.shape == (N * T, encoder.latent_dim, encoder.expected_H, encoder.expected_W)
        mu = mu.reshape(N, encoder.latent_dim * T, encoder.expected_H, encoder.expected_W)
        logvar = logvar.reshape(N, encoder.latent_dim * T, encoder.expected_H, encoder.expected_W)
        return mu, logvar

