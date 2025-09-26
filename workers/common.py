import torch

from models.diffusion import VAEEncoder


class RequireVAEEncoders:

    def vae_encode(self, condition: torch.Tensor, target: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        # Encode condition
        wind_indices: list[int] = self.indices_by_context_group["wind"]
        wind_condition: torch.Tensor = condition[..., wind_indices]
        geopotential_indices: list[int] = self.indices_by_context_group["geopotential"]
        geopotential_condition: torch.Tensor = condition[..., geopotential_indices]
        thermaldynamic_indices: list[int] = self.indices_by_context_group["thermaldynamic"]
        thermaldynamic_condition: torch.Tensor = condition[..., thermaldynamic_indices]
        precipitation_indices: list[int] = self.indices_by_context_group["precipitation"]
        precipitation_condition: torch.Tensor = condition[..., precipitation_indices]

        wind_latents: list[torch.Tensor] = []
        geopotential_latents: list[torch.Tensor] = []
        thermaldynamic_latents: list[torch.Tensor] = []
        precipitation_latents: list[torch.Tensor] = []

        for day in range(condition.shape[1]):  # n_days
            wind_latents.append(
                VAEEncoder.reparameterize(
                    *self.wind_encoder(wind_condition[:, day:day+1, :, :, :])
                )
            )
            geopotential_latents.append(
                VAEEncoder.reparameterize(
                    *self.geopotential_encoder(geopotential_condition[:, day:day+1, :, :, :])
                )
            )
            thermaldynamic_latents.append(
                VAEEncoder.reparameterize(
                    *self.thermaldynamic_encoder(thermaldynamic_condition[:, day:day+1, :, :, :])
                )
            )
            precipitation_latents.append(
                VAEEncoder.reparameterize(
                    *self.precipitation_encoder(precipitation_condition[:, day:day+1, :, :, :])
                )
            )

        wind_latent: torch.Tensor = torch.stack(tensors=wind_latents, dim=2)
        geopotential_latent: torch.Tensor = torch.stack(tensors=geopotential_latents, dim=2)
        thermaldynamic_latent: torch.Tensor = torch.stack(tensors=thermaldynamic_latents, dim=2)
        precipitation_latent: torch.Tensor = torch.stack(tensors=precipitation_latents, dim=2)

        # Encode precipitation target
        assert target.shape[1] == 1
        target_latent: torch.Tensor = VAEEncoder.reparameterize(*self.precipitation_encoder(target))
        return wind_latent, geopotential_latent, thermaldynamic_latent, precipitation_latent, target_latent
    
