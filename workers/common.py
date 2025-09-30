import torch

from models.diffusion import VAEEncoder


class RequireVAEEncoders:

    def vae_encode(self, condition: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encode condition
        wind_indices: list[int] = self.indices_by_context_group["wind"]
        wind_condition: torch.Tensor = condition[..., wind_indices]
        geopotential_indices: list[int] = self.indices_by_context_group["geopotential"]
        geopotential_condition: torch.Tensor = condition[..., geopotential_indices]
        thermaldynamic_indices: list[int] = self.indices_by_context_group["thermaldynamic"]
        thermaldynamic_condition: torch.Tensor = condition[..., thermaldynamic_indices]
        precipitation_indices: list[int] = self.indices_by_context_group["precipitation"]
        precipitation_condition: torch.Tensor = condition[..., precipitation_indices]

        condition_latents: list[torch.Tensor] = []
        for day in range(condition.shape[1]):  # n_days
            condition_latents.append(
                VAEEncoder.reparameterize(
                    *self.wind_encoder(wind_condition[:, day:day+1, :, :, :])
                )
            )
            condition_latents.append(
                VAEEncoder.reparameterize(
                    *self.geopotential_encoder(geopotential_condition[:, day:day+1, :, :, :])
                )
            )
            condition_latents.append(
                VAEEncoder.reparameterize(
                    *self.thermaldynamic_encoder(thermaldynamic_condition[:, day:day+1, :, :, :])
                )
            )
            condition_latents.append(
                VAEEncoder.reparameterize(
                    *self.precipitation_encoder(precipitation_condition[:, day:day+1, :, :, :])
                )
            )

        condition_latent: torch.Tensor = torch.cat(tensors=condition_latents, dim=1)
        # Encode precipitation target
        assert target.shape[1] == 1
        target_latent: torch.Tensor = VAEEncoder.reparameterize(*self.precipitation_encoder(target))
        return target_latent, condition_latent
    
