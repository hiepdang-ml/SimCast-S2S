from typing import *
import torch

from models.diffusion import VAEEncoder


class RequireVAEEncoders:

    def vae_encode(self, condition: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode condition
        wind_indices: List[int] = self.indices_by_context_group["wind"]
        wind_condition: torch.Tensor = condition[..., wind_indices]
        wind_condition_latent: torch.Tensor = VAEEncoder.reparameterize(
            *self.wind_encoder(wind_condition)
        )
        geopotential_indices: List[int] = self.indices_by_context_group["geopotential"]
        geopotential_condition: torch.Tensor = condition[..., geopotential_indices]
        geopotential_condition_latent: torch.Tensor = VAEEncoder.reparameterize(
            *self.geopotential_encoder(geopotential_condition)
        )
        thermaldynamic_indices: List[int] = self.indices_by_context_group["thermaldynamic"]
        thermaldynamic_condition: torch.Tensor = condition[..., thermaldynamic_indices]
        thermaldynamic_condition_latent: torch.Tensor = VAEEncoder.reparameterize(
            *self.thermaldynamic_encoder(thermaldynamic_condition)
        )
        precipitation_indices: List[int] = self.indices_by_context_group["precipitation"]
        precipitation_condition: torch.Tensor = condition[..., precipitation_indices]
        precipitation_condition_latent: torch.Tensor = VAEEncoder.reparameterize(
            *self.precipitation_encoder(precipitation_condition)
        )
        # Encode target (just to get shape)
        assert target.shape[1] == 1
        target_latent: torch.Tensor = VAEEncoder.reparameterize(
            *self.precipitation_encoder(target.expand(-1, self.precipitation_encoder.n_days, -1, -1, -1))
        )
        # Concat condition latent
        condition_latent: torch.Tensor = torch.concat(
            # TODO: remove hard order
            tensors=[
                wind_condition_latent, geopotential_condition_latent, 
                thermaldynamic_condition_latent, precipitation_condition_latent
            ],
            dim=1,
        )
        return condition_latent, target_latent
    