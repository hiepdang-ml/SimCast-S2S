import torch


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
    
        condition_latents: list[torch.Tensor] = []
        for day in range(condition.shape[1]):  # n_days
            # NOTE: only get the mean (deterministic latent)
            condition_latents.append(
                self.wind_encoder(wind_condition[:, day:day+1, :, :, :])[0]
            )
            condition_latents.append(
                self.mass_encoder(mass_condition[:, day:day+1, :, :, :])[0]
            )
            condition_latents.append(
                self.thermal_encoder(thermal_condition[:, day:day+1, :, :, :])[0]
            )
            condition_latents.append(
                self.hydro_encoder(hydro_condition[:, day:day+1, :, :, :])[0]
            )
            condition_latents.append(
                self.precip_encoder(precip_condition[:, day:day+1, :, :, :])[0]
            )
        condition_latent: torch.Tensor = torch.cat(tensors=condition_latents, dim=1)
        # Encode precip target
        assert target.shape[1] == 1
        target_latent: torch.Tensor = self.precip_encoder(target)[0]
        return target_latent, condition_latent
    
