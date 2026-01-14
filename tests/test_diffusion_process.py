import unittest

import torch

from models.diffusion import (
    CosineNoiseScheduler,
    ForwardProcess,
    LinearNoiseScheduler,
    ReverseProcess,
)


class DiffusionProcessConsistencyTest(unittest.TestCase):
    def test_forward_reverse_consistency_eta_zero(self) -> None:
        torch.manual_seed(0)
        scheduler = LinearNoiseScheduler(n_steps=10, beta_min=1e-4, beta_max=0.02)
        forward = ForwardProcess(noise_scheduler=scheduler)
        reverse = ReverseProcess(eta=0.0, noise_scheduler=scheduler)

        original = torch.randn(2, 3, 4, 5)
        for step_value in (1, 5, 10):
            step = torch.full((original.shape[0], 1), step_value, dtype=torch.long)
            noisy, true_velocity = forward.add_noise(original_latent=original, k=step)
            _, reconstructed = reverse.sample(
                target_k=noisy, predicted_velocity=true_velocity, k=step
            )
            torch.testing.assert_close(reconstructed, original, atol=1e-5, rtol=1e-5)

    def test_forward_reverse_consistency_cosine_eta_zero(self) -> None:
        torch.manual_seed(1)
        scheduler = CosineNoiseScheduler(n_steps=12, cosine_offset=0.008)
        forward = ForwardProcess(noise_scheduler=scheduler)
        reverse = ReverseProcess(eta=0.0, noise_scheduler=scheduler)

        original = torch.randn(1, 2, 3, 4)
        for step_value in (1, 6, 12):
            step = torch.full((original.shape[0], 1), step_value, dtype=torch.long)
            noisy, true_velocity = forward.add_noise(original_latent=original, k=step)
            _, reconstructed = reverse.sample(
                target_k=noisy, predicted_velocity=true_velocity, k=step
            )
            torch.testing.assert_close(reconstructed, original, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()


