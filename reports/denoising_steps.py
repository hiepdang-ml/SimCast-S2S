from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import torch

from common.configs import DiffusionConfig, MetaData
from common.utils import CheckpointLoader
from datapipeline.dataset import CESM2, ERA5
from models.diffusion import (
    CosineNoiseScheduler,
    LinearNoiseScheduler,
    ReverseProcess,
    UNetDenoiser,
    VAE,
    VAE_Hydro,
    VAE_Mass,
    VAE_Precip,
    VAE_Thermal,
    VAE_Wind,
)
from workers.common import RequireVAEEncoders


class DenoisingStepFigureBuilder(RequireVAEEncoders):

    def __init__(
        self,
        dataset_name: Literal["cesm2", "era5"],
        sample_index: int,
        output_dir: Path,
        seed: int,
        device: torch.device,
        vlim: float,
    ) -> None:
        self.dataset_name = dataset_name
        self.sample_index = sample_index
        self.output_dir = output_dir
        self.seed = seed
        self.device = device
        if vlim <= 0:
            raise ValueError(f"vlim must be positive, got {vlim}")
        self.vlim = vlim
        self.config = DiffusionConfig()
        self.metadata = MetaData(dataset_name=dataset_name, tp="test")
        self.dataset = CESM2(metadata=self.metadata) if dataset_name == "cesm2" else ERA5(metadata=self.metadata)
        self.H, self.W = self.metadata.resolution
        self.indices_by_context_group = self.dataset.indices_by_context_group
        if not 0 <= sample_index < len(self.dataset):
            raise IndexError(f"sample_index must be in [0, {len(self.dataset) - 1}], got {sample_index}")

        self.denoiser = self._load_denoiser()
        wind_vae = self._load_vae(self.config.vae_wind_checkpoint, label="wind")
        mass_vae = self._load_vae(self.config.vae_mass_checkpoint, label="mass")
        thermal_vae = self._load_vae(self.config.vae_thermal_checkpoint, label="thermal")
        hydro_vae = self._load_vae(self.config.vae_hydro_checkpoint, label="hydro")
        precip_vae = self._load_vae(self.config.vae_precip_checkpoint, label="precip")
        self.wind_encoder = wind_vae.encoder.to(self.device).eval()
        self.mass_encoder = mass_vae.encoder.to(self.device).eval()
        self.thermal_encoder = thermal_vae.encoder.to(self.device).eval()
        self.hydro_encoder = hydro_vae.encoder.to(self.device).eval()
        self.precip_encoder = precip_vae.encoder.to(self.device).eval()
        self.noise_scheduler = self._noise_scheduler()
        self.reverse_process = ReverseProcess(eta=self.config.eta, noise_scheduler=self.noise_scheduler)

    def _load_denoiser(self) -> UNetDenoiser:
        if self.config.from_checkpoint is None:
            raise ValueError("DiffusionConfig.from_checkpoint must point to a denoiser checkpoint")
        print(f"[denoising-steps] Loading denoiser: {self.config.from_checkpoint}")
        net = CheckpointLoader(checkpoint_path=self.config.from_checkpoint.as_posix()).load(scope=globals())
        if not isinstance(net, UNetDenoiser):
            raise TypeError(f"Expected UNetDenoiser, got {type(net).__name__}")
        return net.to(self.device).eval()

    def _load_vae(self, checkpoint_path: Path, label: str) -> VAE:
        print(f"[denoising-steps] Loading {label} VAE: {checkpoint_path}")
        net = CheckpointLoader(checkpoint_path=checkpoint_path.as_posix()).load(scope=globals())
        if not isinstance(net, VAE):
            raise TypeError(f"Expected VAE for {label}, got {type(net).__name__}")
        return net

    def _noise_scheduler(self) -> LinearNoiseScheduler | CosineNoiseScheduler:
        if self.config.noise_scheduler.lower() == "linear":
            return LinearNoiseScheduler(
                n_steps=self.config.n_steps,
                beta_min=self.config.beta_min,
                beta_max=self.config.beta_max,
            )
        if self.config.noise_scheduler.lower() == "cosine":
            return CosineNoiseScheduler(n_steps=self.config.n_steps)
        raise ValueError(f"Invalid diffusion noise scheduler: {self.config.noise_scheduler}")

    @staticmethod
    def _channel_averaged_latent_frame(latent: torch.Tensor) -> torch.Tensor:
        assert latent.ndim == 5
        assert latent.shape[0] == 1
        return latent.squeeze(dim=0).mean(dim=(0, 1)).cpu()

    def _plot_frame(self, frame: torch.Tensor, step: int) -> Path:
        assert frame.ndim == 2
        rotated_frame = torch.rot90(frame, k=1, dims=(0, 1))
        fig, ax = plt.subplots(figsize=(4.2, 8.0))
        ax.imshow(
            rotated_frame.numpy(),
            cmap="RdBu",
            vmin=-self.vlim,
            vmax=self.vlim,
            origin="lower",
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_axis_off()
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir.joinpath(f"denoising_step_{step:04d}.png")
        fig.savefig(output_path, dpi=400, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)
        return output_path

    @torch.no_grad()
    def run(self) -> None:
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        sample = self.dataset[self.sample_index]
        sampleinfo, condition_days, target_days, condition, groundtruth = self.dataset.collate_fn([sample])
        condition = condition.to(self.device)
        groundtruth = groundtruth.to(self.device)
        condition_days = condition_days.to(self.device)
        target_days = target_days.to(self.device)
        condition_latent, target_latent = self.vae_encode(condition=condition, target=groundtruth)
        target_latent_k = torch.randn_like(target_latent)
        condition_kept = torch.ones((1,), device=self.device, dtype=torch.bool)
        condition_dropped = torch.zeros((1,), device=self.device, dtype=torch.bool)

        for step in range(self.noise_scheduler.n_steps, 0, -1):
            integer_step = torch.full((1, 1), fill_value=step, device=self.device, dtype=torch.long)
            predicted_velocity_cond = self.denoiser(
                target=target_latent_k,
                condition=condition_latent,
                integer_step=integer_step,
                condition_days=condition_days,
                target_days=target_days,
                condition_mask=condition_kept,
            )
            if self.config.guidance_scale == 1:
                predicted_velocity = predicted_velocity_cond
            else:
                predicted_velocity_uncond = self.denoiser(
                    target=target_latent_k,
                    condition=condition_latent,
                    integer_step=integer_step,
                    condition_days=condition_days,
                    target_days=target_days,
                    condition_mask=condition_dropped,
                )
                predicted_velocity = (
                    predicted_velocity_uncond
                    + self.config.guidance_scale * (predicted_velocity_cond - predicted_velocity_uncond)
                )
            target_latent_k, predicted_latent_0 = self.reverse_process.sample(
                target_k=target_latent_k,
                predicted_velocity=predicted_velocity,
                k=integer_step,
            )
            frame = self._channel_averaged_latent_frame(latent=predicted_latent_0)
            output_path = self._plot_frame(frame=frame, step=step)
            print(f"[denoising-steps] Saved: {output_path}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=["cesm2", "era5"], default="era5")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("/scratch/zgp2ps/s2s_results/denoising_steps"))
    parser.add_argument("--seed", type=int, default=7341)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vlim", type=float, default=0.04)
    args: Namespace = parser.parse_args()
    builder = DenoisingStepFigureBuilder(
        dataset_name=args.dataset,
        sample_index=args.sample_index,
        output_dir=args.output_dir,
        seed=args.seed,
        device=torch.device(args.device),
        vlim=args.vlim,
    )
    builder.run()


if __name__ == "__main__":
    main()

# python reports/denoising_steps.py --dataset era5 --sample-index 0 --device cuda:0 --vlim 0.2
