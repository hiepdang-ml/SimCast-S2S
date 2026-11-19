from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
from datapipeline.utils import DataBatch
from datapipeline.dataset import CESM2, ERA5
from common.utils import Accumulator, EarlyStopping, Timer, Logger, CheckpointSaver, ParamCounter
from common.losses import VAELoss, DiffusionLoss
from models.baselines import CNN, UNet
from models.simcasts2s import (
    VAE, VAEEncoder, Denoiser, CosineNoiseScheduler, ForwardProcess,
)
from .common import RequireVAEEncoders


class _AbstractTrainer(ABC):

    def __init__(self,
        net: CNN | UNet | VAE | Denoiser,
        lr: float,
        train_dataset: CESM2 | ERA5, val_dataset: CESM2 | ERA5,
        train_batch_size: int, val_batch_size: int,
        local_rank: int,
    ):
        self.model_name: str = net.name
        self.lr: float = lr
        self.train_dataset: CESM2 | ERA5 = train_dataset
        self.val_dataset: CESM2 | ERA5 = val_dataset
        self.train_batch_size: int = train_batch_size
        self.val_batch_size: int = val_batch_size

        self.local_rank: int = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.net: DistributedDataParallel = DistributedDataParallel(
            module=net.to(self.device), device_ids=[self.local_rank],
            output_device=self.local_rank, broadcast_buffers=False,
        )
        if local_rank == 0:
            self.param_counter = ParamCounter(self.net)
            print(self.param_counter.summary())

        self.train_sampler: DistributedSampler = DistributedSampler(dataset=train_dataset, shuffle=True, drop_last=False)
        self.val_sampler: DistributedSampler = DistributedSampler(dataset=val_dataset, shuffle=False, drop_last=False)

        # Dataloaders
        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=train_batch_size,
            collate_fn=CESM2.collate_fn,
            prefetch_factor=2,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
            sampler=self.train_sampler,
        )
        self.val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=CESM2.collate_fn,
            prefetch_factor=2,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
            sampler=self.val_sampler,
        )
        self.__loss_function: nn.Module | None = None
        self.mae: nn.Module = nn.L1Loss(reduction="mean")
        self.optimizer = Adam(params=self.net.parameters(), lr=lr)
        self.scheduler_configs: dict[str, Any] = {}

    @property
    def loss_function(self) -> nn.Module:
        if self.__loss_function is None:
            raise RuntimeError(f"loss_function is not set for {self.__class__.__name__}")
        return self.__loss_function

    @loss_function.setter
    def loss_function(self, callable: nn.Module) -> None:
        self.__loss_function = callable

    def _start_training_epoch(self) -> None:
        pass

    def _finish_training_epoch(self) -> dict[str, float]:
        return {}

    def train(
        self,
        n_epochs: int, patience: int, tolerance: float,
        checkpoint_directory: str | None = None, save_frequency: int = 5,
    ) -> None:
        early_stopping = EarlyStopping(patience, tolerance)
        timer = Timer()
        logger = Logger()
        checkpoint_saver: CheckpointSaver | None = None
        if checkpoint_directory:
            checkpoint_saver = CheckpointSaver(model=self.net, dirpath=checkpoint_directory)

        for epoch in range(1, n_epochs + 1):
            timer.start_epoch(epoch=epoch)
            self.net.train()
            self.train_sampler.set_epoch(epoch)
            self._start_training_epoch()

            is_main_process: bool = dist.get_rank() == 0
            for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{n_epochs}: ", disable=not is_main_process):
                self._train_step(batch=batch)
            train_metrics: dict[str, float] = self._finish_training_epoch()

            # Save checkpoint after each epoch
            if checkpoint_saver is not None and epoch % save_frequency == 0 and is_main_process:
                checkpoint_saver.save(
                    model_states=self.net.state_dict(), filename=f"{self.model_name}{epoch}.pt",
                    scheduler_configs=self.scheduler_configs,
                )

            # Evaluate
            val_metrics: dict[str, float] = self.evaluate()
            assert "watched_metric" in val_metrics.keys()
            timer.end_epoch(epoch=epoch)
            # Log
            if is_main_process:
                logger.log(
                    epoch=epoch, n_epochs=n_epochs, batch=None, n_batches=None, took=timer.time_epoch(epoch),
                    **train_metrics, **val_metrics,
                )

            # Check early-stopping
            early_stopping(value=val_metrics["watched_metric"])
            if early_stopping:
                print("Early Stopped")
                break

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        """
        Baseline models only need to output: mae
        VAE models output: kl_divergence, reconstruction_loss, negative_elbo, reconstruction_mae
        Diffusion models output: velocity_loss, velocity_mae
        """
        pass

    @abstractmethod
    def _train_step(self, batch: DataBatch) -> None:
        pass

    @abstractmethod
    def _eval_step(self, batch: DataBatch) -> torch.Tensor | tuple[torch.Tensor, ...]:
        pass


class BaselineTrainer(_AbstractTrainer):

    def __init__(
        self,
        net: CNN | UNet,
        lr: float,
        train_dataset: CESM2 | ERA5, val_dataset: CESM2 | ERA5,
        train_batch_size: int, val_batch_size: int,
        local_rank: int,
    ) -> None:
        super().__init__(
            net=net, lr=lr, train_dataset=train_dataset, val_dataset=val_dataset,
            train_batch_size=train_batch_size, val_batch_size=val_batch_size,
            local_rank=local_rank,
        )
        self.loss_function: nn.Module = nn.MSELoss(reduction="mean")

    #implement
    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        val_metrics = Accumulator[torch.Tensor]()
        self.net.eval()
        n: int = 0
        for n, batch in enumerate(self.val_dataloader, start=1):
            batch_mae: torch.Tensor = self._eval_step(batch=batch)
            # Accumulate the val_metrics
            val_metrics.add(batch_mae=batch_mae)

        if n == 0:
            raise ValueError("Validation dataloader must contain at least one batch")

        mae: torch.Tensor = val_metrics["batch_mae"] / n
        dist.all_reduce(tensor=mae, op=dist.ReduceOp.AVG)
        return {"mae": mae.item(), "watched_metric": mae.item()}

    #implement
    def _train_step(self, batch: DataBatch) -> None:
        sampleinfos, input_yearday_indices, output_yearday_indices, input_tensor, groundtruth_tensor = batch
        input_tensor = input_tensor.to(self.device, non_blocking=True)
        groundtruth_tensor = groundtruth_tensor.to(self.device, non_blocking=True)
        # Forward pass
        self.optimizer.zero_grad()
        prediction_tensor: torch.Tensor = self.net(input=input_tensor)

        # Backward pass
        assert prediction_tensor.shape == groundtruth_tensor.shape
        mean_mse: torch.Tensor = self.loss_function(input=prediction_tensor, target=groundtruth_tensor)
        mean_mse.backward()
        self.optimizer.step()

    #implement
    def _eval_step(self, batch: DataBatch) -> torch.Tensor:
        sampleinfos, input_yearday_indices, output_yearday_indices, input_tensor, groundtruth_tensor = batch
        # Forward pass
        input_tensor = input_tensor.to(self.device, non_blocking=True)
        groundtruth_tensor = groundtruth_tensor.to(self.device, non_blocking=True)
        prediction_tensor: torch.Tensor = self.net(input=input_tensor)

        # Compute evaluation metrics
        assert prediction_tensor.shape == groundtruth_tensor.shape
        mean_mae: torch.Tensor = self.mae(input=prediction_tensor, target=groundtruth_tensor)
        return mean_mae


class VAETrainer(_AbstractTrainer):

    def __init__(
        self,
        net: VAE, lambda_: float, lr: float,
        train_dataset: CESM2 | ERA5, val_dataset: CESM2 | ERA5,
        train_batch_size: int, val_batch_size: int,
        local_rank: int,
    ) -> None:
        super().__init__(
            net=net, lr=lr, train_dataset=train_dataset, val_dataset=val_dataset,
            train_batch_size=train_batch_size, val_batch_size=val_batch_size,
            local_rank=local_rank,
        )
        self.lambda_: float = lambda_
        self.loss_function: nn.Module = VAELoss(lambda_=self.lambda_)

    #implement
    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        val_metrics = Accumulator[torch.Tensor]()
        self.net.eval()
        n: int = 0
        for n, batch in enumerate(self.val_dataloader, start=1):
            (
                reconstruction_loss,
                kl_divergence,
                negative_elbo,
                reconstruction_mae,
                posterior_count,
                mu_channel_sum,
                mu_sq_channel_sum,
                sigma_channel_sum,
                sigma_sq_channel_sum,
            ) = self._eval_step(batch=batch)
            val_metrics.add(
                reconstruction_loss=reconstruction_loss,
                kl_divergence=kl_divergence,
                negative_elbo=negative_elbo,
                reconstruction_mae=reconstruction_mae,
                posterior_count=posterior_count,
                mu_channel_sum=mu_channel_sum,
                mu_sq_channel_sum=mu_sq_channel_sum,
                sigma_channel_sum=sigma_channel_sum,
                sigma_sq_channel_sum=sigma_sq_channel_sum,
            )

        if n == 0:
            raise ValueError("Validation dataloader must contain at least one batch")

        reconstruction_loss = val_metrics["reconstruction_loss"] / n
        kl_divergence = val_metrics["kl_divergence"] / n
        negative_elbo = val_metrics["negative_elbo"] / n
        reconstruction_mae = val_metrics["reconstruction_mae"] / n
        for metric in (
            reconstruction_loss,
            kl_divergence,
            negative_elbo,
            reconstruction_mae,
        ):
            dist.all_reduce(tensor=metric, op=dist.ReduceOp.AVG)

        posterior_count = val_metrics["posterior_count"]
        mu_channel_sum = val_metrics["mu_channel_sum"]
        mu_sq_channel_sum = val_metrics["mu_sq_channel_sum"]
        sigma_channel_sum = val_metrics["sigma_channel_sum"]
        sigma_sq_channel_sum = val_metrics["sigma_sq_channel_sum"]
        for metric in (
            posterior_count,
            mu_channel_sum,
            mu_sq_channel_sum,
            sigma_channel_sum,
            sigma_sq_channel_sum,
        ):
            dist.all_reduce(tensor=metric, op=dist.ReduceOp.SUM)

        if posterior_count.item() <= 0:
            raise ValueError("VAE posterior statistics require at least one latent value")

        mu_channel_mean = mu_channel_sum / posterior_count
        mu_channel_rms = torch.sqrt(mu_sq_channel_sum / posterior_count)
        sigma_channel_mean = sigma_channel_sum / posterior_count
        sigma_channel_rms = torch.sqrt(sigma_sq_channel_sum / posterior_count)
        latent_channel_variance = (
            (mu_sq_channel_sum + sigma_sq_channel_sum) / posterior_count
            - mu_channel_mean.square()
        ).clamp_min(0)
        latent_channel_std = torch.sqrt(latent_channel_variance)

        n_channels = mu_channel_sum.numel()
        total_count = posterior_count * n_channels
        mu = mu_channel_sum.sum() / total_count
        sigma = sigma_channel_sum.sum() / total_count
        mu_rms = torch.sqrt(mu_sq_channel_sum.sum() / total_count)
        sigma_rms = torch.sqrt(sigma_sq_channel_sum.sum() / total_count)
        latent_variance = (
            (mu_sq_channel_sum.sum() + sigma_sq_channel_sum.sum()) / total_count
            - mu.square()
        ).clamp_min(0)
        latent_std = torch.sqrt(latent_variance)

        if dist.get_rank() == 0:
            def format_channels(values: torch.Tensor) -> str:
                return ", ".join(f"{value:.4e}" for value in values.cpu().tolist())

            print(f"posterior_mu_mean_by_channel: [{format_channels(mu_channel_mean)}]")
            print(f"posterior_mu_rms_by_channel: [{format_channels(mu_channel_rms)}]")
            print(f"posterior_sigma_mean_by_channel: [{format_channels(sigma_channel_mean)}]")
            print(f"posterior_sigma_rms_by_channel: [{format_channels(sigma_channel_rms)}]")
            print(f"posterior_latent_std_by_channel: [{format_channels(latent_channel_std)}]")

        return {
            "reconstruction_loss": reconstruction_loss.item(),
            "kl_divergence": kl_divergence.item(),
            "negative_elbo": negative_elbo.item(),
            "reconstruction_mae": reconstruction_mae.item(),
            "mu": mu.item(),
            "sigma": sigma.item(),
            "mu_rms": mu_rms.item(),
            "sigma_rms": sigma_rms.item(),
            "latent_std": latent_std.item(),
            "watched_metric": reconstruction_mae.item(),
        }

    #implement
    def _train_step(self, batch: DataBatch) -> None:
        sampleinfos, input_yearday_indices, output_yearday_indices, input_tensor, target_tensor = batch
        batch_size, n_days, H, W, n_features = input_tensor.shape
        x: torch.Tensor = input_tensor.to(self.device, non_blocking=True)

        for day in range(x.shape[1]):
            # Forward pass
            self.optimizer.zero_grad()
            true_x: torch.Tensor = x[:, day: day+1, :, :, :]
            reconstructed_x, mu, logvar = self.net(true_x)
            reconstruction_loss, kl_divergence, negative_elbo, reconstruction_mae, mu, sigma = self.loss_function(
                x_hat=reconstructed_x, true_x=true_x, mu=mu, logvar=logvar,
            )
            # Backward pass
            negative_elbo.backward()
            self.optimizer.step()

    #implement
    def _eval_step(self, batch: DataBatch) -> tuple[torch.Tensor, ...]:
        _, _, _, input_tensor, _ = batch
        x = input_tensor.to(self.device, non_blocking=True)

        reconstruction_loss_sum = torch.zeros((), device=x.device, dtype=x.dtype)
        kl_divergence_sum = torch.zeros((), device=x.device, dtype=x.dtype)
        negative_elbo_sum = torch.zeros((), device=x.device, dtype=x.dtype)
        reconstruction_mae_sum = torch.zeros((), device=x.device, dtype=x.dtype)
        posterior_count = torch.zeros((), device=x.device, dtype=torch.float64)
        mu_channel_sum: torch.Tensor | None = None
        mu_sq_channel_sum: torch.Tensor | None = None
        sigma_channel_sum: torch.Tensor | None = None
        sigma_sq_channel_sum: torch.Tensor | None = None

        for day in range(x.shape[1]):
            true_x = x[:, day: day+1, :, :, :]
            reconstructed_x, posterior_mu, logvar = self.net(true_x)
            (
                reconstruction_loss,
                kl_divergence,
                negative_elbo,
                reconstruction_mae,
                _,
                _,
            ) = self.loss_function(
                x_hat=reconstructed_x,
                true_x=true_x,
                mu=posterior_mu,
                logvar=logvar,
            )
            reconstruction_loss_sum += reconstruction_loss
            kl_divergence_sum += kl_divergence
            negative_elbo_sum += negative_elbo
            reconstruction_mae_sum += reconstruction_mae

            posterior_sigma = torch.exp(0.5 * logvar)
            reduce_dims = (0, 2, 3)
            if mu_channel_sum is None:
                n_channels = posterior_mu.shape[1]
                mu_channel_sum = torch.zeros(
                    n_channels, device=x.device, dtype=torch.float64,
                )
                mu_sq_channel_sum = torch.zeros_like(mu_channel_sum)
                sigma_channel_sum = torch.zeros_like(mu_channel_sum)
                sigma_sq_channel_sum = torch.zeros_like(mu_channel_sum)

            mu_channel_sum += posterior_mu.sum(
                dim=reduce_dims, dtype=torch.float64,
            )
            assert mu_sq_channel_sum is not None
            assert sigma_channel_sum is not None
            assert sigma_sq_channel_sum is not None
            mu_sq_channel_sum += posterior_mu.square().sum(
                dim=reduce_dims, dtype=torch.float64,
            )
            sigma_channel_sum += posterior_sigma.sum(
                dim=reduce_dims, dtype=torch.float64,
            )
            sigma_sq_channel_sum += posterior_sigma.square().sum(
                dim=reduce_dims, dtype=torch.float64,
            )
            posterior_count += posterior_mu.numel() // posterior_mu.shape[1]

        if mu_channel_sum is None:
            raise ValueError("VAE batch must contain at least one day")
        assert mu_sq_channel_sum is not None
        assert sigma_channel_sum is not None
        assert sigma_sq_channel_sum is not None

        n_days = x.shape[1]
        return (
            reconstruction_loss_sum / n_days,
            kl_divergence_sum / n_days,
            negative_elbo_sum / n_days,
            reconstruction_mae_sum / n_days,
            posterior_count,
            mu_channel_sum,
            mu_sq_channel_sum,
            sigma_channel_sum,
            sigma_sq_channel_sum,
        )



class DiffusionTrainer(RequireVAEEncoders, _AbstractTrainer):

    def __init__(
        self,
        denoiser: Denoiser,
        wind_encoder: VAEEncoder, mass_encoder: VAEEncoder,
        thermal_encoder: VAEEncoder, hydro_encoder: VAEEncoder,
        precip_encoder: VAEEncoder,
        noise_scheduler: CosineNoiseScheduler, lr: float,
        train_dataset: CESM2 | ERA5, val_dataset: CESM2 | ERA5,
        train_batch_size: int, val_batch_size: int,
        local_rank: int,
    ) -> None:
        super().__init__(
            net=denoiser, lr=lr, train_dataset=train_dataset, val_dataset=val_dataset,
            train_batch_size=train_batch_size, val_batch_size=val_batch_size,
            local_rank=local_rank,
        )
        self.denoiser: Denoiser = denoiser
        self.loss_function = DiffusionLoss()

        # NOTE: no need to wrap VAEEncoder(s) with DDP because no backprop
        # Freeze wind_encoder
        self.wind_encoder: VAEEncoder = wind_encoder.to(self.device)
        self.wind_encoder.freeze_all()
        assert wind_encoder.is_all_frozen()
        # Freeze mass_encoder
        self.mass_encoder: VAEEncoder = mass_encoder.to(self.device)
        self.mass_encoder.freeze_all()
        assert self.mass_encoder.is_all_frozen()
        # Freeze thermal_encoder
        self.thermal_encoder: VAEEncoder = thermal_encoder.to(self.device)
        self.thermal_encoder.freeze_all()
        assert self.thermal_encoder.is_all_frozen()
        # Freeze hydro_encoder
        self.hydro_encoder: VAEEncoder = hydro_encoder.to(self.device)
        self.hydro_encoder.freeze_all()
        assert self.hydro_encoder.is_all_frozen()
        # Freeze precip_encoder
        self.precip_encoder: VAEEncoder = precip_encoder.to(self.device)
        self.precip_encoder.freeze_all()
        assert self.precip_encoder.is_all_frozen()

        if dist.get_rank() == 0:
            del self.param_counter  # inheritted from _AbstractPredictor
            self.denoiser_param_counter = ParamCounter(self.denoiser)
            print(self.denoiser_param_counter.summary())
            self.wind_encoder_param_counter = ParamCounter(self.wind_encoder)
            print(self.wind_encoder_param_counter.summary())
            self.mass_encoder_param_counter = ParamCounter(self.mass_encoder)
            print(self.mass_encoder_param_counter.summary())
            self.thermal_encoder_param_counter = ParamCounter(self.thermal_encoder)
            print(self.thermal_encoder_param_counter.summary())
            self.hydro_encoder_param_counter = ParamCounter(self.hydro_encoder)
            print(self.hydro_encoder_param_counter.summary())
            self.precip_encoder_param_counter = ParamCounter(self.precip_encoder)
            print(self.precip_encoder_param_counter.summary())

        self.noise_scheduler: CosineNoiseScheduler = noise_scheduler.to(self.device)
        self.n_denoising_steps: int = noise_scheduler.n_steps
        self.forward_process: ForwardProcess = ForwardProcess(noise_scheduler=noise_scheduler)
        self.scheduler_configs["diffusion_schedule"] = self.noise_scheduler.to_config()
        self.indices_by_context_group = self.train_dataset.indices_by_context_group
        self._train_metrics = Accumulator[torch.Tensor]()
        self._n_train_batches: int = 0

    def _start_training_epoch(self) -> None:
        self._train_metrics.reset()
        self._n_train_batches = 0

    def _summarize_diffusion_metrics(
        self,
        metrics: Accumulator[torch.Tensor],
        n_batches: int,
        metric_prefix: str,
        channel_log_prefix: str,
    ) -> dict[str, float]:
        if n_batches == 0:
            raise ValueError("Dataloader must contain at least one batch")

        reduced_scalars: dict[str, torch.Tensor] = {}
        for name in (
            "velocity_loss",
            "velocity_mae",
            "zero_velocity_loss",
            "zero_velocity_mae",
        ):
            value = metrics[name] / n_batches
            dist.all_reduce(tensor=value, op=dist.ReduceOp.AVG)
            reduced_scalars[name] = value

        target_latent_count = metrics["target_latent_count"]
        target_latent_channel_sum = metrics["target_latent_channel_sum"]
        target_latent_channel_sq_sum = metrics["target_latent_channel_sq_sum"]
        for value in (
            target_latent_count,
            target_latent_channel_sum,
            target_latent_channel_sq_sum,
        ):
            dist.all_reduce(tensor=value, op=dist.ReduceOp.SUM)

        if target_latent_count.item() <= 0:
            raise ValueError("Target-latent statistics require at least one value")

        target_latent_channel_mean = (
            target_latent_channel_sum / target_latent_count
        )
        target_latent_channel_variance = (
            target_latent_channel_sq_sum / target_latent_count
            - target_latent_channel_mean.square()
        ).clamp_min(0)
        target_latent_channel_std = torch.sqrt(target_latent_channel_variance)

        n_channels = target_latent_channel_sum.numel()
        target_latent_total_count = target_latent_count * n_channels
        target_latent_mean = (
            target_latent_channel_sum.sum() / target_latent_total_count
        )
        target_latent_variance = (
            target_latent_channel_sq_sum.sum() / target_latent_total_count
            - target_latent_mean.square()
        ).clamp_min(0)
        target_latent_std = torch.sqrt(target_latent_variance)

        velocity_loss = reduced_scalars["velocity_loss"]
        velocity_mae = reduced_scalars["velocity_mae"]
        zero_velocity_loss = reduced_scalars["zero_velocity_loss"]
        zero_velocity_mae = reduced_scalars["zero_velocity_mae"]
        if zero_velocity_loss.item() <= 0 or zero_velocity_mae.item() <= 0:
            raise ValueError("Zero-velocity baseline must be positive")

        # Positive skill means the denoiser beats predicting zero velocity.
        velocity_loss_skill = 1 - velocity_loss / zero_velocity_loss
        velocity_mae_skill = 1 - velocity_mae / zero_velocity_mae

        if dist.get_rank() == 0:
            def format_channels(values: torch.Tensor) -> str:
                return ", ".join(
                    f"{value:.4e}" for value in values.cpu().tolist()
                )

            print(
                f"{channel_log_prefix}_target_latent_mean_by_channel: "
                f"[{format_channels(target_latent_channel_mean)}]"
            )
            print(
                f"{channel_log_prefix}_target_latent_std_by_channel: "
                f"[{format_channels(target_latent_channel_std)}]"
            )

        return {
            f"{metric_prefix}velocity_loss": velocity_loss.item(),
            f"{metric_prefix}velocity_mae": velocity_mae.item(),
            f"{metric_prefix}zero_velocity_loss": zero_velocity_loss.item(),
            f"{metric_prefix}zero_velocity_mae": zero_velocity_mae.item(),
            f"{metric_prefix}velocity_loss_skill": velocity_loss_skill.item(),
            f"{metric_prefix}velocity_mae_skill": velocity_mae_skill.item(),
            f"{metric_prefix}target_latent_mean": target_latent_mean.item(),
            f"{metric_prefix}target_latent_std": target_latent_std.item(),
        }

    def _finish_training_epoch(self) -> dict[str, float]:
        return self._summarize_diffusion_metrics(
            metrics=self._train_metrics,
            n_batches=self._n_train_batches,
            metric_prefix="train_",
            channel_log_prefix="train",
        )

    #implement
    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        val_metrics = Accumulator[torch.Tensor]()
        self.net.eval()
        n: int = 0
        for n, batch in enumerate(self.val_dataloader, start=1):
            (
                velocity_loss,
                velocity_mae,
                zero_velocity_loss,
                zero_velocity_mae,
                target_latent_count,
                target_latent_channel_sum,
                target_latent_channel_sq_sum,
            ) = self._eval_step(batch=batch)
            val_metrics.add(
                velocity_loss=velocity_loss,
                velocity_mae=velocity_mae,
                zero_velocity_loss=zero_velocity_loss,
                zero_velocity_mae=zero_velocity_mae,
                target_latent_count=target_latent_count,
                target_latent_channel_sum=target_latent_channel_sum,
                target_latent_channel_sq_sum=target_latent_channel_sq_sum,
            )

        summarized_metrics = self._summarize_diffusion_metrics(
            metrics=val_metrics,
            n_batches=n,
            metric_prefix="",
            channel_log_prefix="validation",
        )
        summarized_metrics["watched_metric"] = summarized_metrics["velocity_mae"]
        return summarized_metrics

    #implement
    def _train_step(self, batch: DataBatch) -> None:
        _, input_yearday_indices, output_yearday_indices, condition, target = batch
        condition = condition.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        input_yearday_indices = input_yearday_indices.to(self.device, non_blocking=True)
        output_yearday_indices = output_yearday_indices.to(self.device, non_blocking=True)
        # Reset gradients
        self.optimizer.zero_grad()
        # Forward propagation
        (
            velocity_loss,
            velocity_mae,
            zero_velocity_loss,
            zero_velocity_mae,
            target_latent_count,
            target_latent_channel_sum,
            target_latent_channel_sq_sum,
        ) = self._forward_pass(
            target=target, condition=condition,
            condition_days=input_yearday_indices,
            target_days=output_yearday_indices,
        )
        # Back propagation
        velocity_loss.backward()
        self.optimizer.step()
        self._train_metrics.add(
            velocity_loss=velocity_loss.detach(),
            velocity_mae=velocity_mae.detach(),
            zero_velocity_loss=zero_velocity_loss.detach(),
            zero_velocity_mae=zero_velocity_mae.detach(),
            target_latent_count=target_latent_count,
            target_latent_channel_sum=target_latent_channel_sum,
            target_latent_channel_sq_sum=target_latent_channel_sq_sum,
        )
        self._n_train_batches += 1

    #implement
    def _eval_step(self, batch: DataBatch) -> tuple[torch.Tensor, ...]:
        _, input_yearday_indices, output_yearday_indices, condition, target = batch
        condition = condition.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        input_yearday_indices = input_yearday_indices.to(self.device, non_blocking=True)
        output_yearday_indices = output_yearday_indices.to(self.device, non_blocking=True)
        # Forward propagation
        metrics = self._forward_pass(
            target=target, condition=condition,
            condition_days=input_yearday_indices,
            target_days=output_yearday_indices,
        )
        return metrics

    def _forward_pass(
        self,
        condition: torch.Tensor,
        target: torch.Tensor,
        condition_days: torch.Tensor,
        target_days: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # Encode (already @torch.no_grad())
        condition_latent, target_latent = self.vae_encode(condition=condition, target=target)
        batch_size: int = target_latent.shape[0]

        # Generate step
        # Diffusion step must range from 1 to K
        integer_step: torch.Tensor = torch.randint(
            low=1, high=self.n_denoising_steps + 1, size=(batch_size, 1), device=target_latent.device,
        )
        # Forward process
        noisy_target, true_velocity = self.forward_process.add_noise(
            original_latent=target_latent, k=integer_step
        )
        # Predict gaussian using Denoiser
        predicted_velocity: torch.Tensor = self.net(
            target=noisy_target,
            condition=condition_latent,
            integer_step=integer_step,
            condition_days=condition_days,
            target_days=target_days,
        )
        # Loss
        velocity_loss, velocity_mae = self.loss_function(
            velocity_hat=predicted_velocity, velocity_true=true_velocity,
        )
        zero_velocity_loss = true_velocity.square().mean()
        zero_velocity_mae = true_velocity.abs().mean()

        target_latent_reduce_dims = (0, 1, 3, 4)
        target_latent_count = torch.tensor(
            target_latent.shape[0]
            * target_latent.shape[1]
            * target_latent.shape[3]
            * target_latent.shape[4],
            device=target_latent.device,
            dtype=torch.float64,
        )
        target_latent_channel_sum = target_latent.sum(
            dim=target_latent_reduce_dims,
            dtype=torch.float64,
        )
        target_latent_channel_sq_sum = target_latent.square().sum(
            dim=target_latent_reduce_dims,
            dtype=torch.float64,
        )
        return (
            velocity_loss,
            velocity_mae,
            zero_velocity_loss,
            zero_velocity_mae,
            target_latent_count,
            target_latent_channel_sum,
            target_latent_channel_sq_sum,
        )
