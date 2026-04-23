from abc import ABC, abstractmethod

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
from models.benchmarks import CNN, UNet, ViT
from models.diffusion import (
    VAE, VAEEncoder, UNetDenoiser, LinearNoiseScheduler, ForwardProcess,
)
from models.diffusion.diffusion import CosineNoiseScheduler
from .common import RequireVAEEncoders


class _AbstractTrainer(ABC):

    def __init__(self,
        net: CNN | UNet | ViT | VAE | UNetDenoiser,
        lr: float,
        train_dataset: CESM2 | ERA5, val_dataset: CESM2 | ERA5,
        train_batch_size: int, val_batch_size: int,
        local_rank: int,
    ):
        self.net: CNN | UNet | ViT | VAE | UNetDenoiser = net
        self.model_name: str = net.name
        self.lr: float = lr
        self.train_dataset: CESM2 | ERA5 = train_dataset
        self.val_dataset: CESM2 | ERA5 = val_dataset
        self.train_batch_size: int = train_batch_size
        self.val_batch_size: int = val_batch_size

        self.local_rank: int = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.net = DistributedDataParallel(
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

    @property
    def loss_function(self) -> nn.Module:
        if self.__loss_function is None:
            raise RuntimeError(f"loss_function is not set for {self.__class__.__name__}")
        return self.__loss_function

    @loss_function.setter
    def loss_function(self, callable: nn.Module) -> None:
        self.__loss_function = callable

    def train(
        self,
        n_epochs: int, patience: int, tolerance: float,
        checkpoint_directory: str | None = None, save_frequency: int = 5,
    ) -> None:
        early_stopping = EarlyStopping(patience, tolerance)
        timer = Timer()
        logger = Logger()
        if checkpoint_directory:
            checkpoint_saver = CheckpointSaver(model=self.net, dirpath=checkpoint_directory)

        for epoch in range(1, n_epochs + 1):
            timer.start_epoch(epoch=epoch)
            self.net.train()
            self.train_sampler.set_epoch(epoch)

            is_main_process: bool = dist.get_rank() == 0
            for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{n_epochs}: ", disable=not is_main_process):
                self._train_step(batch=batch)

            # Save checkpoint after each epoch
            if checkpoint_directory and epoch % save_frequency == 0 and is_main_process:
                checkpoint_saver.save(model_states=self.net.state_dict(), filename=f"{self.model_name}{epoch}.pt")

            # Evaluate
            val_metrics: dict[str, float] = self.evaluate()
            assert "watched_metric" in val_metrics.keys()
            timer.end_epoch(epoch=epoch)
            # Log
            if is_main_process:
                logger.log(
                    epoch=epoch, n_epochs=n_epochs, batch=None, n_batches=None, took=timer.time_epoch(epoch),
                    **val_metrics
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
    def _eval_step(self, batch: DataBatch) -> torch.Tensor:
        pass


class BaselineTrainer(_AbstractTrainer):

    def __init__(
        self,
        net: CNN | UNet | ViT,
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
        val_metrics = Accumulator()
        self.net.eval()
        for n, batch in enumerate(self.val_dataloader, start=1):
            batch_mae: torch.Tensor = self._eval_step(batch=batch)
            # Accumulate the val_metrics
            val_metrics.add(batch_mae=batch_mae)

        mae: torch.Tensor = val_metrics["batch_mae"] / n
        dist.all_reduce(tensor=mae, op=dist.ReduceOp.AVG)
        return {"mae": mae.item(), "watched_metric": mae.item()}

    #implement
    def _train_step(self, batch: DataBatch) -> None:
        sampleinfos, input_yearday_indices, output_yearday_indices, input_tensor, groundtruth_tensor = batch
        input_yearday_indices = input_yearday_indices.to(self.device, non_blocking=True)
        input_tensor = input_tensor.to(self.device, non_blocking=True)
        groundtruth_tensor = groundtruth_tensor.to(self.device, non_blocking=True)
        # Forward pass
        self.optimizer.zero_grad()
        if self.model_name in ["cnn", "unet"]:
            prediction_tensor: torch.Tensor = self.net(input=input_tensor)
        elif self.model_name == "vit":
            prediction_tensor: torch.Tensor = self.net(input=input_tensor, input_indices=input_yearday_indices)
        else:
            raise NotImplementedError(f"Architecture: {type(self.net)} is not implemented")

        # Backward pass
        assert prediction_tensor.shape == groundtruth_tensor.shape
        mean_mse: torch.Tensor = self.loss_function(input=prediction_tensor, target=groundtruth_tensor)
        mean_mse.backward()
        self.optimizer.step()

    #implement
    def _eval_step(self, batch: DataBatch) -> torch.Tensor:
        sampleinfos, input_yearday_indices, output_yearday_indices, input_tensor, groundtruth_tensor = batch
        # Forward pass
        input_yearday_indices = input_yearday_indices.to(self.device, non_blocking=True)
        input_tensor = input_tensor.to(self.device, non_blocking=True)
        groundtruth_tensor = groundtruth_tensor.to(self.device, non_blocking=True)
        if self.model_name in ["cnn", "unet"]:
            prediction_tensor: torch.Tensor = self.net(input=input_tensor)
        elif self.model_name == "vit":
            prediction_tensor: torch.Tensor = self.net(input=input_tensor, input_indices=input_yearday_indices)
        else:
            raise NotImplementedError(f"Architecture: {type(self.net)} is not implemented")

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
        val_metrics = Accumulator()
        self.net.eval()
        for n, batch in enumerate(self.val_dataloader, start=1):
            reconstruction_loss, kl_divergence, negative_elbo, reconstruction_mae, mu, sigma = self._eval_step(batch=batch)
            # Accumulate the val_metrics
            val_metrics.add(
                reconstruction_loss=reconstruction_loss, kl_divergence=kl_divergence, negative_elbo=negative_elbo,
                reconstruction_mae=reconstruction_mae, mu=mu, sigma=sigma,
            )

        reconstruction_loss: torch.Tensor = val_metrics["reconstruction_loss"] / n
        kl_divergence: torch.Tensor = val_metrics["kl_divergence"] / n
        negative_elbo: torch.Tensor = val_metrics["negative_elbo"] / n
        reconstruction_mae: torch.Tensor = val_metrics["reconstruction_mae"] / n
        mu: torch.Tensor = val_metrics["mu"] / n
        sigma: torch.Tensor = val_metrics["sigma"] / n
        dist.all_reduce(tensor=reconstruction_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(tensor=kl_divergence, op=dist.ReduceOp.AVG)
        dist.all_reduce(tensor=negative_elbo, op=dist.ReduceOp.AVG)
        dist.all_reduce(tensor=reconstruction_mae, op=dist.ReduceOp.AVG)
        dist.all_reduce(tensor=mu, op=dist.ReduceOp.AVG)
        dist.all_reduce(tensor=sigma, op=dist.ReduceOp.AVG)
        return {
            "reconstruction_loss": reconstruction_loss.item(),
            "kl_divergence": kl_divergence.item(),
            "negative_elbo": negative_elbo.item(),
            "reconstruction_mae": reconstruction_mae.item(),
            "mu": mu.item(),
            "sigma": sigma.item(),
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
            # Backward pass
            reconstruction_loss, kl_divergence, negative_elbo, reconstruction_mae, mu, sigma = self.loss_function(
                x_hat=reconstructed_x, true_x=true_x, mu=mu, logvar=logvar,
            )
            negative_elbo.backward()
            self.optimizer.step()

    #implement
    def _eval_step(self, batch: DataBatch) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        sampleinfos, input_yearday_indices, output_yearday_indices, input_tensor, target_tensor = batch
        batch_size, n_days, H, W, n_features = input_tensor.shape
        x: torch.Tensor = input_tensor.to(self.device, non_blocking=True)

        reconstruction_loss_sum: torch.Tensor = torch.zeros_like(x).sum()
        kl_divergence_sum: torch.Tensor = torch.zeros_like(x).sum()
        negative_elbo_sum: torch.Tensor = torch.zeros_like(x).sum()
        reconstruction_mae_sum: torch.Tensor = torch.zeros_like(x).sum()
        mu_sum: torch.Tensor = torch.zeros_like(x).sum()
        sigma_sum: torch.Tensor = torch.zeros_like(x).sum()

        for day in range(x.shape[1]):
            # Forward pass
            true_x: torch.Tensor = x[:, day: day+1, :, :, :]
            reconstructed_x, mu, logvar = self.net(true_x)
            # Compute evaluation metrics
            reconstruction_loss, kl_divergence, negative_elbo, reconstruction_mae, mu, sigma  = self.loss_function(
                x_hat=reconstructed_x, true_x=true_x, mu=mu, logvar=logvar,
            )
            reconstruction_loss_sum += reconstruction_loss
            kl_divergence_sum += kl_divergence
            negative_elbo_sum += negative_elbo
            reconstruction_mae_sum += reconstruction_mae
            mu_sum += mu
            sigma_sum += sigma

        n_days: int = x.shape[1]
        return (
            reconstruction_loss_sum / n_days,
            kl_divergence_sum / n_days,
            negative_elbo_sum / n_days,
            reconstruction_mae_sum / n_days,
            mu_sum / n_days,
            sigma_sum / n_days,
        )


class DiffusionTrainer(RequireVAEEncoders, _AbstractTrainer):

    def __init__(
        self,
        denoiser: UNetDenoiser,
        wind_encoder: VAEEncoder, mass_encoder: VAEEncoder,
        thermal_encoder: VAEEncoder, hydro_encoder: VAEEncoder,
        precip_encoder: VAEEncoder,
        noise_scheduler: LinearNoiseScheduler | CosineNoiseScheduler, lr: float,
        train_dataset: CESM2 | ERA5, val_dataset: CESM2 | ERA5,
        train_batch_size: int, val_batch_size: int,
        local_rank: int,
    ) -> None:
        super().__init__(
            net=denoiser, lr=lr, train_dataset=train_dataset, val_dataset=val_dataset,
            train_batch_size=train_batch_size, val_batch_size=val_batch_size,
            local_rank=local_rank,
        )
        self.denoiser: UNetDenoiser = denoiser
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

        self.noise_scheduler: LinearNoiseScheduler = noise_scheduler.to(self.device)
        self.n_denoising_steps: int = noise_scheduler.n_steps
        self.forward_process: ForwardProcess = ForwardProcess(noise_scheduler=noise_scheduler)
        self.indices_by_context_group = self.train_dataset.indices_by_context_group

    #implement
    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        val_metrics = Accumulator()
        self.net.eval()
        for n, batch in enumerate(self.val_dataloader, start=1):
            velocity_loss, velocity_mae = self._eval_step(batch=batch)
            # Accumulate the val_metrics
            val_metrics.add(velocity_loss=velocity_loss, velocity_mae=velocity_mae)

        velocity_loss: torch.Tensor = val_metrics["velocity_loss"] / n
        velocity_mae: torch.Tensor = val_metrics["velocity_mae"] / n
        dist.all_reduce(tensor=velocity_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(tensor=velocity_mae, op=dist.ReduceOp.AVG)
        return {
            "velocity_loss": velocity_loss.item(),
            "velocity_mae": velocity_mae.item(),
            "watched_metric": velocity_mae.item(),
        }

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
        velocity_loss, velocity_mae = self._forward_pass(
            target=target, condition=condition,
            condition_days=input_yearday_indices,
            target_days=output_yearday_indices,
        )
        # Back propagation
        velocity_loss.backward()
        self.optimizer.step()

    #implement
    def _eval_step(self, batch: DataBatch) -> tuple[torch.Tensor, torch.Tensor]:
        _, input_yearday_indices, output_yearday_indices, condition, target = batch
        condition = condition.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        input_yearday_indices = input_yearday_indices.to(self.device, non_blocking=True)
        output_yearday_indices = output_yearday_indices.to(self.device, non_blocking=True)
        # Forward propagation
        velocity_loss, velocity_mae = self._forward_pass(
            target=target, condition=condition,
            condition_days=input_yearday_indices,
            target_days=output_yearday_indices,
        )
        return velocity_loss, velocity_mae

    def _forward_pass(
        self,
        condition: torch.Tensor,
        target: torch.Tensor,
        condition_days: torch.Tensor,
        target_days: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Encode (already @torch.no_grad())
        condition_mu, condition_logvar, target_latent = self.vae_encode(condition=condition, target=target)

        # Generate step
        batch_size: int = target_latent.shape[0]
        # Diffusion step must range from 1 to K
        integer_step: torch.Tensor = torch.randint(
            low=1, high=self.n_denoising_steps + 1, size=(batch_size, 1), device=target_latent.device,
        )
        # Forward process
        noisy_target, true_velocity = self.forward_process.add_noise(
            original_latent=target_latent, k=integer_step
        )
        # Predict gaussian using UNetDenoiser
        predicted_velocity: torch.Tensor = self.net(
            target=noisy_target,
            condition_mu=condition_mu, condition_logvar=condition_logvar,
            integer_step=integer_step,
            condition_days=condition_days,
            target_days=target_days,
        )
        # Loss
        velocity_loss, velocity_mae = self.loss_function(
            velocity_hat=predicted_velocity, velocity_true=true_velocity,
        )
        return velocity_loss, velocity_mae
