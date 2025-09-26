from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
from datasets.common.utils import DataBatch
from datasets.cesm2 import CESM2
from common.utils import Accumulator, EarlyStopping, Timer, Logger, CheckpointSaver
from common.losses import VAELoss
from models.benchmarks import CNN, UNet, ViT
from models.diffusion import (
    VAE, VAEEncoder, UNetDenoiser, 
    LinearNoiseScheduler, CosineNoiseScheduler, DDPMForwardProcess,
)
from .common import RequireVAEEncoders


class _AbstractTrainer(ABC):

    def __init__(self, 
        net: CNN | UNet | ViT | VAE | UNetDenoiser,
        lr: float,
        train_dataset: CESM2,
        val_dataset: CESM2,
        train_batch_size: int,
        val_batch_size: int,
    ):
        self.net: CNN | UNet | ViT | VAE | UNetDenoiser = net
        self.model_name: str = net.name
        self.lr: float = lr
        self.train_dataset: CESM2 = train_dataset
        self.val_dataset: CESM2 = val_dataset
        self.train_batch_size: int = train_batch_size
        self.val_batch_size: int = val_batch_size

        self.local_rank: int = torch.cuda.current_device()
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.net = DistributedDataParallel(
            module=net.to(self.device), device_ids=[self.local_rank], 
            output_device=self.local_rank, broadcast_buffers=False,
        )
        self.train_sampler: DistributedSampler = DistributedSampler(dataset=train_dataset, shuffle=True)
        self.val_sampler: DistributedSampler = DistributedSampler(dataset=val_dataset, shuffle=False)

        # Dataloaders
        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=train_batch_size,
            collate_fn=CESM2.collate_fn,
            prefetch_factor=2,
            num_workers=2,
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
            num_workers=2,
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
        n_epochs: int,
        patience: int,
        tolerance: float,
        checkpoint_directory: str | None = None,
        save_frequency: int = 5,
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
            mean_val_mae: float = self.evaluate()
            timer.end_epoch(epoch=epoch)
            # Log
            if is_main_process:
                logger.log(epoch=epoch, n_epochs=n_epochs, took=timer.time_epoch(epoch), mean_val_mae=mean_val_mae)
            
            # Check early-stopping
            early_stopping(value=mean_val_mae)
            if early_stopping:
                print("Early Stopped")
                break

    def evaluate(self) -> float:
        val_metrics = Accumulator()
        self.net.eval()
        with torch.no_grad():
            for n, batch in enumerate(self.val_dataloader, start=1):
                mean_mae: float = self._eval_step(batch=batch)
                # Accumulate the val_metrics
                val_metrics.add(mean_mae=mean_mae)

        mean_mae_local: float = val_metrics["mean_mae"] / n
        local_mae: torch.Tensor = torch.tensor([mean_mae_local], device=self.device)
        dist.all_reduce(tensor=local_mae, op=dist.ReduceOp.SUM)
        mean_mae_local = local_mae.item() / dist.get_world_size()
        return mean_mae_local
    
    @abstractmethod
    def _train_step(self, batch: DataBatch) -> None:
        pass

    @abstractmethod
    def _eval_step(self, batch: DataBatch) -> float:
        pass


class BaselineTrainer(_AbstractTrainer):

    def __init__(
        self,
        net: CNN | UNet | ViT,
        lr: float,
        train_dataset: CESM2,
        val_dataset: CESM2,
        train_batch_size: int,
        val_batch_size: int,
    ) -> None:
        super().__init__(
            net=net, lr=lr, train_dataset=train_dataset, val_dataset=val_dataset, 
            train_batch_size=train_batch_size, val_batch_size=val_batch_size,
        )
        self.loss_function: nn.Module = nn.MSELoss(reduction="mean")

    #implement
    def _train_step(self, batch: DataBatch) -> None:
        sampleinfos, input_yearday_indices, output_yearday_indices, input_tensor, groundtruth_tensor = batch
        input_tensor = input_tensor.to(self.device, non_blocking=True)
        groundtruth_tensor = groundtruth_tensor.to(self.device, non_blocking=True)
        # Forward pass
        self.optimizer.zero_grad()
        if self.model_name in ["cnn", "unet"]:
            prediction_tensor: torch.Tensor = self.net(input=input_tensor)
        elif self.model_name == "vit":
            prediction_tensor: torch.Tensor = self.net(input=input_tensor, input_yearday_indices=input_yearday_indices)
        else:
            raise NotImplementedError(f"Architecture: {type(self.net)} is not implemented")

        # Backward pass
        assert prediction_tensor.shape == groundtruth_tensor.shape
        mean_mse: torch.Tensor = self.loss_function(input=prediction_tensor, target=groundtruth_tensor)
        mean_mse.backward()
        self.optimizer.step()

    #implement
    def _eval_step(self, batch: DataBatch) -> float:
        sampleinfos, input_yearday_indices, output_yearday_indices, input_tensor, groundtruth_tensor = batch
        # Forward pass
        input_tensor = input_tensor.to(self.device, non_blocking=True)
        groundtruth_tensor = groundtruth_tensor.to(self.device, non_blocking=True)
        if self.model_name in ["cnn", "unet"]:
            prediction_tensor: torch.Tensor = self.net(input=input_tensor)
        elif self.model_name == "vit":
            prediction_tensor: torch.Tensor = self.net(input=input_tensor, input_yearday_indices=input_yearday_indices)
        else:
            raise NotImplementedError(f"Architecture: {type(self.net)} is not implemented")

        # Compute evaluation metrics
        assert prediction_tensor.shape == groundtruth_tensor.shape
        mean_mae: torch.Tensor = self.mae(input=prediction_tensor, target=groundtruth_tensor)
        return mean_mae.item()


class VAETrainer(_AbstractTrainer):

    def __init__(
        self,
        net: VAE,
        lambda_: float,
        lr: float,
        train_dataset: CESM2,
        val_dataset: CESM2,
        train_batch_size: int,
        val_batch_size: int,
    ) -> None:
        super().__init__(
            net=net, lr=lr, train_dataset=train_dataset, val_dataset=val_dataset, 
            train_batch_size=train_batch_size, val_batch_size=val_batch_size,
        )
        self.lambda_: float = lambda_
        self.loss_function: nn.Module = VAELoss(lambda_=self.lambda_)

    #implement
    def _train_step(self, batch: DataBatch) -> None:
        sampleinfos, input_yearday_indices, output_yearday_indices, input_tensor, target_tensor = batch
        batch_size, n_days, H, W, n_features = input_tensor.shape
        for day in range(input_tensor.shape[1]):
            # Forward pass
            self.optimizer.zero_grad()
            true_x: torch.Tensor = input_tensor[:, day: day+1, :, :, :]
            reconstructed_x, mu, logvar = self.net(true_x)
            # Backward pass
            mean_loss: torch.Tensor = self.loss_function(
                x_hat=reconstructed_x, true_x=true_x, mu=mu, logvar=logvar,
            )
            mean_loss.backward()
            self.optimizer.step()

    #implement
    def _eval_step(self, batch: DataBatch) -> float:
        sampleinfos, input_yearday_indices, output_yearday_indices, input_tensor, target_tensor = batch
        batch_size, n_days, H, W, n_features = input_tensor.shape
        mean_mae: torch.Tensor = torch.zeros_like(input_tensor).sum()
        for day in range(input_tensor.shape[1]):
            # Forward pass
            true_x: torch.Tensor = input_tensor[:, day: day+1, :, :, :]
            reconstructed_x, mu, logvar = self.net(true_x)
            # Compute evaluation metrics
            mean_mae += self.mae(input=reconstructed_x, target=true_x)
        return mean_mae.item() / input_tensor.shape[1]


class DDPMTrainer(RequireVAEEncoders, _AbstractTrainer):

    def __init__(
        self,
        denoiser: UNetDenoiser,
        wind_encoder: VAEEncoder, 
        geopotential_encoder: VAEEncoder,
        thermaldynamic_encoder: VAEEncoder,
        precipitation_encoder: VAEEncoder,
        noise_scheduler: LinearNoiseScheduler | CosineNoiseScheduler,
        lr: float,
        train_dataset: CESM2,
        val_dataset: CESM2,
        train_batch_size: int,
        val_batch_size: int,
    ) -> None:
        super().__init__(
            net=denoiser, lr=lr, train_dataset=train_dataset, val_dataset=val_dataset, 
            train_batch_size=train_batch_size, val_batch_size=val_batch_size,
        )
        self.denoiser: UNetDenoiser = denoiser
        self.loss_function: nn.Module = nn.MSELoss(reduction="mean")

        # Freeze encoders
        wind_encoder.freeze()
        geopotential_encoder.freeze()
        thermaldynamic_encoder.freeze()
        precipitation_encoder.freeze()
        assert wind_encoder.is_frozen
        assert geopotential_encoder.is_frozen
        assert thermaldynamic_encoder.is_frozen
        assert precipitation_encoder.is_frozen

        self.wind_encoder: VAEEncoder = wind_encoder.to(self.device)
        self.geopotential_encoder: VAEEncoder = geopotential_encoder.to(self.device)
        self.thermaldynamic_encoder: VAEEncoder = thermaldynamic_encoder.to(self.device)
        self.precipitation_encoder: VAEEncoder = precipitation_encoder.to(self.device)

        self.noise_scheduler: LinearNoiseScheduler | CosineNoiseScheduler = noise_scheduler
        self.n_denoising_steps: int = noise_scheduler.n_steps
        self.forward_process: DDPMForwardProcess = DDPMForwardProcess(noise_scheduler)
        self.indices_by_context_group = self.train_dataset.indices_by_context_group

    #implement
    def _train_step(self, batch: DataBatch) -> None:
        _, _, _, condition, target = batch
        condition = condition.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        # Reset gradients
        self.optimizer.zero_grad()
        # Forward propagation
        mean_mse, mean_mae = self._forward_pass(target=target, condition=condition)
        # Back propagation
        mean_mse.backward()
        self.optimizer.step()

    #implement
    def _eval_step(self, batch: DataBatch) -> float:
        _, _, _, condition, target = batch
        condition = condition.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        # Forward propagation
        mean_mse, mean_mae = self._forward_pass(target=target, condition=condition)
        return mean_mae.item()

    def _forward_pass(self, condition: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encode
        with torch.no_grad():
            wind_latent, geopotential_latent, thermaldynamic_latent, precipitation_latent, target_latent = (
                self.vae_encode(condition=condition, target=target)
            )
        # Generate step
        batch_size: int = target_latent.shape[0]
        step: torch.Tensor = torch.randint(
            low=0, high=self.n_denoising_steps,
            size=(batch_size, 1), device=target_latent.device
        )
        # DistributedDataParallelM forward process
        noisy_target, true_gaussian = self.forward_process.add_noise(original_latent=target_latent, step=step)
        # Predict gaussian using UNetDenoiser
        predicted_gaussian: torch.Tensor = self.net(
            target=noisy_target, 
            wind_condition=wind_latent, geopotential_condition=geopotential_latent, 
            thermaldynamic_condition=thermaldynamic_latent, precipitation_condition=precipitation_latent,
            step=step,
        )
        # MSE
        mean_mse: torch.Tensor = self.loss_function(input=predicted_gaussian, target=true_gaussian)
        # MAE
        mean_mae: torch.Tensor = self.mae(input=predicted_gaussian, target=true_gaussian)
        return mean_mse, mean_mae
