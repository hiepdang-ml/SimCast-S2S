from abc import *
from typing import *
from functools import cached_property

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from tqdm import tqdm
from cesm2.utils import DataBatch
from models import CNN, UNet, ViT, VAE
from cesm2 import CESM2
from common.utils import Accumulator, EarlyStopping, Timer, Logger, CheckpointSaver
from common.losses import VAELoss
from .utils import HasModelName


class AbstractTrainer(HasModelName, ABC):

    def __init__(self, 
        net: CNN | UNet | ViT | VAE,
        lr: float,
        train_dataset: CESM2,
        val_dataset: CESM2,
        train_batch_size: int,
        val_batch_size: int,
    ):
        self.net: CNN | UNet | ViT | VAE = net
        self.lr: float = lr
        self.train_dataset: CESM2 = train_dataset
        self.val_dataset: CESM2 = val_dataset
        self.train_batch_size: int = train_batch_size
        self.val_batch_size: int = val_batch_size

        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=CESM2.collate_fn)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=CESM2.collate_fn)
        self.__loss_function: nn.Module = None
        self.mae: nn.Module = nn.L1Loss(reduction="mean")

        if torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net).cuda()
        elif torch.cuda.device_count() == 1:
            self.net = self.net.cuda()
        else:
            raise ValueError("No GPUs are found in the system")
        
        self.optimizer = Adam(params=self.net.parameters(), lr=lr)

    @property
    def loss_function(self) -> nn.Module:
        if self.__loss_function is None:
            raise f"loss_function is not set for {self.__class__.__name__}"
        return self.__loss_function

    @loss_function.setter
    def loss_function(self, callable: nn.Module) -> None:
        self.__loss_function = callable

    def train(
        self,
        n_epochs: int,
        patience: int,
        tolerance: float,
        checkpoint_directory: Optional[str] = None,
        save_frequency: int = 5,
    ) -> None:
        early_stopping = EarlyStopping(patience, tolerance)
        timer = Timer()
        logger = Logger()
        checkpoint_saver = CheckpointSaver(model=self.net, dirpath=checkpoint_directory)

        for epoch in range(1, n_epochs + 1):
            timer.start_epoch(epoch=epoch)
            self.net.train()
            for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{n_epochs}: "):
                self._train_step(batch=batch)

            # Save checkpoint after each epoch
            if checkpoint_directory is not None and epoch % save_frequency == 0:
                checkpoint_saver.save(model_states=self.net.state_dict(), filename=f"{self.model_name}{epoch}.pt")

            # Evaluate
            mean_val_mae: float = self.evaluate()
            timer.end_epoch(epoch=epoch)
            # Log
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

        # Compute the aggregate metrics
        mean_mae: float = val_metrics["mean_mae"] / n
        return mean_mae
    
    @abstractmethod
    def _train_step(self, batch: DataBatch) -> None:
        pass

    @abstractmethod
    def _eval_step(self, batch: DataBatch) -> float:
        pass
∞

class BaselineTrainer(AbstractTrainer):

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
        sampleinfos, input_indices, output_indices, input_tensor, self_tensor, groundtruth_tensor = batch
        # Forward pass
        self.optimizer.zero_grad()
        if isinstance(self.net, (CNN, UNet)):
            prediction_tensor: torch.Tensor = self.net(input=input_tensor)
        elif isinstance(self.net, ViT):
            prediction_tensor: torch.Tensor = self.net(input=input_tensor, input_indices=input_indices)
        else:
            raise NotImplementedError(f"Architecture: {type(self.net)} is not implemented")

        # Backward pass
        assert prediction_tensor.shape == groundtruth_tensor.shape
        mean_mse: torch.Tensor = self.loss_function(input=prediction_tensor, target=groundtruth_tensor)
        mean_mse.backward()
        self.optimizer.step()

    #implement
    def _eval_step(self, batch: DataBatch) -> float:
        sampleinfos, input_indices, output_indices, input_tensor, self_tensor, groundtruth_tensor = batch
        # Forward pass
        if isinstance(self.net, (CNN, UNet)):
            prediction_tensor: torch.Tensor = self.net(input=input_tensor)
        elif isinstance(self.net, ViT):
            prediction_tensor: torch.Tensor = self.net(input=input_tensor, input_indices=input_indices)
        else:
            raise NotImplementedError(f"Architecture: {type(self.net)} is not implemented")

        # Compute evaluation metrics
        assert prediction_tensor.shape == groundtruth_tensor.shape
        mean_mae: torch.Tensor = self.mae(input=prediction_tensor, target=groundtruth_tensor)
        return mean_mae.item()


class VAETrainer(AbstractTrainer):

    def __init__(
        self,
        net: VAE,
        tp: Literal["context", "target"],
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
        self.tp: Literal["context", "target"] = tp
        self.lambda_: float = lambda_
        self.loss_function: nn.Module = VAELoss(lambda_=self.lambda_)

    #implement
    def _train_step(self, batch: DataBatch) -> None:
        sampleinfos, input_indices, output_indices, input_tensor, self_tensor, target_tensor = batch
        # Forward pass
        self.optimizer.zero_grad()
        true_x: torch.Tensor = input_tensor if self.tp.lower() == "context" else target_tensor
        batch_size, n_days, H, W, n_features = true_x.shape
        assert n_days == 1 and n_days * n_features == self.net.pixel_dim
        true_x = true_x.permute(0, 2, 3, 1, 4).flatten(start_dim=3, end_dim=4)
        reconstructed_x, mu, logvar = self.net(true_x)
        assert reconstructed_x.shape == true_x.shape
        assert mu.shape == logvar.shape
        # Backward pass
        mean_mse: torch.Tensor = self.loss_function(
            x_hat=reconstructed_x, true_x=true_x, mu=mu, logvar=logvar,
        )
        mean_mse.backward()
        self.optimizer.step()

    #implement
    def _eval_step(self, batch: DataBatch) -> float:
        sampleinfos, input_indices, output_indices, input_tensor, self_tensor, target_tensor = batch
        # Forward pass
        true_x: torch.Tensor = input_tensor if self.tp.lower() == "context" else target_tensor
        batch_size, n_days, H, W, n_features = true_x.shape
        assert n_days * n_features == self.net.pixel_dim
        true_x = true_x.permute(0, 2, 3, 1, 4).flatten(start_dim=3, end_dim=4)
        reconstructed_x, mu, logvar = self.net(true_x)
        assert reconstructed_x.shape == true_x.shape == (1, 192, 288, self.net.pixel_dim)
        # Compute evaluation metrics
        mean_mae: torch.Tensor = self.mae(input=reconstructed_x, target=true_x)
        return mean_mae.item()


    