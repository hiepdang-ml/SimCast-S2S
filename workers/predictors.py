from abc import ABC, abstractmethod
from typing import Iterable
from functools import cached_property
from itertools import product
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from datasets.cesm2 import CESM2, CoordinatesReader, LandMaskReader
from datasets.common.utils import DataBatch, SampleInfo
from common.metrics import ErrorMap, MAEMap, RsquaredMap
from common.plotting import MetricPlotter, PredictionPlotter
from models.benchmarks import CNN, UNet, ViT
from models.diffusion import (
    VAE, VAEEncoder, VAEDecoder, UNetDenoiser, 
    LinearNoiseScheduler, CosineNoiseScheduler, DDPMReverseProcess,
)
from .common import RequireVAEEncoders


class _AbstractPredictor(ABC):

    def __init__(self, net: CNN | UNet | ViT | VAE | UNetDenoiser, dataset: CESM2):
        self.net: CNN | UNet | ViT | VAE | UNetDenoiser = net
        self.dataset: CESM2 = dataset
        self.indices_by_context_group = dataset.indices_by_context_group

        self.local_rank: int = torch.cuda.current_device()
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.net = DistributedDataParallel(
            module=net.to(self.device), device_ids=[self.local_rank], 
            output_device=self.local_rank, broadcast_buffers=False,
        )
        self.sampler: DistributedSampler = DistributedSampler(dataset=dataset, shuffle=False)
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            collate_fn=CESM2.collate_fn,
            prefetch_factor=2,
            num_workers=2,
            pin_memory=True,
            persistent_workers=False,
            sampler=self.sampler,
        )

        if isinstance(net, (CNN, UNet, ViT)):
            self.out_features: int = net.out_features
        elif isinstance(net, VAE):
            self.out_features: int = net.pixel_dim
        elif isinstance(net, UNetDenoiser):
            self.out_features: int = 1

        self.mse = nn.MSELoss(reduction='mean')
        self.mae = nn.L1Loss(reduction='mean')
        self.model_name: str = net.name
        self.rsquared_map: RsquaredMap = RsquaredMap(n_features=self.out_features)
        self.mae_map: MAEMap = MAEMap(n_features=self.out_features)
        self.error_map: ErrorMap = ErrorMap(n_features=self.out_features)
        self.prediction_plotter: PredictionPlotter = PredictionPlotter()
        self.metric_plotter: MetricPlotter = MetricPlotter()
        self.landmask_reader: LandMaskReader = LandMaskReader(device="cpu")
        self.coordinates_reader: CoordinatesReader = CoordinatesReader(device="cpu")

    def predict(self) -> None:
        self.net.eval()
        is_main_process: bool = dist.get_rank() == 0
        # Batch size should always be 1
        records: dict[str, list[torch.Tensor]] = {"predictions": [], "groundtruths": []}
        # Predict
        with torch.no_grad():
            for batch in self.dataloader:
                prediction_tensor, groundtruth_tensor = self._predict_step(batch=batch)
                # Record for aggregate metrics
                records["groundtruths"].append(groundtruth_tensor)
                records["predictions"].append(prediction_tensor)

        # Compute aggregate metrics
        rsquared_frame: torch.Tensor = self.rsquared_map(
            predictions=records["predictions"], groundtruths=records["groundtruths"],
        )
        assert rsquared_frame.shape == (192, 288, self.out_features)

        mae_frame: torch.Tensor = self.mae_map(
            predictions=records["predictions"], groundtruths=records["groundtruths"],
        )
        assert mae_frame.shape == (192, 288, self.out_features)

        # Plot aggregate metrics
        for idx, output_name in enumerate(self.output_names):
            self.metric_plotter.plot(
                mae_frame=mae_frame[..., idx],
                rsquared_frame=rsquared_frame[..., idx],
                landmask=self.landmask_reader.tensor,
                coordinates=self.coordinates_reader.tensors,
                title=f"{output_name}\n{self.dataset.metadata.start_year} - {self.dataset.metadata.end_year}",
                filename=f"{self.model_name.upper()}_{output_name}_metrics.png",
            )

    @abstractmethod
    def _predict_step(self, batch: DataBatch) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @property
    @abstractmethod
    def output_names(self) -> list[str]:
        pass


class BaselinePredictor(_AbstractPredictor):

    #implement
    def _predict_step(self, batch: DataBatch) -> tuple[torch.Tensor, torch.Tensor]:
        sampleinfos, input_indices, output_indices, input_tensor, groundtruth_tensor = batch
        sampleinfo: SampleInfo = sampleinfos[0] # because batch_size=1
        assert input_tensor.shape == (1, self.net.n_input_days, 192, 288, self.net.in_features)
        assert input_indices.shape == (1, self.net.n_input_days)

        # Forward pass
        if self.model_name in ["cnn", "unet"]:
            prediction_tensor: torch.Tensor = self.net(input=input_tensor)
        elif self.model_name == "vit":
            prediction_tensor: torch.Tensor = self.net(input=input_tensor, input_indices=input_indices)
        else:
            raise NotImplementedError(f"{self.model_name} is not implemented")

        assert prediction_tensor.shape == groundtruth_tensor.shape == (1, 1, 192, 288, self.out_features)
        # Error map
        error_tensor: torch.Tensor = self.error_map(prediction=prediction_tensor, groundtruth=groundtruth_tensor)
        error_tensor = error_tensor.squeeze(dim=(0, 1))

        # Plotting
        groundtruth_tensor = groundtruth_tensor.squeeze(dim=(0, 1))
        prediction_tensor = prediction_tensor.squeeze(dim=(0, 1))
        for idx, output_name in enumerate(self.output_names):
            # Select by output variable
            groundtruth_frame: torch.Tensor = groundtruth_tensor[..., idx]
            prediction_frame: torch.Tensor = prediction_tensor[..., idx]
            error_frame: torch.Tensor = error_tensor[..., idx]
            # MSE value
            mean_mse: float = self.mse(input=prediction_frame, target=groundtruth_frame).item()
            # RMSE value
            mean_rmse: float = mean_mse ** 0.5
            # MAE value
            mean_mae: float = self.mae(input=prediction_frame, target=groundtruth_frame).item()
            # Make title
            title: str = (
                f"{self.model_name.upper()}: {output_name} - {sampleinfo.sim_id}\n"
                f"[In]: {sampleinfo.in_startdate} - {sampleinfo.in_enddate}\n"
                f"[Out]: {sampleinfo.out_startdate} - {sampleinfo.out_enddate}\n"
                f"MSE: {mean_mse:.4f}, RMSE: {mean_rmse:.4f}, MAE: {mean_mae:.4f}\n"
            )
            print(title)
            print("------")
            # Make file name
            filename: str = (
                f"{self.model_name.upper()}_{output_name}_{sampleinfo.sim_id}_"
                f"{sampleinfo.in_startdate}{sampleinfo.in_enddate}_"
                f"{sampleinfo.out_startdate}{sampleinfo.out_enddate}"
                f".png"
            )
            filename = filename.replace("/", "")
            # Plot single frame
            self.prediction_plotter.plot(
                groundtruth_frame=groundtruth_frame,
                prediction_frame=prediction_frame,
                error_frame=error_frame,
                landmask=self.landmask_reader.tensor,
                coordinates=self.coordinates_reader.tensors,
                title=title,
                filename=filename,
            )

        return prediction_tensor, groundtruth_tensor

    #implement
    @property
    def output_names(self) -> list[str]:
        return self.dataset.metadata.output_vars


class VAEPredictor(_AbstractPredictor):

    #implement
    def _predict_step(self, batch: DataBatch) -> tuple[torch.Tensor, torch.Tensor]:
        sampleinfos, input_indices, output_indices, input_tensor, target_tensor = batch
        input_tensor = input_tensor.to(device=self.device)
        sampleinfo: SampleInfo = sampleinfos[0] # because batch_size=1
        batch_size, n_days, H, W, n_features = input_tensor.shape
        reconstructions: list[torch.Tensor] = []
        groundtruths: list[torch.Tennsor] = []
        for day in range(input_tensor.shape[1]):
            true_x: torch.Tensor = input_tensor[:, day: day+1, :, :, :]
            reconstructed_x, mu, logvar = self.net(true_x)
            # Compute metrics
            error_tensor: torch.Tensor = self.error_map(prediction=reconstructed_x, groundtruth=true_x)

            # Plotting
            true_x = true_x.squeeze(dim=(0, 1))
            reconstructed_x = reconstructed_x.squeeze(dim=(0, 1))
            error_tensor = error_tensor.squeeze(dim=(0, 1))
            reconstructions.append(reconstructed_x)
            groundtruths.append(true_x)
            for idx in range(len(self.dataset.metadata.input_vars)):
                # Select by output variable
                true_frame: torch.Tensor = true_x[..., idx]
                reconstructed_frame: torch.Tensor = reconstructed_x[..., idx]
                error_frame: torch.Tensor = error_tensor[..., idx]
                # MSE value
                mean_mse: float = self.mse(input=reconstructed_frame, target=true_frame).item()
                # RMSE value
                mean_rmse: float = mean_mse ** 0.5
                # MAE value
                mean_mae: float = self.mae(input=reconstructed_frame, target=true_frame).item()

                # Make title
                output_name: str = self.output_names[day * len(self.dataset.metadata.input_vars) + idx]
                title: str = (
                    f"{self.model_name.upper()}\n"
                    f"{output_name}\n"
                    f"{sampleinfo.sim_id}\n"
                    f"MSE: {mean_mse:.4f}, RMSE: {mean_rmse:.4f}, MAE: {mean_mae:.4f}\n"
                )
                print(title)
                print("------")

                # Make file name
                filename: str = (
                    f"{self.model_name.upper()}_{output_name}_{sampleinfo.sim_id}_"
                    f"{sampleinfo.in_startdate}{sampleinfo.in_enddate}_"
                    f"{sampleinfo.out_startdate}{sampleinfo.out_enddate}"
                    f".png"
                )
                filename = filename.replace("/", "")
                # Plot single frame
                self.prediction_plotter.plot(
                    groundtruth_frame=true_frame,
                    prediction_frame=reconstructed_frame,
                    error_frame=error_frame,
                    landmask=self.landmask_reader.tensor,
                    coordinates=self.coordinates_reader.tensors,
                    title=title,
                    filename=filename,
                )

        reconstruction: torch.Tensor = torch.stack(reconstructions, dim=0)
        groundtruth: torch.Tensor = torch.stack(groundtruths, dim=0)
        return reconstruction, groundtruth

    #implement
    @cached_property
    def output_names(self) -> list[str]:
        days: Iterable[int] = range(1, self.dataset.metadata.n_input_days + 1)
        var_names: list[str] = self.dataset.metadata.input_vars
        output_names: list[str] = [f"{item[1]}_DAY{item[0]}" for item in product(days, var_names)]
        return output_names


class DDPMPredictor(RequireVAEEncoders, _AbstractPredictor):

    def __init__(
        self, 
        denoiser: UNetDenoiser,
        wind_encoder: VAEEncoder, 
        geopotential_encoder: VAEEncoder,
        thermaldynamic_encoder: VAEEncoder,
        precipitation_encoder: VAEEncoder,
        precipitation_decoder: VAEDecoder,
        noise_scheduler: LinearNoiseScheduler | CosineNoiseScheduler,
        dataset: CESM2,
    ) -> None:
        super().__init__(net=denoiser, dataset=dataset)
        self.denoiser: UNetDenoiser = denoiser

        # Freeze wind_encoder
        self.wind_encoder: VAEEncoder = wind_encoder
        self.wind_encoder.freeze()
        assert wind_encoder.is_frozen
        # Freeze geopotential_encoder
        self.geopotential_encoder: VAEEncoder = geopotential_encoder
        self.geopotential_encoder.freeze()
        assert self.geopotential_encoder.is_frozen
        # Freeze thermaldynamic_encoder
        self.thermaldynamic_encoder: VAEEncoder = thermaldynamic_encoder
        self.thermaldynamic_encoder.freeze()
        assert self.thermaldynamic_encoder.is_frozen
        # Freeze precipitation_encoder
        self.precipitation_encoder: VAEEncoder = precipitation_encoder
        self.precipitation_encoder.freeze()
        assert self.precipitation_encoder.is_frozen
        # Freeze precipitation_decoder
        self.precipitation_decoder: VAEDecoder = precipitation_decoder
        self.precipitation_decoder.freeze()
        assert self.precipitation_decoder.is_frozen

        self.noise_scheduler: LinearNoiseScheduler | CosineNoiseScheduler = noise_scheduler
        self.n_denoising_steps: int = noise_scheduler.n_steps
        self.reverse_process: DDPMReverseProcess = DDPMReverseProcess(noise_scheduler)

    def _predict_step(self, batch: DataBatch) -> tuple[torch.Tensor, torch.Tensor]:
        sampleinfos, _, _, condition, groundtruth = batch
        condition = condition.to(device=self.device)
        groundtruth = groundtruth.to(device=self.device)
        sampleinfo: SampleInfo = sampleinfos[0] # because batch_size=1
        # Encode
        wind_latent, geopotential_latent, thermaldynamic_latent, precipitation_latent, target_latent = (
            self.vae_encode(condition=condition, target=groundtruth)
        )
        # Generate gaussian
        gaussian: torch.Tensor = torch.randn_like(target_latent)
        # Denoise
        target_latent_k: torch.Tensor = gaussian
        for k in reversed(range(1, self.noise_scheduler.n_steps)):
            step: torch.Tensor = torch.ones((1, 1), device=target_latent.device, dtype=torch.long) * k
            # DDPM backward process
            predicted_gaussian: torch.Tensor = self.denoiser(
                target=target_latent_k, 
                wind_condition=wind_latent, geopotential_condition=geopotential_latent, 
                thermaldynamic_condition=thermaldynamic_latent, precipitation_condition=precipitation_latent,
                step=step,
            )
            target_latent_k, target_latent_0 = self.reverse_process.sample(
                target_k=target_latent_k, predicted_noise=predicted_gaussian, step=step,
            )

        # At k=0 (last denoising step), target_latent_k = target_latent_0
        assert target_latent_k.isclose(target_latent_0).all()
        # Decode target back to physical space
        prediction: torch.Tensor = self.precipitation_decoder(target_latent_0)
        assert prediction.shape == groundtruth.shape == (1, 1, 192, 288, self.out_features)
        print(f"Mean prediction: {prediction.mean().item()}")
        print(f"Min prediction: {prediction.min().item()}")
        print(f"Max prediction: {prediction.max().item()}")
        print(f"Std prediction: {prediction.std().item()}")
        # Error map
        error: torch.Tensor = self.error_map(prediction=prediction, groundtruth=groundtruth)
        error = error.squeeze(dim=(0, 1))

        # Plotting
        groundtruth = groundtruth.squeeze(dim=(0, 1))
        prediction = prediction.squeeze(dim=(0, 1))
        for idx, output_name in enumerate(self.output_names):
            # Select by output variable
            groundtruth_frame: torch.Tensor = groundtruth[..., idx]
            prediction_frame: torch.Tensor = prediction[..., idx]
            error_frame: torch.Tensor = error[..., idx]
            # MSE value
            mean_mse: float = self.mse(input=prediction_frame, target=groundtruth_frame).item()
            # RMSE value
            mean_rmse: float = mean_mse ** 0.5
            # MAE value
            mean_mae: float = self.mae(input=prediction_frame, target=groundtruth_frame).item()
            # Make title
            title: str = (
                f"{self.model_name.upper()}: {output_name} - {sampleinfo.sim_id}\n"
                f"[In]: {sampleinfo.in_startdate} - {sampleinfo.in_enddate}\n"
                f"[Out]: {sampleinfo.out_startdate} - {sampleinfo.out_enddate}\n"
                f"MSE: {mean_mse:.4f}, RMSE: {mean_rmse:.4f}, MAE: {mean_mae:.4f}\n"
            )
            print(title)
            print("------")
            # Make file name
            filename: str = (
                f"{self.model_name.upper()}_{output_name}_{sampleinfo.sim_id}_"
                f"{sampleinfo.in_startdate}{sampleinfo.in_enddate}_"
                f"{sampleinfo.out_startdate}{sampleinfo.out_enddate}"
                f".png"
            )
            filename = filename.replace("/", "")
            # Plot single frame
            self.prediction_plotter.plot(
                groundtruth_frame=groundtruth_frame,
                prediction_frame=prediction_frame,
                error_frame=error_frame,
                landmask=self.landmask_reader.tensor,
                coordinates=self.coordinates_reader.tensors,
                title=title,
                filename=filename,
            )

        return prediction, groundtruth

    #implement
    @cached_property
    def output_names(self) -> list[str]:
        return self.dataset.metadata.output_vars
