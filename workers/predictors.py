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
from common.metrics import ErrorMap, MAEMap, GeographicalRsquaredMap, GeographicalMAE, GeographicalMSE
from common.plotting import MetricPlotter, PredictionPlotter, DenoisingPlotter
from models.benchmarks import CNN, UNet, ViT
from models.diffusion import (
    VAE, VAE_Target, VAEEncoder, VAEDecoder, UNetDenoiser, 
    LinearNoiseScheduler, CosineNoiseScheduler, ReverseProcess
)
from .common import RequireVAEEncoders

import torch.nn.functional as F


class _AbstractPredictor(ABC):

    def __init__(self, net: CNN | UNet | ViT | VAE, dataset: CESM2, local_rank: int):
        self.net: CNN | UNet | ViT | VAE | UNetDenoiser = net
        self.dataset: CESM2 = dataset
        self.local_rank: int = local_rank
        self.indices_by_context_group = dataset.indices_by_context_group

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
            num_workers=4,
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

        self.tropical_lats: tuple[float, float] = (-25.,+25.)   # fixed
        self.mse = GeographicalMSE(tropical_lats=self.tropical_lats)
        self.mae = GeographicalMAE(tropical_lats=self.tropical_lats)
        self.model_name: str = net.name
        self.rsquared_map: GeographicalRsquaredMap = GeographicalRsquaredMap(
            n_features=self.out_features, tropical_lats=self.tropical_lats
        )
        self.mae_map: MAEMap = MAEMap(n_features=self.out_features, tropical_lats=self.tropical_lats)
        self.error_map: ErrorMap = ErrorMap(n_features=self.out_features)
        self.prediction_plotter: PredictionPlotter = PredictionPlotter()
        self.metric_plotter: MetricPlotter = MetricPlotter()
        self.landmask_reader: LandMaskReader = LandMaskReader(device="cpu")
        self.coordinates_reader: CoordinatesReader = CoordinatesReader(device="cpu")

    @torch.no_grad()
    def predict(self) -> None:
        self.net.eval()
        is_main_process: bool = dist.get_rank() == 0
        # Batch size should always be 1
        records: dict[str, list[torch.Tensor]] = {"predictions": [], "groundtruths": []}
        # Predict
        for batch in self.dataloader:
            prediction_mean, groundtruth_mean = self._predict_step(batch=batch)
            # Record for aggregate metrics
            records["groundtruths"].append(groundtruth_mean)
            records["predictions"].append(prediction_mean)

        # Compute aggregate metrics
        rsquared_frame, global_rsquared, tropical_rsquared, extratropical_rsquared = self.rsquared_map(
            predictions=records["predictions"], groundtruths=records["groundtruths"],
        )
        assert rsquared_frame.shape == (192, 288, self.out_features)

        mae_frame, global_mae, tropical_mae, extratropical_mae = self.mae_map(
            predictions=records["predictions"], groundtruths=records["groundtruths"],
        )
        assert mae_frame.shape == (192, 288, self.out_features)

        # Plot aggregate metrics
        for idx, output_name in enumerate(self.output_names):
            self.metric_plotter.plot(
                mae_frame=mae_frame[..., idx],
                global_mae=global_mae,
                tropical_mae=tropical_mae,
                extratropical_mae=extratropical_mae,
                rsquared_frame=rsquared_frame[..., idx],
                global_rsquared=global_rsquared, 
                tropical_rsquared=tropical_rsquared, 
                extratropical_rsquared=extratropical_rsquared,
                landmask=self.landmask_reader.tensor,
                tropical_lats=self.tropical_lats,
                coordinates=self.coordinates_reader.tensors,
                title=f"{output_name}: {self.dataset.metadata.start_year} - {self.dataset.metadata.end_year}",
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

    def __init__(self, net: CNN | UNet | ViT | VAE, dataset: CESM2, local_rank: int):
        super().__init__(net=net, dataset=dataset, local_rank=local_rank)
        self.in_features: int = net.in_features
        self.n_input_days: int = dataset.metadata.n_input_days
        self.n_output_days: int = dataset.metadata.n_output_days

    #implement
    def _predict_step(self, batch: DataBatch) -> tuple[torch.Tensor, torch.Tensor]:
        sampleinfos, input_indices, output_indices, input_tensor, groundtruth_tensor = batch
        input_tensor = input_tensor.to(self.device)
        groundtruth_tensor = groundtruth_tensor.to(self.device)
        input_indices = input_indices.to(self.device)
        sampleinfo: SampleInfo = sampleinfos[0] # because batch_size=1
        assert input_tensor.shape == (1, self.n_input_days, 192, 288, self.in_features)
        assert input_indices.shape == (1, self.n_input_days)

        # Forward pass
        if self.model_name in ["cnn", "unet"]:
            prediction_tensor: torch.Tensor = self.net(input=input_tensor)
        elif self.model_name == "vit":
            prediction_tensor: torch.Tensor = self.net(input=input_tensor, input_indices=input_indices)
        else:
            raise NotImplementedError(f"{self.model_name} is not implemented")

        prediction_mean: torch.Tensor = prediction_tensor.mean(dim=1, keepdim=True) # along T
        prediction_tensor = torch.cat([prediction_tensor, prediction_mean], dim=1)
        groundtruth_mean: torch.Tensor = groundtruth_tensor.mean(dim=1, keepdim=True) # along T
        groundtruth_tensor = torch.cat([groundtruth_tensor, groundtruth_mean], dim=1)
        assert prediction_tensor.shape == groundtruth_tensor.shape == (1, self.n_output_days + 1, 192, 288, self.out_features)
        assert len(sampleinfo.out_dates) == self.n_output_days
        out_dates: list[str] = sampleinfo.out_dates + ["mean"]

        for d in range(self.n_output_days + 1):
            out_date: str = out_dates[d]
            prediction_d: torch.Tensor = prediction_tensor[0, d]
            groundtruth_d: torch.Tensor = groundtruth_tensor[0, d]
            error_d: torch.Tensor = self.error_map(prediction=prediction_d, groundtruth=groundtruth_d)

            # Plotting
            for idx, output_name in enumerate(self.output_names):
                # Select by output variable
                groundtruth_frame: torch.Tensor = groundtruth_d[..., idx]
                prediction_frame: torch.Tensor = prediction_d[..., idx]
                error_frame: torch.Tensor = error_d[..., idx]
                # MSE value
                global_mse, tropical_mse, extratropical_mse = self.mse(prediction=prediction_frame, groundtruth=groundtruth_frame)
                global_mse: float = global_mse.item()
                tropical_mse: float = tropical_mse.item()
                extratropical_mse: float = extratropical_mse.item()
                # RMSE value
                global_rmse: float = global_mse ** 0.5
                tropical_rmse: float = tropical_mse ** 0.5
                extratropical_rmse: float = extratropical_mse ** 0.5
                # MAE value
                global_mae, tropical_mae, extratropical_mae = self.mae(prediction=prediction_frame, groundtruth=groundtruth_frame)
                global_mae: float = global_mae.item()
                tropical_mae: float = tropical_mae.item()
                extratropical_mae: float = extratropical_mae.item()
                # Make title
                title: str = (
                    f"{self.model_name.upper()}: {output_name} - {sampleinfo.sim_id}\n"
                    f"[In]: {sampleinfo.in_startdate} - {sampleinfo.in_enddate}\n"
                    f"[Out]: {sampleinfo.out_startdate} - {sampleinfo.out_enddate} ({out_date.capitalize()})\n"
                    f"RMSE (Global): {global_rmse:.4f}, MAE (Global): {global_mae:.4f}\n"
                    f"RMSE (Tropic): {tropical_rmse:.4f}, MAE (Tropic): {tropical_mae:.4f}\n"
                    f"RMSE (Extratropic): {extratropical_rmse:.4f}, MAE (Extratropic): {extratropical_mae:.4f}\n"
                )
                print(title)
                print("------")
                # Make file name
                filename: str = (
                    f"{self.model_name.upper()}_{output_name}_{sampleinfo.sim_id}_"
                    f"{sampleinfo.in_startdate}{sampleinfo.in_enddate}_"
                    f"{sampleinfo.out_startdate}{sampleinfo.out_enddate}_"
                    f"{out_date}.png"
                )
                filename = filename.replace("/", "")
                # Plot single frame
                self.prediction_plotter.plot(
                    groundtruth_frame=groundtruth_frame,
                    prediction_frame=prediction_frame,
                    error_frame=error_frame,
                    tropical_lats=self.tropical_lats,
                    landmask=self.landmask_reader.tensor,
                    coordinates=self.coordinates_reader.tensors,
                    title=title,
                    filename=filename,
                )

        prediction_mean = prediction_mean.squeeze(0).squeeze(0)
        groundtruth_mean = groundtruth_mean.squeeze(0).squeeze(0)
        return prediction_mean, groundtruth_mean

    #implement
    @property
    def output_names(self) -> list[str]:
        return self.dataset.metadata.output_vars


class VAEPredictor(_AbstractPredictor):

    #implement
    def _predict_step(self, batch: DataBatch) -> tuple[torch.Tensor, torch.Tensor]:
        sampleinfos, input_indices, output_indices, input_tensor, target_tensor = batch
        if isinstance(self.net.module, VAE_Target):
            x: torch.Tensor = target_tensor.to(self.device, non_blocking=True)
        else:
            x: torch.Tensor = input_tensor.to(self.device, non_blocking=True)

        sampleinfo: SampleInfo = sampleinfos[0] # because batch_size=1
        batch_size, n_days, H, W, n_features = x.shape
        reconstructions: list[torch.Tensor] = []
        groundtruths: list[torch.Tensor] = []
        for day in range(x.shape[1]):
            true_x: torch.Tensor = x[:, day: day+1, :, :, :]
            reconstructed_x, mu, logvar = self.net(true_x)
            mu_mean: torch.Tensor = mu.mean()
            sigma_mean: torch.Tensor = logvar.exp().sqrt().mean()
            # Compute metrics
            true_x = true_x.squeeze(dim=(0, 1))
            reconstructed_x = reconstructed_x.squeeze(dim=(0, 1))
            error_tensor: torch.Tensor = self.error_map(prediction=reconstructed_x, groundtruth=true_x)
            # Plotting
            reconstructions.append(reconstructed_x)
            groundtruths.append(true_x)
            for idx in range(len(self.dataset.metadata.input_vars)):
                # Select by output variable
                true_frame: torch.Tensor = true_x[..., idx]
                reconstructed_frame: torch.Tensor = reconstructed_x[..., idx]
                error_frame: torch.Tensor = error_tensor[..., idx]
                # MSE value
                global_mse, tropical_mse, extratropical_mse = self.mse(prediction=reconstructed_frame, groundtruth=true_frame)
                global_mse: float = global_mse.item()
                tropical_mse: float = tropical_mse.item()
                extratropical_mse: float = extratropical_mse.item()
                # RMSE value
                global_rmse: float = global_mse ** 0.5
                tropical_rmse: float = tropical_mse ** 0.5
                extratropical_rmse: float = extratropical_mse ** 0.5
                # MAE value
                global_mae, tropical_mae, extratropical_mae = self.mae(prediction=reconstructed_frame, groundtruth=true_frame)
                global_mae: float = global_mae.item()
                tropical_mae: float = tropical_mae.item()
                extratropical_mae: float = extratropical_mae.item()
                # Make title
                output_name: str = self.output_names[day * len(self.dataset.metadata.input_vars) + idx]
                title: str = (
                    f"{self.model_name.upper()}\n"
                    f"{output_name}\n"
                    f"{sampleinfo.sim_id}\n"
                    f"RMSE (Global): {global_rmse:.4f}, MAE (Global): {global_mae:.4f},\n"
                    f"RMSE (Tropic): {tropical_rmse:.4f}, MAE (Tropic): {tropical_mae:.4f},\n"
                    f"RMSE (Extratropic): {extratropical_rmse:.4f}, MAE (Extratropic): {extratropical_mae:.4f},\n"
                    f"mu: {mu_mean:.4f}, sigma: {sigma_mean:.4f}\n"
                    # NOTE: `mu` and `sigma` is just a proxy. they come from the latent of all the variables, not just this variable
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
                    tropical_lats=self.tropical_lats,
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


class DiffusionPredictor(RequireVAEEncoders, _AbstractPredictor):

    def __init__(
        self, 
        denoiser: UNetDenoiser,
        wind_encoder: VAEEncoder, 
        mass_encoder: VAEEncoder,
        thermal_encoder: VAEEncoder,
        hydro_encoder: VAEEncoder,
        precip_encoder: VAEEncoder,
        precip_decoder: VAEEncoder,
        noise_scheduler: LinearNoiseScheduler | CosineNoiseScheduler,
        eta: float,
        dataset: CESM2,
        local_rank: int,
    ) -> None:
        super().__init__(net=denoiser, dataset=dataset, local_rank=local_rank)
        self.denoiser: UNetDenoiser = denoiser

        # Freeze wind_encoder
        self.wind_encoder: VAEEncoder = wind_encoder.to(self.device)
        self.wind_encoder.freeze()
        assert wind_encoder.is_frozen
        # Freeze mass_encoder
        self.mass_encoder: VAEEncoder = mass_encoder.to(self.device)
        self.mass_encoder.freeze()
        assert self.mass_encoder.is_frozen
        # Freeze thermal_encoder
        self.thermal_encoder: VAEEncoder = thermal_encoder.to(self.device)
        self.thermal_encoder.freeze()
        assert self.thermal_encoder.is_frozen
        # Freeze hydro_encoder
        self.hydro_encoder: VAEEncoder = hydro_encoder.to(self.device)
        self.hydro_encoder.freeze()
        # Freeze precip_encoder
        self.precip_encoder: VAEEncoder = precip_encoder.to(self.device)
        self.precip_encoder.freeze()
        assert self.precip_encoder.is_frozen
        # Freeze precip_decoder
        self.precip_decoder: VAEDecoder = precip_decoder.to(self.device)
        self.precip_decoder.freeze()
        assert self.precip_decoder.is_frozen

        self.noise_scheduler: LinearNoiseScheduler = noise_scheduler
        self.n_denoising_steps: int = noise_scheduler.n_steps
        self.eta: float = eta
        self.reverse_process: ReverseProcess = ReverseProcess(eta=eta, noise_scheduler=noise_scheduler)
        self.denoising_plotter: DenoisingPlotter = DenoisingPlotter()

    def _predict_step(self, batch: DataBatch) -> tuple[torch.Tensor, torch.Tensor]:
        sampleinfos, condition_days, _, condition, groundtruth = batch
        condition = condition.to(device=self.device)
        groundtruth = groundtruth.to(device=self.device)
        condition_days = condition_days.to(device=self.device)
        sampleinfo: SampleInfo = sampleinfos[0] # because batch_size=1

        # Encode
        condition_mu, condition_logvar, target_latent = self.vae_encode(condition=condition, target=groundtruth)
        # Generate gaussian
        gaussian: torch.Tensor = torch.randn_like(target_latent)
        # Denoise
        target_latent_k: torch.Tensor = gaussian
        # Denoising step must range from 1 to K
        for k in tqdm(list(reversed(range(1, self.noise_scheduler.n_steps + 1))), desc=f"Sampling step: "):
            integer_step: torch.Tensor = torch.ones((1, 1), device=target_latent.device, dtype=torch.long) * k
            # Backward process
            predicted_velocity: torch.Tensor = self.net(
                target=target_latent_k, 
                condition_mu=condition_mu, condition_logvar=condition_logvar,
                integer_step=integer_step, condition_days=condition_days,
            )
            target_latent_k, target_latent_0 = self.reverse_process.sample(
                target_k=target_latent_k, predicted_velocity=predicted_velocity, k=integer_step,
            )

        # At k=0 (last denoising step), target_latent_k = target_latent_0
        assert target_latent_k.isclose(target_latent_0).all()
        # Decode target back to physical space
        L: int = target_latent_0.shape[1]
        prediction: torch.Tensor = self.precip_decoder(target_latent_0.flatten(0, 1)).transpose(0, 1)
        assert prediction.shape == (1, L, 192, 288, self.out_features)
        prediction_mean: torch.Tensor = prediction.mean(dim=1, keepdim=True)
        prediction = torch.cat([prediction, prediction_mean], dim=1)
        groundtruth_mean: torch.Tensor = groundtruth.mean(dim=1, keepdim=True)
        groundtruth = torch.cat([groundtruth, groundtruth_mean], dim=1)
        assert prediction.shape == groundtruth.shape == (1, L + 1, 192, 288, self.out_features)
        assert len(sampleinfo.out_dates) == L
        out_dates: list[str] = sampleinfo.out_dates + ["mean"]
        
        for d in range(L + 1):
            out_date: str = out_dates[d]
            prediction_d: torch.Tensor = prediction[0, d]
            groundtruth_d: torch.Tensor = groundtruth[0, d]
            error_d: torch.Tensor = self.error_map(prediction=prediction_d, groundtruth=groundtruth_d)

            # Prediction plots
            for idx, output_name in enumerate(self.output_names):
                # Select by output variable
                groundtruth_frame: torch.Tensor = groundtruth_d[..., idx]
                prediction_frame: torch.Tensor = prediction_d[..., idx]
                error_frame: torch.Tensor = error_d[..., idx]
                # MSE value
                global_mse, tropical_mse, extratropical_mse = self.mse(prediction=prediction_frame, groundtruth=groundtruth_frame)
                global_mse: float = global_mse.item()
                tropical_mse: float = tropical_mse.item()
                extratropical_mse: float = extratropical_mse.item()
                # RMSE value
                global_rmse: float = global_mse ** 0.5
                tropical_rmse: float = tropical_mse ** 0.5
                extratropical_rmse: float = extratropical_mse ** 0.5
                # MAE value
                global_mae, tropical_mae, extratropical_mae = self.mae(prediction=prediction_frame, groundtruth=groundtruth_frame)
                global_mae: float = global_mae.item()
                tropical_mae: float = tropical_mae.item()
                extratropical_mae: float = extratropical_mae.item()
                # Make title
                title: str = (
                    f"{self.model_name.upper()}: {output_name} - {sampleinfo.sim_id}\n"
                    f"[In]: {sampleinfo.in_startdate} - {sampleinfo.in_enddate}\n"
                    f"[Out]: {sampleinfo.out_startdate} - {sampleinfo.out_enddate} ({out_date.capitalize()})\n"
                    f"RMSE (Global): {global_rmse:.4f}, MAE (Global): {global_mae:.4f}\n"
                    f"RMSE (Tropic): {tropical_rmse:.4f}, MAE (Tropic): {tropical_mae:.4f}\n"
                    f"RMSE (Extratropic): {extratropical_rmse:.4f}, MAE (Extratropic): {extratropical_mae:.4f}\n"
                )
                print(title)
                print("------")
                # Make file name
                filename: str = (
                    f"{self.model_name.upper()}_{sampleinfo.sim_id}_"
                    f"{sampleinfo.in_startdate}{sampleinfo.in_enddate}_"
                    f"{sampleinfo.out_startdate}{sampleinfo.out_enddate}_"
                    f"{out_date}.png"
                )
                filename = filename.replace("/", "")
                # Plot single frame
                self.prediction_plotter.plot(
                    groundtruth_frame=groundtruth_frame,
                    prediction_frame=prediction_frame,
                    error_frame=error_frame,
                    landmask=self.landmask_reader.tensor,
                    tropical_lats=self.tropical_lats,
                    coordinates=self.coordinates_reader.tensors,
                    title=title,
                    filename=filename,
                )

        prediction_mean = prediction_mean.squeeze(0).squeeze(0)
        groundtruth_mean = groundtruth_mean.squeeze(0).squeeze(0)
        return prediction_mean, groundtruth_mean

    #implement
    @cached_property
    def output_names(self) -> list[str]:
        return self.dataset.metadata.output_vars
