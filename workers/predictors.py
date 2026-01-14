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

    def __init__(self, net: CNN | UNet | ViT | VAE | UNetDenoiser, dataset: CESM2, local_rank: int):
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
        self.mae_map: MAEMap = MAEMap(
            n_features=self.out_features, tropical_lats=self.tropical_lats
        )
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

    #implement
    def _predict_step(self, batch: DataBatch) -> tuple[torch.Tensor, torch.Tensor]:
        sampleinfos, input_indices, output_indices, input_tensor, groundtruth_tensor = batch
        input_tensor = input_tensor.to(self.device)
        groundtruth_tensor = groundtruth_tensor.to(self.device)
        input_indices = input_indices.to(self.device)
        sampleinfo: SampleInfo = sampleinfos[0] # because batch_size=1
        assert input_tensor.shape == (1, self.net.module.n_input_days, 192, 288, self.net.module.in_features)
        assert input_indices.shape == (1, self.net.module.n_input_days)

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
                f"[Out]: {sampleinfo.out_startdate} - {sampleinfo.out_enddate}\n"
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
                f"{sampleinfo.out_startdate}{sampleinfo.out_enddate}"
                f".png"
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

        return prediction_tensor, groundtruth_tensor

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
        target_encoder: VAEEncoder,
        target_decoder: VAEDecoder,
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
        # Freeze target_encoder
        self.target_encoder: VAEEncoder = target_encoder.to(self.device)
        self.target_encoder.freeze()
        assert self.target_encoder.is_frozen
        # Freeze target_decoder
        self.target_decoder: VAEDecoder = target_decoder.to(self.device)
        self.target_decoder.freeze()
        assert self.target_decoder.is_frozen

        self.noise_scheduler: LinearNoiseScheduler = noise_scheduler
        self.n_denoising_steps: int = noise_scheduler.n_steps
        self.eta: float = eta
        self.reverse_process: ReverseProcess = ReverseProcess(eta=eta, noise_scheduler=noise_scheduler)
        self.denoising_plotter: DenoisingPlotter = DenoisingPlotter()

        # DEBUG:
        from models.diffusion import ForwardProcess
        self.forward_process: ForwardProcess = ForwardProcess(noise_scheduler=noise_scheduler)

    def _predict_step(self, batch: DataBatch) -> tuple[torch.Tensor, torch.Tensor]:
        sampleinfos, condition_days, _, condition, groundtruth = batch
        condition = condition.to(device=self.device)
        groundtruth = groundtruth.to(device=self.device)
        condition_days = condition_days.to(device=self.device)
        sampleinfo: SampleInfo = sampleinfos[0] # because batch_size=1

        # Keep track denoising errors
        x0_x0_mae_values: list[float] = []
        xk_x0_mae_values: list[float] = []
        # Encode
        condition_mu, condition_logvar, target_latent = self.vae_encode(condition=condition, target=groundtruth)
        # Generate gaussian
        # DEBUG
        gaussian: torch.Tensor = torch.randn_like(target_latent)
        # gaussian, _ = self.forward_process.add_noise(
        #     original_latent=precip_latent, k=500 * torch.ones(size=(target_latent.shape[0], 1), device=self.local_rank)
        # )
        # Denoise
        target_latent_k: torch.Tensor = gaussian
        # Denoising step must range from 1 to K
        # DEBUG
        for k in tqdm(list(reversed(range(1, self.noise_scheduler.n_steps + 1))), desc=f"Sampling step: "):
        # for k in tqdm(list(reversed(range(1, 501))), desc=f"Sampling step: "):
            integer_step: torch.Tensor = torch.ones((1, 1), device=target_latent.device, dtype=torch.long) * k
            # Backward process
            # normalized_step: torch.Tensor = self.step_normalizer(integer_step=integer_step)
            predicted_velocity: torch.Tensor = self.net(
                target=target_latent_k, 
                condition_mu=condition_mu, condition_logvar=condition_logvar,
                integer_step=integer_step, condition_days=condition_days,
            )
            target_latent_k, target_latent_0 = self.reverse_process.sample(
                target_k=target_latent_k, predicted_velocity=predicted_velocity, k=integer_step,
            )
            print(f"predicted_velocity: {predicted_velocity.mean()}")
            print(f"target_latent_k: {target_latent_k.mean()}")
            print(f"target_latent_0.mean(): {target_latent_0.mean()}")
            print(f"target_latent.mean(): {target_latent.mean()}")
            print(f"target_latent_0.std(): {target_latent_0.std()}")
            print(f"target_latent.std(): {target_latent.std()}")
            print(f"|target_latent_0|: {target_latent_0.abs().mean()}")
            print(f"|target_latent|: {target_latent.abs().mean()}")
            print(f"target_latent: {target_latent.mean()}")
            x0_x0_mae: float = torch.mean(torch.abs(target_latent_0 - target_latent)).cpu().item()
            xk_x0_mae: float = torch.mean(torch.abs(target_latent_k - target_latent)).cpu().item()
            x0_x0_mae_values.append(x0_x0_mae)
            xk_x0_mae_values.append(xk_x0_mae)
            # DEBUG
            print(f"target_latent_0 - target_latent: {x0_x0_mae}")
            print(f"target_latent_k - target_latent: {xk_x0_mae}")

        # DEBUG
        # target_latent_0 = target_latent# + torch.randn_like(target_latent)
        # At k=0 (last denoising step), target_latent_k = target_latent_0
        # assert target_latent_k.isclose(target_latent_0).all()
        # DEBUG
        latent_error: torch.Tensor = torch.mean(torch.abs(target_latent_0 - target_latent))
        print(f"latent_error: {latent_error}")
        print(f"latent_abs: {target_latent.abs().mean()}")
        print(f"latent_mean: {target_latent.mean()}")
        print(f"latent_std: {target_latent.std()}")
        print(f"predicted_abs: {target_latent_0.abs().mean()}")
        print(f"predicted_mean: {target_latent_0.mean()}")
        print(f"predicted_std: {target_latent_0.std()}")
        # Decode target back to physical space
        # DEBUG
        prediction: torch.Tensor = self.target_decoder(target_latent_0)
        # prediction: torch.Tensor = self.target_decoder(target_latent_0)
        assert prediction.shape == groundtruth.shape == (1, 1, 192, 288, self.out_features)
        print(f"Mean prediction: {prediction.mean()}")
        print(f"Min prediction: {prediction.min()}")
        print(f"Max prediction: {prediction.max()}")
        print(f"Std prediction: {prediction.std()}")
        # Error map
        error: torch.Tensor = self.error_map(prediction=prediction, groundtruth=groundtruth)
        error = error.squeeze(dim=(0, 1))

        # DEBUG
        # Denoising plots
        filename: str = (
            f"{self.model_name.upper()}_{sampleinfo.sim_id}_"
            f"{sampleinfo.in_startdate}{sampleinfo.in_enddate}_"
            f"{sampleinfo.out_startdate}{sampleinfo.out_enddate}"
            f".png"
        )
        filename = filename.replace("/", "")
        noise: list[float] = torch.sqrt(1 - self.noise_scheduler.alpha_bar_schedule).cpu().tolist()[1:]
        self.denoising_plotter.plot(
            x0_x0_mae=x0_x0_mae_values, xk_x0_mae=xk_x0_mae_values, noise=noise,
            filename=filename,
        )
        # Prediction plots
        groundtruth = groundtruth.squeeze(dim=(0, 1))
        prediction = prediction.squeeze(dim=(0, 1))
        for idx, output_name in enumerate(self.output_names):
            # Select by output variable
            groundtruth_frame: torch.Tensor = groundtruth[..., idx]
            prediction_frame: torch.Tensor = prediction[..., idx]
            error_frame: torch.Tensor = error[..., idx]
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
                f"[Out]: {sampleinfo.out_startdate} - {sampleinfo.out_enddate}\n"
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
                tropical_lats=self.tropical_lats,
                coordinates=self.coordinates_reader.tensors,
                title=title,
                filename=filename,
            )

        return prediction, groundtruth

    #implement
    @cached_property
    def output_names(self) -> list[str]:
        return self.dataset.metadata.output_vars
