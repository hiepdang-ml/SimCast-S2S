from abc import ABC, abstractmethod
from typing import Iterable, Any, cast
from functools import cached_property
from itertools import product
from pathlib import Path
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from datapipeline.dataset import CESM2
from datapipeline.readers import CESM2_LandmaskReader, CESM2_CoordinatesReader
from datapipeline.dataset import ERA5
from datapipeline.readers import ERA5_LandmaskReader, ERA5_CoordinatesReader
from datapipeline.utils import DataBatch, SampleInfo
from common.metrics import ErrorMap, MAEMap, GeographicalRsquaredMap, GeographicalMAE, GeographicalMSE
from common.plotting import MetricPlotter, PredictionPlotter
from common.utils import TorchDictIO, ParamCounter
from common.configs import MetaData
from models.benchmarks import CNN, UNet, ViT
from models.diffusion import (
    VAE, VAEEncoder, VAEDecoder, UNetDenoiser,
    LinearNoiseScheduler, CosineNoiseScheduler, ReverseProcess
)
from .common import RequireVAEEncoders


class _AbstractPredictor(ABC):

    def __init__(
        self, net: CNN | UNet | ViT | VAE | UNetDenoiser,
        dataset: CESM2 | ERA5,
        landmask_path: str,
        target_path: str, local_rank: int,
    ):
        self.base_net: CNN | UNet | ViT | VAE | UNetDenoiser = net
        self.dataset: CESM2 | ERA5 = dataset
        self.landmask_path: Path = Path(landmask_path)
        self.target_path: Path = Path(target_path)
        self.local_rank: int = local_rank
        self.H, self.W = dataset.metadata.resolution

        self.device: torch.device = torch.device(f"cuda:{self.local_rank}")
        self.indices_by_context_group: dict[str, list[int]] = dataset.indices_by_context_group
        self.net: DistributedDataParallel = DistributedDataParallel(
            module=net.to(self.device), device_ids=[self.local_rank],
            output_device=self.local_rank, broadcast_buffers=False,
        )
        if local_rank == 0:
            self.param_counter = ParamCounter(self.net)
            print(self.param_counter.summary())

        self.sampler: DistributedSampler[CESM2] = DistributedSampler(dataset=dataset, shuffle=False)
        self.dataloader: DataLoader[DataBatch] = DataLoader(
            dataset=dataset,
            batch_size=1,
            collate_fn=self.dataset.collate_fn,
            prefetch_factor=2,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
            sampler=self.sampler,
        )

        if isinstance(self.base_net, (CNN, UNet, ViT)):
            self.out_features: int = self.base_net.out_features
        elif isinstance(self.base_net, VAE):
            self.out_features: int = self.base_net.pixel_dim
        elif isinstance(self.base_net, UNetDenoiser):
            self.out_features: int = 1

        self.tropical_lats: tuple[float, float] = (-25.0, 25.0)   # fixed
        self.mse = GeographicalMSE(landmask_path=landmask_path, tropical_lats=self.tropical_lats)
        self.mae = GeographicalMAE(landmask_path=landmask_path, tropical_lats=self.tropical_lats)
        self.model_name: str = self.base_net.name
        self.rsquared_map: GeographicalRsquaredMap = GeographicalRsquaredMap(
            n_features=self.out_features, landmask_path=landmask_path, tropical_lats=self.tropical_lats,
        )
        self.mae_map: MAEMap = MAEMap(
            n_features=self.out_features, landmask_path=landmask_path, tropical_lats=self.tropical_lats,
        )
        self.error_map: ErrorMap = ErrorMap(n_features=self.out_features)
        self.torchio: TorchDictIO = TorchDictIO(dirpath=target_path)
        self.prediction_plotter: PredictionPlotter = PredictionPlotter(dirpath=target_path)
        self.metric_plotter: MetricPlotter = MetricPlotter(dirpath=target_path)
        if isinstance(dataset, CESM2):
            self.landmask_reader = CESM2_LandmaskReader()
            self.coordinates_reader = CESM2_CoordinatesReader()
        else:
            self.landmask_reader = ERA5_LandmaskReader(resolution=(self.H, self.W))
            self.coordinates_reader = ERA5_CoordinatesReader(resolution=(self.H, self.W))

    @torch.no_grad()
    def predict(self) -> None:
        self.net.eval()
        is_dist: bool = dist.is_initialized()
        rank: int = dist.get_rank() if is_dist else 0
        world_size: int = dist.get_world_size() if is_dist else 1
        # Batch size should always be 1
        records: dict[str, list[torch.Tensor]] = {"predictions": [], "groundtruths": []}
        # Predict
        for batch in self.dataloader:
            prediction_mean, groundtruth_mean = self._predict_step(batch=batch)
            # Record for aggregate metrics
            records["groundtruths"].append(groundtruth_mean)
            records["predictions"].append(prediction_mean)

        if is_dist and world_size > 1:
            gathered: list[dict[str, list[torch.Tensor]]] | None = (
                [None for _ in range(world_size)] if rank == 0 else None
            )
            dist.gather_object(obj=records, object_gather_list=gathered, dst=0)
            if rank != 0:
                return  # only rank 0 computes/plots

            # Merge lists on rank 0
            merged = {"predictions": [], "groundtruths": []}
            assert gathered is not None
            for shard in gathered:
                merged["predictions"].extend(shard["predictions"])
                merged["groundtruths"].extend(shard["groundtruths"])
            records = merged

        # TODO: should save the tensors only and build a different command to plot from results saved by TorchDictIO
        # # Compute aggregate metrics (rank 0 only)
        # rsquared_frame, global_rsquared, tropical_rsquared, extratropical_rsquared = self.rsquared_map(
        #     predictions=records["predictions"], groundtruths=records["groundtruths"],
        # )
        # assert rsquared_frame.shape == (self.H, self.W, self.out_features)

        # mae_frame, global_mae, tropical_mae, extratropical_mae = self.mae_map(
        #     predictions=records["predictions"], groundtruths=records["groundtruths"],
        # )
        # assert mae_frame.shape == (self.H, self.W, self.out_features)

        # # Plot aggregate metrics
        # for idx, output_name in enumerate(self.output_names):
        #     self.metric_plotter.plot(
        #         mae_frame=mae_frame[..., idx],
        #         global_mae=global_mae,
        #         tropical_mae=tropical_mae,
        #         extratropical_mae=extratropical_mae,
        #         rsquared_frame=rsquared_frame[..., idx],
        #         global_rsquared=global_rsquared,
        #         tropical_rsquared=tropical_rsquared,
        #         extratropical_rsquared=extratropical_rsquared,
        #         landmask=self.landmask_reader.tensor,
        #         tropical_lats=self.tropical_lats,
        #         coordinates=self.coordinates_reader.tensors,
        #         title=f"{output_name}: {self.dataset.metadata.start_year} - {self.dataset.metadata.end_year}",
        #         filename=f"{self.model_name.upper()}_{output_name}_metrics.png",
        #     )

    @abstractmethod
    def _predict_step(self, batch: DataBatch) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @property
    @abstractmethod
    def output_names(self) -> list[str]:
        pass


class BaselinePredictor(_AbstractPredictor):

    def __init__(
        self,
        net: CNN | UNet | ViT | VAE,
        dataset: CESM2 | ERA5,
        landmask_path: str, target_path: str,
        local_rank: int
    ):
        super().__init__(
            net=net, dataset=dataset,
            landmask_path=landmask_path, target_path=target_path,
            local_rank=local_rank,
        )
        if isinstance(self.base_net, (CNN, UNet, ViT)):
            self.in_features: int = self.base_net.in_features
        self.n_input_days: int = dataset.metadata.n_input_days
        self.n_output_days: int = dataset.metadata.n_output_days

    #implement
    def _predict_step(self, batch: DataBatch) -> tuple[torch.Tensor, torch.Tensor]:
        sampleinfos, input_indices, output_indices, input_tensor, groundtruth_tensor = batch
        input_tensor = input_tensor.to(self.device)
        groundtruth_tensor = groundtruth_tensor.to(self.device)
        input_indices = input_indices.to(self.device)
        sampleinfo: SampleInfo = sampleinfos[0] # because batch_size=1
        assert input_tensor.shape == (1, self.n_input_days, self.H, self.W, self.in_features)
        assert input_indices.shape == (1, self.n_input_days)

        # Forward pass
        if self.model_name in ["cnn", "unet"]:
            prediction_tensor: torch.Tensor = self.net(input=input_tensor)
        elif self.model_name == "vit":
            prediction_tensor: torch.Tensor = self.net(input=input_tensor, input_indices=input_indices)
        else:
            raise NotImplementedError(f"{self.model_name} is not implemented")

        prediction_tensor = prediction_tensor.mean(dim=1, keepdim=False).squeeze(dim=0) # along T
        groundtruth_tensor = groundtruth_tensor.mean(dim=1, keepdim=False).squeeze(dim=0) # along T
        error_tensor: torch.Tensor = self.error_map(prediction=prediction_tensor, groundtruth=groundtruth_tensor)
        assert prediction_tensor.shape == groundtruth_tensor.shape == error_tensor.shape == (self.H, self.W, self.out_features)

        # Plotting
        for idx, output_name in enumerate(self.output_names):
            # Select by output variable
            groundtruth_frame: torch.Tensor = groundtruth_tensor[..., idx]
            prediction_frame: torch.Tensor = prediction_tensor[..., idx]
            error_frame: torch.Tensor = error_tensor[..., idx]
            # MSE value
            global_mse_, tropical_mse_, extratropical_mse_ = self.mse(prediction=prediction_frame, groundtruth=groundtruth_frame)
            global_mse: float = global_mse_.item()
            tropical_mse: float = tropical_mse_.item()
            extratropical_mse: float = extratropical_mse_.item()
            # RMSE value
            global_rmse: float = global_mse ** 0.5
            tropical_rmse: float = tropical_mse ** 0.5
            extratropical_rmse: float = extratropical_mse ** 0.5
            # MAE value
            global_mae_, tropical_mae_, extratropical_mae_ = self.mae(prediction=prediction_frame, groundtruth=groundtruth_frame)
            global_mae: float = global_mae_.item()
            tropical_mae: float = tropical_mae_.item()
            extratropical_mae: float = extratropical_mae_.item()
            # Save results
            result_object: dict[str, Any] = {
                "groundtruth": groundtruth_frame,
                "prediction": prediction_frame,
                "error_map": error_frame,
                "global_mse": global_mse,
                "tropical_mse": tropical_mse,
                "extratropical_mse": extratropical_mse,
                "global_rmse": global_rmse,
                "tropical_rmse": tropical_rmse,
                "extratropical_rmse": extratropical_rmse,
                "global_mae": global_mae,
                "tropical_mae": tropical_mae,
                "extratropical_mae": extratropical_mae,
                "model_name": self.model_name.upper(),
                "output_name": output_name,
                "sim_id": sampleinfo.sim_id,
                "in_startdate": sampleinfo.in_startdate,
                "in_enddate": sampleinfo.in_enddate,
                "out_startdate": sampleinfo.out_startdate,
                "out_enddate": sampleinfo.out_enddate,
                "tropical_lats": self.tropical_lats,
            }
            filename: str = (
                f"{self.model_name.upper()}_{output_name}_{sampleinfo.sim_id}_"
                f"{sampleinfo.in_startdate}{sampleinfo.in_enddate}_"
                f"{sampleinfo.out_startdate}{sampleinfo.out_enddate}.pt"
            ).replace("/", "")
            self.torchio.save(obj=result_object, filename=filename)
            print({k: v for k, v in result_object.items() if isinstance(v, (str, float, int, tuple))})

        return prediction_tensor, groundtruth_tensor

    #implement
    @property
    def output_names(self) -> list[str]:
        return self.dataset.metadata.output_vars


class VAEPredictor(_AbstractPredictor):

    #implement
    def _predict_step(self, batch: DataBatch) -> tuple[torch.Tensor, torch.Tensor]:
        sampleinfos, input_indices, output_indices, input_tensor, target_tensor = batch
        x: torch.Tensor = input_tensor.to(self.device, non_blocking=True)

        sampleinfo: SampleInfo = sampleinfos[0] # because batch_size=1
        batch_size, n_days, H, W, n_features = x.shape
        reconstructions: list[torch.Tensor] = []
        groundtruths: list[torch.Tensor] = []
        for date_idx in range(x.shape[1]):
            true_x: torch.Tensor = x[:, date_idx: date_idx+1, :, :, :]
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
            on_date: str = sampleinfo.in_dates[date_idx]
            for idx in range(len(self.dataset.metadata.input_vars)):
                # Select by output variable
                true_frame: torch.Tensor = true_x[..., idx]
                reconstructed_frame: torch.Tensor = reconstructed_x[..., idx]
                error_frame: torch.Tensor = error_tensor[..., idx]
                # MSE value
                global_mse_, tropical_mse_, extratropical_mse_ = self.mse(prediction=reconstructed_frame, groundtruth=true_frame)
                global_mse: float = global_mse_.item()
                tropical_mse: float = tropical_mse_.item()
                extratropical_mse: float = extratropical_mse_.item()
                # RMSE value
                global_rmse: float = global_mse ** 0.5
                tropical_rmse: float = tropical_mse ** 0.5
                extratropical_rmse: float = extratropical_mse ** 0.5
                # MAE value
                global_mae_, tropical_mae_, extratropical_mae_ = self.mae(prediction=reconstructed_frame, groundtruth=true_frame)
                global_mae: float = global_mae_.item()
                tropical_mae: float = tropical_mae_.item()
                extratropical_mae: float = extratropical_mae_.item()
                # Save results
                output_name: str = self.output_names[date_idx * len(self.dataset.metadata.input_vars) + idx]
                result_object: dict[str, Any] = {
                    "groundtruth": true_frame,
                    "reconstruction": reconstructed_frame,
                    "error_map": error_frame,
                    "global_mse": global_mse,
                    "tropical_mse": tropical_mse,
                    "extratropical_mse": extratropical_mse,
                    "global_rmse": global_rmse,
                    "tropical_rmse": tropical_rmse,
                    "extratropical_rmse": extratropical_rmse,
                    "global_mae": global_mae,
                    "tropical_mae": tropical_mae,
                    "extratropical_mae": extratropical_mae,
                    "mu": mu_mean,
                    "sigma": sigma_mean,
                    # NOTE: `mu` and `sigma` is just a proxy. they come from the latent of all the variables, not just this variable
                    "model_name": self.model_name.upper(),
                    "output_name": output_name,
                    "sim_id": sampleinfo.sim_id,
                    "on_date": on_date,
                    "tropical_lats": self.tropical_lats,
                }
                filename: str = f"{self.model_name.upper()}_{output_name}_{sampleinfo.sim_id}_{on_date}.pt".replace("/", "")
                self.torchio.save(obj=result_object, filename=filename)
                print({k: v for k, v in result_object.items() if isinstance(v, (str, float, int, tuple))})

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
        precip_decoder: VAEDecoder,
        noise_scheduler: LinearNoiseScheduler | CosineNoiseScheduler,
        eta: float,
        guidance_scale: float,
        dataset: CESM2 | ERA5,
        landmask_path: str,
        target_path: str,
        local_rank: int,
        ensemble_size: int,
    ) -> None:
        super().__init__(
            net=denoiser, dataset=dataset,
            landmask_path=landmask_path, target_path=target_path,
            local_rank=local_rank,
        )
        self.denoiser: UNetDenoiser = denoiser

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
        # Freeze precip_encoder
        self.precip_encoder: VAEEncoder = precip_encoder.to(self.device)
        self.precip_encoder.freeze_all()
        assert self.precip_encoder.is_all_frozen()
        # Freeze precip_decoder
        self.precip_decoder: VAEDecoder = precip_decoder.to(self.device)
        self.precip_decoder.freeze_all()
        assert self.precip_decoder.is_all_frozen()

        if local_rank == 0:
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
            self.precip_decoder_param_counter = ParamCounter(self.precip_decoder)
            print(self.precip_decoder_param_counter.summary())

        self.noise_scheduler: LinearNoiseScheduler | CosineNoiseScheduler = noise_scheduler
        self.n_denoising_steps: int = noise_scheduler.n_steps
        self.eta: float = eta
        self.guidance_scale: float = guidance_scale
        self.ensemble_size: int = ensemble_size
        assert ensemble_size > 0
        assert self.guidance_scale >= 0.
        self.reverse_process: ReverseProcess = ReverseProcess(eta=eta, noise_scheduler=noise_scheduler)

    def _predict_step(self, batch: DataBatch) -> tuple[torch.Tensor, torch.Tensor]:
        sampleinfos, condition_days, target_days, condition, groundtruth = batch
        condition = condition.to(device=self.device)
        groundtruth = groundtruth.to(device=self.device)
        condition_days = condition_days.to(device=self.device)
        target_days = target_days.to(device=self.device)
        sampleinfo: SampleInfo = sampleinfos[0] # because batch_size=1

        # Encode
        condition_latent, target_latent = self.vae_encode(condition=condition, target=groundtruth)
        condition_kept: torch.Tensor = torch.ones((1,), device=target_latent.device, dtype=torch.bool)
        condition_dropped: torch.Tensor = torch.zeros((1,), device=target_latent.device, dtype=torch.bool)
        L: int = groundtruth.shape[1]
        groundtruth: torch.Tensor = groundtruth.mean(dim=1, keepdim=False).squeeze(dim=0)

        member_predictions: list[torch.Tensor] = []
        for member_idx in range(self.ensemble_size):
            # Generate gaussian
            gaussian: torch.Tensor = torch.randn_like(target_latent)
            # Denoise
            target_latent_k: torch.Tensor = gaussian
            # Denoising step must range from 1 to K
            for k in tqdm(
                range(self.noise_scheduler.n_steps, 0, -1),
                desc=f"[Ensemble member {member_idx + 1}/{self.ensemble_size}] Sampling step: ",
            ):
                integer_step: torch.Tensor = (
                    torch.ones((1, 1), device=target_latent.device, dtype=torch.long) * k
                )
                # Backward process
                predicted_velocity_cond: torch.Tensor = self.net(
                    target=target_latent_k,
                    condition=condition_latent,
                    integer_step=integer_step,
                    condition_days=condition_days,
                    target_days=target_days,
                    condition_mask=condition_kept,
                )
                predicted_velocity_uncond: torch.Tensor = self.net(
                    target=target_latent_k,
                    condition=condition_latent,
                    integer_step=integer_step,
                    condition_days=condition_days,
                    target_days=target_days,
                    condition_mask=condition_dropped,
                )
                predicted_velocity: torch.Tensor = (
                    predicted_velocity_uncond
                    + self.guidance_scale * (predicted_velocity_cond - predicted_velocity_uncond)
                )
                target_latent_k, target_latent_0 = self.reverse_process.sample(
                    target_k=target_latent_k, predicted_velocity=predicted_velocity, k=integer_step,
                )

            # At k=0 (last denoising step), target_latent_k = target_latent_0
            assert target_latent_k.isclose(target_latent_0).all()
            # Decode target back to physical space
            prediction: torch.Tensor = self.precip_decoder(target_latent_0.flatten(0, 1))
            assert prediction.shape == (L, 1, self.H, self.W, self.out_features)
            prediction = prediction.squeeze(dim=1)
            prediction: torch.Tensor = prediction.mean(dim=0, keepdim=False)
            member_predictions.append(prediction)
            error_map: torch.Tensor = self.error_map(prediction=prediction, groundtruth=groundtruth)

            for idx, output_name in enumerate(self.output_names):
                # Select by output variable
                groundtruth_frame: torch.Tensor = groundtruth[..., idx]
                prediction_frame: torch.Tensor = prediction[..., idx]
                error_frame: torch.Tensor = error_map[..., idx]
                # MSE value
                global_mse_, tropical_mse_, extratropical_mse_ = self.mse(
                    prediction=prediction_frame, groundtruth=groundtruth_frame
                )
                global_mse: float = global_mse_.item()
                tropical_mse: float = tropical_mse_.item()
                extratropical_mse: float = extratropical_mse_.item()
                # RMSE value
                global_rmse: float = global_mse ** 0.5
                tropical_rmse: float = tropical_mse ** 0.5
                extratropical_rmse: float = extratropical_mse ** 0.5
                # MAE value
                global_mae_, tropical_mae_, extratropical_mae_ = self.mae(
                    prediction=prediction_frame, groundtruth=groundtruth_frame
                )
                global_mae: float = global_mae_.item()
                tropical_mae: float = tropical_mae_.item()
                extratropical_mae: float = extratropical_mae_.item()
                # Save member results
                prefix: str = self._make_filename_prefix(sampleinfo=sampleinfo, output_name=output_name)
                suffix: str = f"ens_{member_idx:04d}.pt"
                result_object: dict[str, Any] = {
                    "groundtruth": groundtruth_frame,
                    "prediction": prediction_frame,
                    "error_map": error_frame,
                    "global_mse": global_mse,
                    "tropical_mse": tropical_mse,
                    "extratropical_mse": extratropical_mse,
                    "global_rmse": global_rmse,
                    "tropical_rmse": tropical_rmse,
                    "extratropical_rmse": extratropical_rmse,
                    "global_mae": global_mae,
                    "tropical_mae": tropical_mae,
                    "extratropical_mae": extratropical_mae,
                    "model_name": self.model_name.upper(),
                    "output_name": output_name,
                    "sim_id": sampleinfo.sim_id,
                    "in_startdate": sampleinfo.in_startdate,
                    "in_enddate": sampleinfo.in_enddate,
                    "out_startdate": sampleinfo.out_startdate,
                    "out_enddate": sampleinfo.out_enddate,
                    "tropical_lats": self.tropical_lats,
                    "ensemble_size": self.ensemble_size,
                    "ensemble_member": member_idx + 1,
                    "ensemble_stat": "member",
                    "prefix": prefix,
                    "suffix": suffix,
                }
                self.torchio.save(obj=result_object, filename=f"{prefix}_{suffix}")
                print({k: v for k, v in result_object.items() if isinstance(v, (str, float, int, tuple))})

        stack: torch.Tensor = torch.stack(member_predictions, dim=0)
        ensemble_mean: torch.Tensor = stack.mean(dim=0, keepdim=False)
        ensemble_var: torch.Tensor = stack.var(dim=0, unbiased=True, keepdim=False)
        ensemble_std: torch.Tensor = stack.std(dim=0, unbiased=True, keepdim=False)
        ensemble_q001: torch.Tensor = stack.quantile(q=0.01, dim=0, keepdim=False)
        ensemble_q005: torch.Tensor = stack.quantile(q=0.05, dim=0, keepdim=False)
        ensemble_q010: torch.Tensor = stack.quantile(q=0.10, dim=0, keepdim=False)
        ensemble_q025: torch.Tensor = stack.quantile(q=0.25, dim=0, keepdim=False)
        ensemble_q050: torch.Tensor = stack.quantile(q=0.50, dim=0, keepdim=False)
        ensemble_q075: torch.Tensor = stack.quantile(q=0.75, dim=0, keepdim=False)
        ensemble_q090: torch.Tensor = stack.quantile(q=0.90, dim=0, keepdim=False)
        ensemble_q095: torch.Tensor = stack.quantile(q=0.95, dim=0, keepdim=False)
        ensemble_q099: torch.Tensor = stack.quantile(q=0.99, dim=0, keepdim=False)
        ensemble_error_mean: torch.Tensor = self.error_map(prediction=ensemble_mean, groundtruth=groundtruth)
        ensemble_error_q050: torch.Tensor = self.error_map(prediction=ensemble_q050, groundtruth=groundtruth)

        # Save ensemble results
        for idx, output_name in enumerate(self.output_names):
            groundtruth_frame: torch.Tensor = groundtruth[..., idx]
            ensemble_mean_frame: torch.Tensor = ensemble_mean[..., idx]
            ensemble_std_frame: torch.Tensor = ensemble_std[..., idx]
            ensemble_var_frame: torch.Tensor = ensemble_var[..., idx]
            ensemble_q001_frame: torch.Tensor = ensemble_q001[..., idx]
            ensemble_q005_frame: torch.Tensor = ensemble_q005[..., idx]
            ensemble_q010_frame: torch.Tensor = ensemble_q010[..., idx]
            ensemble_q025_frame: torch.Tensor = ensemble_q025[..., idx]
            ensemble_q050_frame: torch.Tensor = ensemble_q050[..., idx]
            ensemble_q075_frame: torch.Tensor = ensemble_q075[..., idx]
            ensemble_q090_frame: torch.Tensor = ensemble_q090[..., idx]
            ensemble_q095_frame: torch.Tensor = ensemble_q095[..., idx]
            ensemble_q099_frame: torch.Tensor = ensemble_q099[..., idx]
            error_mean_frame: torch.Tensor = ensemble_error_mean[..., idx]
            error_q050_frame: torch.Tensor = ensemble_error_q050[..., idx]
            # MSE value
            global_mse_, tropical_mse_, extratropical_mse_ = self.mse(
                prediction=ensemble_mean_frame, groundtruth=groundtruth_frame
            )
            global_mse: float = global_mse_.item()
            tropical_mse: float = tropical_mse_.item()
            extratropical_mse: float = extratropical_mse_.item()
            # RMSE value
            global_rmse: float = global_mse ** 0.5
            tropical_rmse: float = tropical_mse ** 0.5
            extratropical_rmse: float = extratropical_mse ** 0.5
            # MAE value
            global_mae_, tropical_mae_, extratropical_mae_ = self.mae(
                prediction=ensemble_mean_frame, groundtruth=groundtruth_frame
            )
            global_mae: float = global_mae_.item()
            tropical_mae: float = tropical_mae_.item()
            extratropical_mae: float = extratropical_mae_.item()

            prefix: str = self._make_filename_prefix(sampleinfo=sampleinfo, output_name=output_name)
            suffix: str = "ens_aggregate.pt"
            result_object: dict[str, Any] = {
                "groundtruth": groundtruth_frame,
                "ensemble_mean": ensemble_mean_frame,
                "ensemble_std": ensemble_std_frame,
                "ensemble_var": ensemble_var_frame,
                "ensemble_q001": ensemble_q001_frame,
                "ensemble_q005": ensemble_q005_frame,
                "ensemble_q010": ensemble_q010_frame,
                "ensemble_q025": ensemble_q025_frame,
                "ensemble_q050": ensemble_q050_frame,
                "ensemble_q075": ensemble_q075_frame,
                "ensemble_q090": ensemble_q090_frame,
                "ensemble_q095": ensemble_q095_frame,
                "ensemble_q099": ensemble_q099_frame,
                "error_mean_frame": error_mean_frame,
                "error_q050_frame": error_q050_frame,
                "global_mse": global_mse,
                "tropical_mse": tropical_mse,
                "extratropical_mse": extratropical_mse,
                "global_rmse": global_rmse,
                "tropical_rmse": tropical_rmse,
                "extratropical_rmse": extratropical_rmse,
                "global_mae": global_mae,
                "tropical_mae": tropical_mae,
                "extratropical_mae": extratropical_mae,
                "model_name": self.model_name.upper(),
                "output_name": output_name,
                "sim_id": sampleinfo.sim_id,
                "in_startdate": sampleinfo.in_startdate,
                "in_enddate": sampleinfo.in_enddate,
                "out_startdate": sampleinfo.out_startdate,
                "out_enddate": sampleinfo.out_enddate,
                "tropical_lats": self.tropical_lats,
                "ensemble_size": self.ensemble_size,
                "ensemble_member": -1,
                "ensemble_stat": "aggregate",
                "prefix": prefix,
                "suffix": suffix,
            }
            self.torchio.save(obj=result_object, filename=f"{prefix}_{suffix}")
            print({k: v for k, v in result_object.items() if isinstance(v, (str, float, int, tuple))})

        return ensemble_mean, groundtruth

    #implement
    @cached_property
    def output_names(self) -> list[str]:
        return self.dataset.metadata.output_vars

    def _make_filename_prefix(self, sampleinfo: SampleInfo, output_name: str) -> str:
        filename_prefix: str = (
            f"{self.model_name.upper()}_{output_name}_{sampleinfo.sim_id}_"
            f"{sampleinfo.in_startdate}{sampleinfo.in_enddate}_"
            f"{sampleinfo.out_startdate}{sampleinfo.out_enddate}"
        )
        return filename_prefix.replace("/", "")


class Visualizer:

    def __init__(self, metadata: MetaData, source_dir: str, target_dir: str):
        self.metadata = metadata
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.torchio = TorchDictIO(dirpath=source_dir)
        self.prediction_plotter = PredictionPlotter(dirpath=target_dir)
        self.H, self.W = self.metadata.resolution
        if self.metadata.dataset_name == "cesm2":
            self.landmask_reader = CESM2_LandmaskReader()
            self.coordinates_reader = CESM2_CoordinatesReader()
        else:
            self.landmask_reader = ERA5_LandmaskReader(resolution=(self.H, self.W))
            self.coordinates_reader = ERA5_CoordinatesReader(resolution=(self.H, self.W))

    def plot_baseline_prediction(self, filename: str) -> None:
        result_object: dict[str, torch.Tensor | float | str] = self.torchio.load(filename=filename)
        # Make title
        title: str = (
            f"{result_object['model_name']}: {result_object['output_name']} - {result_object['sim_id']}\n"
            f"[In]: {result_object['in_startdate']} - {result_object['in_enddate']}\n"
            f"[Out]: {result_object['out_startdate']} - {result_object['out_enddate']} (Mean)\n"
            f"RMSE (Global): {result_object['global_rmse']:.4f}, MAE (Global): {result_object['global_mae']:.4f}\n"
            f"RMSE (Tropic): {result_object['tropical_rmse']:.4f}, MAE (Tropic): {result_object['tropical_mae']:.4f}\n"
            f"RMSE (Extratropic): {result_object['extratropical_rmse']:.4f}, MAE (Extratropic): {result_object['extratropical_mae']:.4f}\n"
        )
        print(title)
        print("------")

        assert isinstance(result_object["tropical_lats"], tuple) and len(result_object["tropical_lats"]) == 2
        tropical_lats: tuple[float, float] = result_object["tropical_lats"]
        groundtruth_frame = cast(torch.Tensor, result_object["groundtruth"])
        prediction_frame = cast(torch.Tensor, result_object["prediction"])
        error_frame = cast(torch.Tensor, result_object["error_map"])
        self.prediction_plotter.plot(
            groundtruth_frame=groundtruth_frame,
            prediction_frame=prediction_frame,
            error_frame=error_frame,
            uncertainty_frame=None,
            tropical_lats=tropical_lats,
            landmask=self.landmask_reader.tensor,
            coordinates=self.coordinates_reader.tensors,
            title=title,
            filename=filename.replace(".pt", ".png"),
            vlim=0.04,
        )

    def plot_vae_prediction(self, filename: str) -> None:
        result_object: dict[str, torch.Tensor | float | str] = self.torchio.load(filename=filename)
        # Make title
        title: str = (
            f"{result_object['model_name']}: {result_object['output_name']} - {result_object['sim_id']}\n"
            f"[On]: {result_object['on_date']}\n"
            f"RMSE (Global): {result_object['global_rmse']:.4f}, MAE (Global): {result_object['global_mae']:.4f}\n"
            f"RMSE (Tropic): {result_object['tropical_rmse']:.4f}, MAE (Tropic): {result_object['tropical_mae']:.4f}\n"
            f"RMSE (Extratropic): {result_object['extratropical_rmse']:.4f}, MAE (Extratropic): {result_object['extratropical_mae']:.4f}\n"
            f"mu: {float(result_object['mu']):.4f}, sigma: {float(result_object['sigma']):.4f}\n"
        )
        print(title)
        print("------")
        tropical_lats_obj = result_object["tropical_lats"]
        if isinstance(tropical_lats_obj, tuple) and len(tropical_lats_obj) == 2:
            tropical_lats: tuple[float, float] = (
                float(tropical_lats_obj[0]),
                float(tropical_lats_obj[1]),
            )
        else:
            raise TypeError(f"Invalid tropical_lats value: {tropical_lats_obj}")
        groundtruth_frame = cast(torch.Tensor, result_object["groundtruth"])
        prediction_frame = cast(torch.Tensor, result_object["reconstruction"])
        error_frame = cast(torch.Tensor, result_object["error_map"])
        self.prediction_plotter.plot(
            groundtruth_frame=groundtruth_frame,
            prediction_frame=prediction_frame,
            error_frame=error_frame,
            uncertainty_frame=None,
            landmask=self.landmask_reader.tensor,
            tropical_lats=tropical_lats,
            coordinates=self.coordinates_reader.tensors,
            title=title,
            filename=filename.replace(".pt", ".png"),
            vlim=None,
        )

    def plot_diffusion_prediction(self, filename: str) -> None:
        result_object: dict[str, torch.Tensor | float | str] = self.torchio.load(filename=filename)
        if result_object["ensemble_stat"] == "member":
            ensemble_label: str = f"Ensemble Member: {result_object['ensemble_member']}/{result_object['ensemble_size']}"
            prediction_frame: torch.Tensor = result_object["prediction"]
            groundtruth_frame = None
            error_frame = None
            uncertainty_frame = None
        else:
            ensemble_label: str = "Ensemble Mean"
            prediction_frame: torch.Tensor = result_object["ensemble_mean"]
            groundtruth_frame: torch.Tensor = result_object["groundtruth"]
            error_frame: torch.Tensor = result_object["error_mean_frame"]
            uncertainty_frame: torch.Tensor = result_object["ensemble_var"]

        # Make title
        title: str = (
            f"{result_object['model_name']}: {result_object['output_name']} - {result_object['sim_id']} ({ensemble_label})\n"
            f"[In]: {result_object['in_startdate']} - {result_object['in_enddate']}\n"
            f"[Out]: {result_object['out_startdate']} - {result_object['out_enddate']} (Mean)\n"
            f"RMSE (Global): {result_object['global_rmse']:.4f}, MAE (Global): {result_object['global_mae']:.4f}\n"
            f"RMSE (Tropic): {result_object['tropical_rmse']:.4f}, MAE (Tropic): {result_object['tropical_mae']:.4f}\n"
            f"RMSE (Extratropic): {result_object['extratropical_rmse']:.4f}, MAE (Extratropic): {result_object['extratropical_mae']:.4f}\n"
        )
        print(title)
        print("------")
        # Plot single frame
        tropical_lats_obj = result_object["tropical_lats"]
        if isinstance(tropical_lats_obj, tuple) and len(tropical_lats_obj) == 2:
            tropical_lats: tuple[float, float] = (
                float(tropical_lats_obj[0]),
                float(tropical_lats_obj[1]),
            )
        else:
            raise TypeError(f"Invalid tropical_lats value: {tropical_lats_obj}")
        self.prediction_plotter.plot(
            prediction_frame=prediction_frame,
            groundtruth_frame=groundtruth_frame,
            error_frame=error_frame,
            uncertainty_frame=uncertainty_frame,
            tropical_lats=tropical_lats,
            landmask=self.landmask_reader.tensor,
            coordinates=self.coordinates_reader.tensors,
            title=title,
            filename=filename.replace(".pt", ".png"),
            vlim=0.04,
        )
