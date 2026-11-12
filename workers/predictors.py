from abc import ABC, abstractmethod
from typing import Iterable, Any, Literal, cast
from functools import cached_property
from itertools import product
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from datapipeline.dataset import CESM2
from datapipeline.readers import CESM2_LandmaskReader, CESM2_CoordinatesReader
from datapipeline.dataset import ERA5
from datapipeline.readers import ERA5_LandmaskReader, ERA5_CoordinatesReader
from datapipeline.utils import DataBatch, SampleInfo
from common.metrics import ErrorMap, MAEMap, GeographicalRsquaredMap, GeographicalMAE, GeographicalMSE
from common.plotting import PredictionPlotter
from common.utils import TorchDictIO, ParamCounter
from common.configs import MetaData
from models.baselines import CNN, UNet
from models.simcasts2s import (
    VAE, VAEEncoder, VAEDecoder, Denoiser,
    CosineNoiseScheduler, ReverseProcess
)
from .common import RequireVAEEncoders


class _AbstractPredictor(ABC):

    def __init__(
        self, net: CNN | UNet | VAE | Denoiser,
        dataset: CESM2 | ERA5,
        landmask_path: str,
        target_path: str, local_rank: int,
        batch_size: int = 1,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Prediction batch_size must be positive")
        self.batch_size: int = batch_size
        self.base_net: CNN | UNet | VAE | Denoiser = net
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
            batch_size=self.batch_size,
            drop_last=False,
            collate_fn=self.dataset.collate_fn,
            prefetch_factor=2,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
            sampler=self.sampler,
        )

        self.out_features: int
        if isinstance(self.base_net, (CNN, UNet)):
            self.out_features = self.base_net.out_features
        elif isinstance(self.base_net, VAE):
            self.out_features = self.base_net.pixel_dim
        elif isinstance(self.base_net, Denoiser):
            self.out_features = 1

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
        if isinstance(dataset, CESM2):
            self.landmask_reader = CESM2_LandmaskReader()
            self.coordinates_reader = CESM2_CoordinatesReader()
        else:
            self.landmask_reader = ERA5_LandmaskReader(resolution=(self.H, self.W))
            self.coordinates_reader = ERA5_CoordinatesReader(resolution=(self.H, self.W))

    @torch.no_grad()
    def predict(self) -> None:
        self.net.eval()
        # Each step saves per-sample results; do not retain all predictions in GPU memory.
        for batch in self.dataloader:
            self._predict_step(batch=batch)

    @abstractmethod
    def _predict_step(self, batch: DataBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predictions and targets with the batch dimension preserved."""
        pass

    @property
    @abstractmethod
    def output_names(self) -> list[str]:
        pass


class _WeeklyPredictor(_AbstractPredictor):

    def __init__(
        self, net: CNN | UNet | Denoiser,
        dataset: CESM2 | ERA5,
        landmask_path: str, target_path: str, local_rank: int,
        batch_size: int = 1,
    ) -> None:
        self.n_output_days: int = dataset.metadata.n_output_days
        if self.n_output_days <= 0 or self.n_output_days % 7 != 0:
            raise ValueError("n_output_days must be positive and divisible by 7")
        self.n_output_weeks: int = self.n_output_days // 7
        super().__init__(
            net=net, dataset=dataset, landmask_path=landmask_path,
            target_path=target_path, local_rank=local_rank, batch_size=batch_size,
        )

    @staticmethod
    def _weekly_mean(daily: torch.Tensor) -> torch.Tensor:
        # daily: (..., days, H, W, variables); preserve leading batch dimensions.
        n_days: int
        H: int
        W: int
        C: int
        n_days, H, W, C = daily.shape[-4:]
        if n_days <= 0 or n_days % 7 != 0:
            raise ValueError("The number of daily frames must be positive and divisible by 7")
        return daily.reshape(*daily.shape[:-4], n_days // 7, 7, H, W, C).mean(dim=-4)

    def _save_weekly_results(
        self,
        fields: dict[str, torch.Tensor],
        week_infos: list[list[SampleInfo]],
        *,
        prediction_key: Literal["prediction", "ensemble_mean"],
        member_idx: int | None = None,
        ensemble_size: int | None = None,
    ) -> None:
        """Save one spatial field per week and variable, for deterministic forecasts, members, or aggregates."""
        expected_shape: tuple[int, int, int, int, int] = (
            len(week_infos), self.n_output_weeks, self.H, self.W, self.out_features,
        )
        assert all(len(infos) == self.n_output_weeks for infos in week_infos)
        assert all(value.shape == expected_shape for value in fields.values())
        suffix: str = "pt"
        if ensemble_size is not None:
            suffix = f"ens_{member_idx:04d}.pt" if member_idx is not None else "ens_aggregate.pt"
        regions: tuple[str, ...] = ("global", "tropical", "extratropical")
        # loop over each sample -> each week -> each output variable
        for sample_idx, sample_week_infos in enumerate(week_infos):
            for week_idx, week_info in enumerate(sample_week_infos):
                forecast_week: int = self.dataset.metadata.n_lead_days // 7 + week_idx + 1
                for idx, output_name in enumerate(self.output_names):
                    prediction: torch.Tensor = fields[prediction_key][sample_idx, week_idx, ..., idx]
                    groundtruth: torch.Tensor = fields["groundtruth"][sample_idx, week_idx, ..., idx]
                    # Independent CPU storage prevents saving other weeks through tensor views.
                    result: dict[str, Any] = {
                        name: value[sample_idx, week_idx, ..., idx].detach().to(device="cpu", copy=True)
                        for name, value in fields.items()
                    }
                    for metric_name, metric in (("mse", self.mse), ("mae", self.mae)):
                        values: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = metric(
                            prediction=prediction, groundtruth=groundtruth,
                        )
                        for region, value in zip(regions, values):
                            result[f"{region}_{metric_name}"] = value.item()
                    for region in regions:
                        result[f"{region}_rmse"] = result[f"{region}_mse"] ** 0.5

                    prefix: str = self._make_filename_prefix(sampleinfo=week_info, output_name=output_name)
                    prefix += f"_week{forecast_week:02d}"
                    result.update({
                        "model_name": self.model_name.upper(),
                        "output_name": output_name,
                        "sim_id": week_info.sim_id,
                        "in_startdate": week_info.in_startdate,
                        "in_enddate": week_info.in_enddate,
                        "out_startdate": week_info.out_startdate,
                        "out_enddate": week_info.out_enddate,
                        "tropical_lats": self.tropical_lats,
                        "prefix": prefix,
                        "suffix": suffix,
                        "forecast_week": forecast_week,
                        "averaging_days": 7,
                    })
                    filename: str = f"{prefix}.pt"
                    if ensemble_size is not None:
                        result.update({
                            "ensemble_size": ensemble_size,
                            "ensemble_member": member_idx + 1 if member_idx is not None else -1,
                            "ensemble_stat": "member" if member_idx is not None else "aggregate",
                        })
                        filename = f"{prefix}_{suffix}"
                    self.torchio.save(obj=result, filename=filename)
                    print({k: v for k, v in result.items() if isinstance(v, (str, float, int, tuple))})

    #implement
    @property
    def output_names(self) -> list[str]:
        return self.dataset.metadata.output_vars

    def _make_filename_prefix(self, sampleinfo: SampleInfo, output_name: str) -> str:
        filename_prefix: str = (
            f"{self.model_name.upper()}_{output_name}_{sampleinfo.sim_id}_"
            f"{sampleinfo.in_startdate}{sampleinfo.in_enddate}_"
            f"{sampleinfo.out_startdate}{sampleinfo.out_enddate}"
        )
        return filename_prefix.replace("/", "")

    def _get_week_infos(self, sampleinfos: list[SampleInfo], n_days: int) -> list[list[SampleInfo]]:
        week_infos: list[list[SampleInfo]] = []
        for sampleinfo in sampleinfos:
            dates: list[str] = sampleinfo.out_dates
            assert len(dates) == n_days
            week_infos.append([
                SampleInfo(
                    sim_id=sampleinfo.sim_id,
                    in_startdate=sampleinfo.in_startdate,
                    in_enddate=sampleinfo.in_enddate,
                    out_startdate=dates[start],
                    out_enddate=dates[start + 6],
                )
                for start in range(0, n_days, 7)
            ])
        return week_infos


class BaselinePredictor(_WeeklyPredictor):

    def __init__(
        self, net: CNN | UNet,
        dataset: CESM2 | ERA5,
        landmask_path: str, target_path: str,
        local_rank: int, batch_size: int = 1,
    ) -> None:
        super().__init__(
            net=net, dataset=dataset,
            landmask_path=landmask_path, target_path=target_path,
            local_rank=local_rank, batch_size=batch_size,
        )
        self.in_features: int = net.in_features
        self.n_input_days: int = dataset.metadata.n_input_days

    def _predict_step(self, batch: DataBatch) -> tuple[torch.Tensor, torch.Tensor]:
        sampleinfos, _, _, input_tensor, groundtruth = batch
        input_tensor = input_tensor.to(self.device)
        groundtruth = groundtruth.to(self.device)
        batch_size: int = len(sampleinfos)
        assert input_tensor.shape == (batch_size, self.n_input_days, self.H, self.W, self.in_features)

        prediction: torch.Tensor = self.net(input=input_tensor)
        assert prediction.shape == groundtruth.shape == (
            batch_size, self.n_output_days, self.H, self.W, self.out_features,
        )
        prediction = self._weekly_mean(prediction)
        groundtruth = self._weekly_mean(groundtruth)
        week_infos: list[list[SampleInfo]] = self._get_week_infos(sampleinfos, self.n_output_days)
        self._save_weekly_results(
            fields={
                "groundtruth": groundtruth,
                "prediction": prediction,
                "error_map": self.error_map(prediction=prediction, groundtruth=groundtruth),
            },
            week_infos=week_infos,
            prediction_key="prediction",
        )
        return prediction, groundtruth


class VAEPredictor(_AbstractPredictor):

    #implement
    def _predict_step(self, batch: DataBatch) -> tuple[torch.Tensor, torch.Tensor]:
        sampleinfos, input_indices, output_indices, input_tensor, target_tensor = batch
        x: torch.Tensor = input_tensor.to(self.device, non_blocking=True)

        batch_size: int = x.shape[0]
        assert batch_size == len(sampleinfos)
        reconstructions: list[torch.Tensor] = []
        for date_idx in range(x.shape[1]):
            true_x: torch.Tensor = x[:, date_idx: date_idx+1, :, :, :]
            reconstructed_x, mu, logvar = self.net(true_x)
            mu_mean: torch.Tensor = mu.flatten(start_dim=1).mean(dim=1)
            sigma_mean: torch.Tensor = logvar.exp().sqrt().flatten(start_dim=1).mean(dim=1)
            # Compute metrics
            true_x = true_x.squeeze(dim=1)
            reconstructed_x = reconstructed_x.squeeze(dim=1)
            error_tensor: torch.Tensor = self.error_map(prediction=reconstructed_x, groundtruth=true_x)
            # Plotting
            reconstructions.append(reconstructed_x)
            for sample_idx, sampleinfo in enumerate(sampleinfos):
                on_date: str = sampleinfo.in_dates[date_idx]
                for idx in range(len(self.dataset.metadata.input_vars)):
                    # Select by output variable
                    true_frame: torch.Tensor = true_x[sample_idx, ..., idx]
                    reconstructed_frame: torch.Tensor = reconstructed_x[sample_idx, ..., idx]
                    error_frame: torch.Tensor = error_tensor[sample_idx, ..., idx]
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
                        "groundtruth": true_frame.detach().to(device="cpu", copy=True),
                        "reconstruction": reconstructed_frame.detach().to(device="cpu", copy=True),
                        "error_map": error_frame.detach().to(device="cpu", copy=True),
                        "global_mse": global_mse,
                        "tropical_mse": tropical_mse,
                        "extratropical_mse": extratropical_mse,
                        "global_rmse": global_rmse,
                        "tropical_rmse": tropical_rmse,
                        "extratropical_rmse": extratropical_rmse,
                        "global_mae": global_mae,
                        "tropical_mae": tropical_mae,
                        "extratropical_mae": extratropical_mae,
                        "mu": mu_mean[sample_idx].item(),
                        "sigma": sigma_mean[sample_idx].item(),
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

        reconstruction: torch.Tensor = torch.stack(reconstructions, dim=1)
        return reconstruction, x

    #implement
    @property
    def output_names(self) -> list[str]:
        return self._output_names

    @cached_property
    def _output_names(self) -> list[str]:
        days: Iterable[int] = range(1, self.dataset.metadata.n_input_days + 1)
        var_names: list[str] = self.dataset.metadata.input_vars
        output_names: list[str] = [f"{item[1]}_DAY{item[0]}" for item in product(days, var_names)]
        return output_names


class SimCastS2SPredictor(RequireVAEEncoders, _WeeklyPredictor):

    def __init__(
        self,
        denoiser: Denoiser,
        wind_encoder: VAEEncoder,
        mass_encoder: VAEEncoder,
        thermal_encoder: VAEEncoder,
        hydro_encoder: VAEEncoder,
        precip_encoder: VAEEncoder,
        precip_decoder: VAEDecoder,
        noise_scheduler: CosineNoiseScheduler,
        eta: float,
        guidance_scale: float,
        dataset: CESM2 | ERA5,
        landmask_path: str,
        target_path: str,
        local_rank: int,
        ensemble_size: int,
        inference_steps: int,
        batch_size: int = 1,
    ) -> None:

        super().__init__(
            net=denoiser, dataset=dataset,
            landmask_path=landmask_path, target_path=target_path,
            local_rank=local_rank,
            batch_size=batch_size,
        )
        self.denoiser: Denoiser = denoiser

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

        self.noise_scheduler: CosineNoiseScheduler = noise_scheduler
        self.eta: float = eta
        self.guidance_scale: float = guidance_scale
        self.ensemble_size: int = ensemble_size
        assert ensemble_size > 0
        assert self.guidance_scale >= 0.
        self.reverse_process: ReverseProcess = ReverseProcess(eta=eta, noise_scheduler=noise_scheduler)
        self.sampling_timesteps = self.reverse_process.sampling_timesteps(inference_steps)
        self.n_denoising_steps: int = len(self.sampling_timesteps) - 1

    #implement
    def _predict_step(self, batch: DataBatch) -> tuple[torch.Tensor, torch.Tensor]:
        sampleinfos, condition_days, target_days, condition, groundtruth = batch
        condition = condition.to(device=self.device)
        groundtruth = groundtruth.to(device=self.device)
        condition_days = condition_days.to(device=self.device)
        target_days = target_days.to(device=self.device)
        batch_size: int = len(sampleinfos)

        L: int = groundtruth.shape[1]
        assert groundtruth.shape[0] == condition.shape[0] == batch_size
        assert L == self.dataset.metadata.n_output_days
        week_infos: list[list[SampleInfo]] = self._get_week_infos(sampleinfos, L)

        # Encode daily fields; average only after decoding each ensemble member.
        condition_latent: torch.Tensor
        target_latent: torch.Tensor
        condition_latent, target_latent = self.vae_encode(condition=condition, target=groundtruth)
        condition_kept: torch.Tensor = torch.ones((batch_size,), device=target_latent.device, dtype=torch.bool)
        condition_dropped: torch.Tensor = torch.zeros((batch_size,), device=target_latent.device, dtype=torch.bool)
        groundtruth = self._weekly_mean(groundtruth)

        member_predictions: list[torch.Tensor] = []
        for member_idx in range(self.ensemble_size):
            # Generate gaussian
            gaussian: torch.Tensor = torch.randn_like(target_latent)
            # Denoise
            target_latent_k: torch.Tensor = gaussian
            target_latent_0: torch.Tensor | None = None
            # Keep the original training labels while jumping between selected noise levels.
            for k, k_prev in tqdm(
                zip(self.sampling_timesteps[:-1], self.sampling_timesteps[1:]),
                total=self.n_denoising_steps,
                desc=f"[Ensemble member {member_idx + 1}/{self.ensemble_size}] Sampling step: ",
            ):
                integer_step: torch.Tensor = (
                    torch.ones((batch_size, 1), device=target_latent.device, dtype=torch.long) * k
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
                predicted_velocity: torch.Tensor
                if self.guidance_scale == 1:
                    predicted_velocity = predicted_velocity_cond
                else:
                    predicted_velocity_uncond: torch.Tensor = self.net(
                        target=target_latent_k,
                        condition=condition_latent,
                        integer_step=integer_step,
                        condition_days=condition_days,
                        target_days=target_days,
                        condition_mask=condition_dropped,
                    )
                    predicted_velocity = (
                        predicted_velocity_uncond
                        + self.guidance_scale * (predicted_velocity_cond - predicted_velocity_uncond)
                    )
                target_latent_k, target_latent_0 = self.reverse_process.sample(
                    target_k=target_latent_k, predicted_velocity=predicted_velocity, k=integer_step,
                    k_prev=torch.full_like(integer_step, k_prev),
                )

            assert isinstance(target_latent_0, torch.Tensor)
            # At k=0 (last denoising step), target_latent_k = target_latent_0
            assert target_latent_k.isclose(target_latent_0).all()
            # Decode daily fields in the dataset's standardized-anomaly units.
            prediction: torch.Tensor = self.precip_decoder(target_latent_0.flatten(0, 1))
            assert prediction.shape == (batch_size * L, 1, self.H, self.W, self.out_features)
            prediction = prediction.reshape(batch_size, L, self.H, self.W, self.out_features)
            prediction = self._weekly_mean(prediction)
            member_predictions.append(prediction)
            self._save_weekly_results(
                fields={
                    "groundtruth": groundtruth,
                    "prediction": prediction,
                    "error_map": self.error_map(prediction=prediction, groundtruth=groundtruth),
                },
                week_infos=week_infos,
                prediction_key="prediction",
                member_idx=member_idx,
                ensemble_size=self.ensemble_size,
            )

        # Reduce across members, preserving batch and week dimensions.
        stack: torch.Tensor = torch.stack(member_predictions, dim=0)
        del member_predictions
        ensemble_var: torch.Tensor
        ensemble_mean: torch.Tensor
        ensemble_var, ensemble_mean = torch.var_mean(
            stack, dim=0, unbiased=self.ensemble_size > 1,
        )
        quantile_percentages: tuple[int, ...] = (1, 5, 10, 25, 50, 75, 90, 95, 99)
        probabilities: torch.Tensor = stack.new_tensor(quantile_percentages) / 100
        quantiles: torch.Tensor = stack.new_empty((len(quantile_percentages), *ensemble_mean.shape))
        # Bound sorting memory by computing all quantiles for one sample/week at a time.
        for sample_idx in range(batch_size):
            for week_idx in range(self.n_output_weeks):
                quantiles[:, sample_idx, week_idx] = stack[:, sample_idx, week_idx].quantile(
                    q=probabilities, dim=0,
                )
        del stack
        quantile_fields: dict[str, torch.Tensor] = {
            f"ensemble_q{percentage:03d}": quantiles[i]
            for i, percentage in enumerate(quantile_percentages)
        }
        self._save_weekly_results(
            fields={
                "groundtruth": groundtruth,
                "ensemble_mean": ensemble_mean,
                "ensemble_var": ensemble_var,
                "ensemble_std": ensemble_var.sqrt(),
                **quantile_fields,
                "error_mean_frame": self.error_map(prediction=ensemble_mean, groundtruth=groundtruth),
                "error_q050_frame": self.error_map(prediction=quantile_fields["ensemble_q050"], groundtruth=groundtruth),
            },
            week_infos=week_infos,
            prediction_key="ensemble_mean",
            ensemble_size=self.ensemble_size,
        )

        return ensemble_mean, groundtruth


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
            f"[Out]: Week {result_object['forecast_week']} | "
            f"{result_object['out_startdate']} - {result_object['out_enddate']} (Mean)\n"
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
            use_cartopy_projection=True,
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
            use_cartopy_projection=True,
        )

    def plot_simcasts2s_prediction(self, filename: str) -> None:
        result_object: dict[str, torch.Tensor | float | str] = self.torchio.load(filename=filename)
        ensemble_label: str
        prediction_frame: torch.Tensor
        groundtruth_frame: torch.Tensor | None
        error_frame: torch.Tensor | None
        uncertainty_frame: torch.Tensor | None
        if result_object["ensemble_stat"] == "member":
            ensemble_label = f"Ensemble Member: {result_object['ensemble_member']}/{result_object['ensemble_size']}"
            prediction_frame = cast(torch.Tensor, result_object["prediction"])
            groundtruth_frame = None
            error_frame = None
            uncertainty_frame = None
        else:
            ensemble_label = "Ensemble Mean"
            prediction_frame = cast(torch.Tensor, result_object["ensemble_mean"])
            groundtruth_frame = cast(torch.Tensor, result_object["groundtruth"])
            error_frame = cast(torch.Tensor, result_object["error_mean_frame"])
            uncertainty_frame = cast(torch.Tensor, result_object["ensemble_var"])

        # Make title
        title: str = (
            f"{result_object['model_name']}: {result_object['output_name']} - {result_object['sim_id']} ({ensemble_label})\n"
            f"[In]: {result_object['in_startdate']} - {result_object['in_enddate']}\n"
            f"[Out]: Week {result_object['forecast_week']} | "
            f"{result_object['out_startdate']} - {result_object['out_enddate']} (Mean)\n"
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
            vlim=2,
            use_cartopy_projection=True,
        )
