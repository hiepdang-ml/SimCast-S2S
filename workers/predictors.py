from typing import *
from abc import ABC, abstractmethod
from functools import cached_property
from itertools import product
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cesm2 import CESM2, CoordinatesReader, LandMaskReader
from cesm2.utils import DataBatch, SampleInfo
from common.metrics import ErrorMap, MAEMap, RsquaredMap
from common.plotting import MetricPlotter, PredictionPlotter
from models.benchmarks import CNN, UNet, ViT
from models.diffusion import (
    VAE, VAEEncoder, VAEDecoder, UNetDenoiser, 
    LinearNoiseScheduler, CosineNoiseScheduler, 
    DDPMForwardProcess, DDPMReverseProcess,
)


class _AbstractPredictor(ABC):

    def __init__(self, net: CNN | UNet | ViT | VAE):
        self.net: CNN | UNet | ViT | VAE = net
        if isinstance(net, (CNN, UNet, ViT)):
            self.out_features: int = net.out_features
        elif isinstance(net, VAE):
            self.out_features: int = net.pixel_dim
        elif isinstance(net, UNetDenoiser):
            # TODO: need to avoid magic number
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

    def predict(self, dataset: CESM2) -> None:
        self.net.eval()
        # Batch size should always be 1
        self.dataloader = DataLoader(dataset, batch_size=1, collate_fn=CESM2.collate_fn)
        records: Dict[str, List[torch.Tensor]] = {"predictions": [], "groundtruths": []}
        # Predict
        output_names: List[str] = self.get_output_names(dataset=dataset)
        with torch.no_grad():
            for batch in self.dataloader:
                prediction_tensor, groundtruth_tensor = self._predict_step(batch=batch, output_names=output_names)
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
        for idx, output_name in enumerate(output_names):
            self.metric_plotter.plot(
                mae_frame=mae_frame[..., idx],
                rsquared_frame=rsquared_frame[..., idx],
                landmask=self.landmask_reader.tensor,
                coordinates=self.coordinates_reader.tensors,
                title=f"{output_name}\n{dataset.metadata.start_year} - {dataset.metadata.end_year}",
                filename=f"{self.model_name.upper()}_{output_name}_metrics.png",
            )

    @abstractmethod
    def _predict_step(self, batch: DataBatch, output_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def get_output_names(self, dataset: CESM2) -> List[str]:
        pass


class BaselinePredictor(_AbstractPredictor):

    #implement
    def _predict_step(self, batch: DataBatch, output_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
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
        for idx, output_name in enumerate(output_names):
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
    def get_output_names(self, dataset: CESM2) -> List[str]:
        return dataset.metadata.output_vars


class VAEPredictor(_AbstractPredictor):

    def __init__(self, net: VAE, tp: Literal["context", "target"]) -> None:
        super().__init__(net=net)
        self.tp: Literal["context", "target"] = tp

    #implement
    def _predict_step(self, batch: DataBatch, output_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        sampleinfos, input_indices, output_indices, input_tensor, target_tensor = batch
        sampleinfo: SampleInfo = sampleinfos[0] # because batch_size=1
        true_x: torch.Tensor = input_tensor if self.tp.lower() == "context" else target_tensor
        reconstructed_x, mu, logvar = self.net(true_x)
        # Compute metrics
        true_x = true_x.permute(0, 2, 3, 1, 4).flatten(3, 4).unsqueeze(dim=1)
        reconstructed_x = reconstructed_x.permute(0, 2, 3, 1, 4).flatten(3, 4).unsqueeze(dim=1)
        error_tensor: torch.Tensor = self.error_map(prediction=reconstructed_x, groundtruth=true_x)

        # Plotting
        true_x = true_x.squeeze(dim=(0, 1))
        reconstructed_x = reconstructed_x.squeeze(dim=(0, 1))
        error_tensor = error_tensor.squeeze(dim=(0, 1))
        for idx, output_name in enumerate(output_names):
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
            # TODO: uncomment
            self.prediction_plotter.plot(
                groundtruth_frame=true_frame,
                prediction_frame=reconstructed_frame,
                error_frame=error_frame,
                landmask=self.landmask_reader.tensor,
                coordinates=self.coordinates_reader.tensors,
                title=title,
                filename=filename,
            )

        return reconstructed_x, true_x

    #implement
    def get_output_names(self, dataset: CESM2) -> List[str]:
        if self.tp.lower() == "context":
            days: Iterable[int] = range(1, dataset.metadata.n_input_days + 1)
            var_names: List[str] = dataset.metadata.input_vars
            output_names: List[str] = [f"{item[1]}_DAY{item[0]}" for item in product(days, var_names)]
        else:
            var_names: List[str] = dataset.metadata.output_vars
            output_names: List[str] = [f"{item}_AVG" for item in var_names]

        return output_names


class DDPMPredictor(_AbstractPredictor):

    def __init__(
        self, 
        denoiser: UNetDenoiser, 
        target_encoder: VAEEncoder, 
        target_decoder: VAEDecoder,
        context_encoder: VAEEncoder,
        noise_scheduler: LinearNoiseScheduler | CosineNoiseScheduler,
    ) -> None:
        super().__init__(net=denoiser)
        self.denoiser: UNetDenoiser = denoiser

        # Freeze target encoder
        self.target_encoder: VAEEncoder = target_encoder
        self.target_encoder.freeze()
        assert target_encoder.is_frozen
        # Freeze target decoder
        self.target_decoder: VAEDecoder = target_decoder
        self.target_decoder.freeze()
        assert target_decoder.is_frozen
        # Freeze context encoder
        self.context_encoder: VAEEncoder = context_encoder
        self.context_encoder.freeze()
        assert self.context_encoder.is_frozen

        self.noise_scheduler: LinearNoiseScheduler | CosineNoiseScheduler = noise_scheduler
        self.n_denoising_steps: int = noise_scheduler.n_steps
        self.reverse_process: DDPMReverseProcess = DDPMReverseProcess(noise_scheduler)

    def _predict_step(self, batch: DataBatch, output_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        sampleinfos, _, _, condition, groundtruth = batch
        sampleinfo: SampleInfo = sampleinfos[0] # because batch_size=1
        # FIXME: UNetDenoiser does not see n_input_days and in_features
        # assert condition.shape == (1, self.net.n_input_days, 192, 288, self.net.in_features)

        # Encode condition to latent space
        condition_latent: torch.Tensor = VAEEncoder.reparameterize(*self.context_encoder(condition))
        target_latent: torch.Tensor = VAEEncoder.reparameterize(*self.target_encoder(groundtruth)) # just to get shape
        # Generate gaussian
        gaussian: torch.Tensor = torch.randn_like(target_latent)
        # Denoise
        target_latent_k: torch.Tensor = gaussian
        with torch.no_grad():
            for k in reversed(range(1, self.noise_scheduler.n_steps)):
                step: torch.Tensor = torch.ones(1, 1, device=target_latent.device, dtype=torch.int32) * k
                # DDPM backward process
                predicted_gaussian: torch.Tensor = self.denoiser(
                    target=target_latent_k, condition=condition_latent, step=step,
                )
                target_latent_k, target_latent_0 = self.reverse_process.sample(
                    target_k=target_latent_k, predicted_noise=predicted_gaussian, step=step,
                )
                print(f"k={k}")
                print(f"target_latent_k.isnan: {target_latent_k.isnan().any()}")
                print(f"target_latent_0.isnan: {target_latent_0.isnan().any()}")
                print("-----")

        # At k=0 (last denoising step), target_latent_k = target_latent_0
        print(target_latent_k.mean())
        print(target_latent_0.mean())
        assert target_latent_k.isclose(target_latent_0).all()
        # Decode target back to physical space
        prediction: torch.Tensor = self.target_decoder(target_latent_0)
        assert prediction.shape == groundtruth.shape == (1, 1, 192, 288, self.out_features)
        # Error map
        error: torch.Tensor = self.error_map(prediction=prediction, groundtruth=groundtruth)
        error = error.squeeze(dim=(0, 1))

        # Plotting
        groundtruth = groundtruth.squeeze(dim=(0, 1))
        prediction = prediction.squeeze(dim=(0, 1))
        for idx, output_name in enumerate(output_names):
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
    def get_output_names(self, dataset: CESM2) -> List[str]:
        return dataset.metadata.output_vars
