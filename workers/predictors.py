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
from models import CNN, VAE, UNet, ViT
from .utils import HasModelName


class AbstractPredictor(HasModelName, ABC):

    def __init__(self, net: CNN | UNet | ViT | VAE):
        self.net: CNN | UNet | ViT | VAE = net
        self.mse = nn.MSELoss(reduction='mean')
        self.mae = nn.L1Loss(reduction='mean')
        self.rsquared_map: RsquaredMap = RsquaredMap(n_output_vars=net.out_features)
        self.mae_map: MAEMap = MAEMap(n_output_vars=net.out_features)
        self.error_map: ErrorMap = ErrorMap(n_output_vars=net.out_features)
        self.prediction_plotter: PredictionPlotter = PredictionPlotter()
        self.metric_plotter: MetricPlotter = MetricPlotter()
        self.landmask_reader: LandMaskReader = LandMaskReader(device="cpu")
        self.coordinates_reader: CoordinatesReader = CoordinatesReader(device="cpu")

    def predict(self, dataset: CESM2) -> None:
        # Batch size should always be 1
        self.dataloader = DataLoader(dataset, batch_size=1, collate_fn=CESM2.collate_fn)
        records: Dict[str, List[torch.Tensor]] = {"predictions": [], "groundtruths": []}
        # Predict
        output_names: List[str] = self.get_output_names(dataset=dataset)
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                prediction_tensor, groundtruth_tensor = self._predict_step(batch=batch, output_names=output_names)
                # Record for aggregate metrics
                records["groundtruths"].append(groundtruth_tensor)
                records["predictions"].append(prediction_tensor)

        # Compute aggregate metrics
        rsquared_frame: torch.Tensor = self.rsquared_map(
            predictions=records["predictions"], groundtruths=records["groundtruths"],
        )
        if isinstance(self.net, (CNN, UNet, ViT)):
            assert rsquared_frame.shape == (192, 288, self.net.out_features)
        elif isinstance(self.net, VAE):
            assert rsquared_frame.shape == (192, 288, self.net.pixel_dim)

        mae_frame: torch.Tensor = self.mae_map(
            predictions=records["predictions"], groundtruths=records["groundtruths"],
        )
        if isinstance(self.net, (CNN, UNet, ViT)):
            assert mae_frame.shape == (192, 288, self.net.out_features)
        elif isinstance(self.net, VAE):
            assert mae_frame.shape == (192, 288, self.net.pixel_dim)

        # Plot aggregate metrics
        for idx, output_name in enumerate(output_names):
            self.metric_plotter.plot(
                mae_frame=mae_frame[..., idx],
                rsquared_frame=rsquared_frame[..., idx],
                landmask=self.landmask_reader.tensor,
                coordinates=self.coordinates_reader.tensors,
                title=f"{output_name}\n{dataset.metadata.start_year} - {dataset.metadata.end_year}",
                filename=f"{self.model_name}_{output_name}_metrics.png",
            )

    @abstractmethod
    def _predict_step(self, batch: DataBatch, output_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def get_output_names(self, dataset: CESM2) -> List[str]:
        pass


class VAEPredictor(AbstractPredictor):

    def __init__(self, net: VAE, tp: Literal["context", "target"]) -> None:
        super().__init__(net=net)
        self.tp: Literal["context", "target"] = tp

    #implement
    def _predict_step(self, batch: DataBatch, output_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        sampleinfos, input_indices, output_indices, input_tensor, self_tensor, target_tensor = batch
        sampleinfo: SampleInfo = sampleinfos[0] # because batch_size=1
        true_x: torch.Tensor = input_tensor if self.tp.lower() == "context" else target_tensor
        batch_size, n_days, H, W, n_features = true_x.shape
        assert n_days * n_features == self.net.pixel_dim
        true_x = true_x.permute(0, 2, 3, 1, 4).flatten(start_dim=3, end_dim=4) # must not .permute(0, 2, 3, 4, 1)
        reconstructed_x, mu, logvar = self.net(true_x)
        assert reconstructed_x.shape == true_x.shape == (1, 192, 288, self.net.pixel_dim)

        # Error map
        error_tensor: torch.Tensor = self.error_map(prediction=reconstructed_x, groundtruth=true_x)
        error_tensor = error_tensor.squeeze(dim=0)

        # Plotting
        reconstructed_x = reconstructed_x.squeeze(dim=0)
        true_x = true_x.squeeze(dim=0)
        for idx, output_name in enumerate(output_names):
            # Select by output variable
            true_x: torch.Tensor = true_x[..., idx]
            reconstructed_x: torch.Tensor = reconstructed_x[..., idx]
            error_frame: torch.Tensor = error_tensor[..., idx]
            # MSE value
            mean_mse: float = self.mse(input=reconstructed_x, target=true_x).item()
            # RMSE value
            mean_rmse: float = mean_mse ** 0.5
            # MAE value
            mean_mae: float = self.mae(input=reconstructed_x, target=true_x).item()

            # Make title
            title: str = (
                f"{self.model_name}\n"
                f"{output_name}\n"
                f"{sampleinfo.sim_id}\n"
                f"MSE: {mean_mse:.4f}, RMSE: {mean_rmse:.4f}, MAE: {mean_mae:.4f}\n"
            )
            print(title)
            print("------")

            # Make file name
            filename: str = (
                f"{self.model_name}_{output_name}_{sampleinfo.sim_id}_"
                f"{sampleinfo.in_startdate}{sampleinfo.in_enddate}_"
                f"{sampleinfo.out_startdate}{sampleinfo.out_enddate}"
                f".png"
            )
            filename = filename.replace("/", "")
            # Plot single frame
            self.prediction_plotter.plot(
                groundtruth_frame=true_x,
                prediction_frame=reconstructed_x,
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
            var_names: List[str] = dataset.metadata.input_vars
        else:
            var_names: List[str] = dataset.metadata.output_vars

        combinations: Iterable[Tuple[str, str]] = product(range(1, dataset.metadata.n_input_days + 1), var_names)
        return [f"{item[1]}_DAY{item[0]}" for item in combinations]


class BaselinePredictor(AbstractPredictor):

    #implement
    def _predict_step(self, batch: DataBatch, output_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        sampleinfos, input_indices, output_indices, input_tensor, self_tensor, groundtruth_tensor = batch
        sampleinfo: SampleInfo = sampleinfos[0] # because batch_size=1
        assert input_tensor.shape == (1, self.net.n_input_days, 192, 288, self.net.in_features)
        assert prediction_tensor.shape == groundtruth_tensor.shape == (1, 1, 192, 288, self.net.out_features)

        # Forward pass
        if isinstance(self.net, (CNN, UNet)):
            prediction_tensor: torch.Tensor = self.net(input=input_tensor)
        elif isinstance(self.net, ViT):
            prediction_tensor: torch.Tensor = self.net(input=input_tensor, input_indices=input_indices)
        else:
            raise NotImplementedError(f"{self.model_name} is not implemented")

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
                f"{self.model_name}\n"
                f"{output_name}\n"
                f"{sampleinfo.sim_id}\n"
                f"[In]: {sampleinfo.in_startdate} - {sampleinfo.in_enddate}\n"
                f"[Out]: {sampleinfo.out_startdate} - {sampleinfo.out_enddate}\n"
                f"MSE: {mean_mse:.4f}, RMSE: {mean_rmse:.4f}, MAE: {mean_mae:.4f}\n"
            )
            print(title)
            print("------")
            # Make file name
            filename: str = (
                f"{self.model_name}_{output_name}_{sampleinfo.sim_id}_"
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

