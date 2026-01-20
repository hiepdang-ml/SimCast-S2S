from functools import cached_property
from abc import ABC, abstractmethod
import xarray as xr
import torch


class _GeographicalMetricOrMap(ABC):

    def __init__(self, tropical_lats: tuple[float, float]):
        self.tropical_lats: tuple[float, float] = tropical_lats

    @cached_property
    def tropical_mask(self) -> torch.Tensor:
        # TODO: make configurable path
        ds: xr.Dataset = xr.open_dataset(
            f"/scratch/zgp2ps/cesm2/landmask/b.e21.BHISTcmip6.f09_g17.LE2-1001.001.clm2.h3.C14_SOILC_vr.18500101-18590101.nc"
        )
        mask: xr.DataArray = (self.tropical_lats[0] <= ds["lat"]) & (ds["lat"] <= self.tropical_lats[1])
        mask: torch.Tensor = torch.from_numpy(mask.values)
        assert mask.shape == (192,)
        return mask

    @abstractmethod
    def __call__(self, prediction: torch.Tensor, groundtruth: torch.Tensor) -> torch.Tensor:
        pass


class GeographicalMAE(_GeographicalMetricOrMap):
    
    # implement
    def __call__(self, prediction: torch.Tensor, groundtruth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert prediction.shape == groundtruth.shape
        assert prediction.shape[0] == self.tropical_mask.shape[0]
        mae_map: torch.Tensor = torch.nn.functional.l1_loss(input=prediction, target=groundtruth, reduction="none")
        global_mae: torch.Tensor = torch.mean(mae_map)
        tropical_mae: torch.Tensor = torch.mean(mae_map[self.tropical_mask])
        extratropical_mae: torch.Tensor = torch.mean(mae_map[~self.tropical_mask])
        return global_mae, tropical_mae, extratropical_mae


class GeographicalMSE(_GeographicalMetricOrMap):
    
    # implement
    def __call__(self, prediction: torch.Tensor, groundtruth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert prediction.shape == groundtruth.shape
        assert prediction.shape[0] == self.tropical_mask.shape[0]
        mse_map: torch.Tensor = torch.nn.functional.mse_loss(input=prediction, target=groundtruth, reduction="none")
        global_mse: torch.Tensor = torch.mean(mse_map)
        tropical_mse: torch.Tensor = torch.mean(mse_map[self.tropical_mask])
        extratropical_mse: torch.Tensor = torch.mean(mse_map[~self.tropical_mask])
        return global_mse, tropical_mse, extratropical_mse
    

class _BaseMap(ABC):

    def __init__(self, n_features: int) -> None:
        self.n_features: int = n_features

class _SampleLevelMap(_BaseMap):

    @abstractmethod
    def __call__(self, prediction: torch.Tensor, groundtruth: torch.Tensor) -> torch.Tensor:
        pass

class _SequenceLevelMap(_BaseMap):

    @abstractmethod
    def __call__(self, predictions: list[torch.Tensor], groundtruths: list[torch.Tensor]) -> torch.Tensor:
        pass


class ErrorMap(_SampleLevelMap):

    #implement
    def __call__(self, prediction: torch.Tensor, groundtruth: torch.Tensor) -> torch.Tensor:
        assert prediction.shape == groundtruth.shape == (192, 288, self.n_features)
        return groundtruth - prediction


class GeographicalRsquaredMap(_SequenceLevelMap, _GeographicalMetricOrMap):

    def __init__(self, n_features: int, tropical_lats: tuple[float, float]):
        _GeographicalMetricOrMap.__init__(self, tropical_lats)
        _SequenceLevelMap.__init__(self, n_features)

    #implement
    def __call__(self, predictions: list[torch.Tensor], groundtruths: list[torch.Tensor]) -> torch.Tensor:
        
        n_samples: int = len(predictions)
        assert len(predictions) == len(groundtruths) == n_samples
        assert all(tensor.shape == (192, 288, self.n_features) for tensor in predictions)
        assert all(tensor.shape == (192, 288, self.n_features) for tensor in groundtruths)

        prediction_tensor: torch.Tensor = torch.stack(predictions, dim=0).cpu()     # to avoid out of VRAM
        groundtruth_tensor: torch.Tensor = torch.stack(groundtruths, dim=0).cpu()   # to avoid out of VRAM
        true_mean_tensor: torch.Tensor = torch.mean(groundtruth_tensor, dim=0, keepdim=True)
        total_variation_tensor: torch.Tensor = torch.sum(input=(groundtruth_tensor - true_mean_tensor) ** 2, dim=0, keepdim=False)
        residual_tensor: torch.Tensor = torch.sum(input=(prediction_tensor - groundtruth_tensor) ** 2, dim=0, keepdim=False)
        assert total_variation_tensor.shape == residual_tensor.shape == (192, 288, self.n_features)
        rsquared_map: torch.Tensor = 1 - residual_tensor / (total_variation_tensor + 1e-6)
        tropical_rsquared_map = rsquared_map[self.tropical_mask]
        extratropical_rsquared_map = rsquared_map[~self.tropical_mask]
        global_rsquared: torch.Tensor = rsquared_map[rsquared_map > 0].mean()
        tropical_rsquared: torch.Tensor = tropical_rsquared_map[tropical_rsquared_map > 0].mean()
        extratropical_rsquared: torch.Tensor = extratropical_rsquared_map[extratropical_rsquared_map > 0].mean()
        print(f"R-squared (Global): {global_rsquared}")
        print(f"R-squared (Tropic): {tropical_rsquared}")
        print(f"R-squared (Extratropic): {extratropical_rsquared}")
        assert rsquared_map.shape == (192, 288, self.n_features)
        return rsquared_map, global_rsquared, tropical_rsquared, extratropical_rsquared


class MAEMap(_SequenceLevelMap, _GeographicalMetricOrMap):

    def __init__(self, n_features: int, tropical_lats: tuple[float, float]):
        _GeographicalMetricOrMap.__init__(self, tropical_lats)
        _SequenceLevelMap.__init__(self, n_features)
        
    #implement
    def __call__(self, predictions: list[torch.Tensor], groundtruths: list[torch.Tensor]) -> torch.Tensor:

        n_samples: int = len(predictions)
        assert len(predictions) == len(groundtruths) == n_samples
        batch_size: int = groundtruths[0].shape[0]
        assert all(tensor.shape == (192, 288, self.n_features) for tensor in predictions)
        assert all(tensor.shape == (192, 288, self.n_features) for tensor in groundtruths)
        prediction_tensor: torch.Tensor = torch.stack(predictions, dim=0).cpu()     # to avoid out of VRAM
        groundtruth_tensor: torch.Tensor = torch.stack(groundtruths, dim=0).cpu()   # to avoid out of VRAM
        mae_map: torch.Tensor = (groundtruth_tensor - prediction_tensor).abs().mean(dim=0, keepdim=False)
        tropical_mae_map = mae_map[self.tropical_mask]
        extratropical_mae_map = mae_map[~self.tropical_mask]
        global_mae: torch.Tensor = mae_map[mae_map > 0].mean()
        tropical_mae: torch.Tensor = tropical_mae_map[tropical_mae_map > 0].mean()
        extratropical_mae: torch.Tensor = extratropical_mae_map[extratropical_mae_map > 0].mean()
        print(f"MAE (Global): {global_mae}")
        print(f"MAE (Tropic): {tropical_mae}")
        print(f"MAE (Extratropic): {extratropical_mae}")
        assert mae_map.shape == (192, 288, self.n_features)
        return mae_map, global_mae, tropical_mae, extratropical_mae


