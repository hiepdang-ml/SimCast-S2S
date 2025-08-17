from typing import *
from abc import ABC, abstractmethod
import torch


class _BaseMap(ABC):

    def __init__(self, n_features: int) -> None:
        self.n_features: int = n_features

class _SampleLevelMap(_BaseMap):

    @abstractmethod
    def __call__(self, prediction: torch.Tensor, groundtruth: torch.Tensor) -> torch.Tensor:
        pass

class _SequenceLevelMap(_BaseMap):

    @abstractmethod
    def __call__(self, predictions: List[torch.Tensor], groundtruths: List[torch.Tensor]) -> torch.Tensor:
        pass


class ErrorMap(_SampleLevelMap):

    #implement
    def __call__(self, prediction: torch.Tensor, groundtruth: torch.Tensor) -> torch.Tensor:
        batch_size: int = prediction.shape[0]
        assert prediction.shape == groundtruth.shape == (batch_size, 1, 192, 288, self.n_features)
        return groundtruth - prediction


class RsquaredMap(_SequenceLevelMap):

    #implement
    def __call__(self, predictions: List[torch.Tensor], groundtruths: List[torch.Tensor]) -> torch.Tensor:
        
        n_samples: int = len(predictions)
        assert len(predictions) == len(groundtruths) == n_samples
        assert all(tensor.shape == (192, 288, self.n_features) for tensor in predictions)
        assert all(tensor.shape == (192, 288, self.n_features) for tensor in groundtruths)

        prediction_tensor: torch.Tensor = torch.stack(predictions, dim=0)
        groundtruth_tensor: torch.Tensor = torch.stack(groundtruths, dim=0)
        true_mean_tensor: torch.Tensor = torch.mean(groundtruth_tensor, dim=0, keepdim=True)
        total_variation_tensor: torch.Tensor = torch.sum(input=(groundtruth_tensor - true_mean_tensor) ** 2, dim=0, keepdim=False)
        residual_tensor: torch.Tensor = torch.sum(input=(prediction_tensor - groundtruth_tensor) ** 2, dim=0, keepdim=False)
        assert total_variation_tensor.shape == residual_tensor.shape == (192, 288, self.n_features)
        rsquared_map: torch.Tensor = 1 - residual_tensor / (total_variation_tensor + 1e-6)
        print(f"Max R-squared: {rsquared_map.max().item()}")
        print(f"Min R-squared: {rsquared_map.min().item()}")
        print(f"Mean R-squared: {rsquared_map.mean().item()}")
        print(f"Std R-squared: {rsquared_map.std().item()}")
        assert rsquared_map.shape == (192, 288, self.n_features)
        return rsquared_map


class MAEMap(_SequenceLevelMap):

    #implement
    def __call__(self, predictions: List[torch.Tensor], groundtruths: List[torch.Tensor]) -> torch.Tensor:

        n_samples: int = len(predictions)
        assert len(predictions) == len(groundtruths) == n_samples
        batch_size: int = groundtruths[0].shape[0]
        assert all(tensor.shape == (192, 288, self.n_features) for tensor in predictions)
        assert all(tensor.shape == (192, 288, self.n_features) for tensor in groundtruths)
        prediction_tensor: torch.Tensor = torch.stack(predictions, dim=0)
        groundtruth_tensor: torch.Tensor = torch.stack(groundtruths, dim=0)
        mae_map: torch.Tensor = (groundtruth_tensor - prediction_tensor).abs().mean(dim=0, keepdim=False)
        print(f"Max MAE: {mae_map.max().item()}")
        print(f"Min MAE: {mae_map.min().item()}")
        print(f"Mean MAE: {mae_map.mean().item()}")
        print(f"Std MAE: {mae_map.std().item()}")
        torch.save(obj=mae_map, f="./mae.pt")
        assert mae_map.shape == (192, 288, self.n_features)
        return mae_map



