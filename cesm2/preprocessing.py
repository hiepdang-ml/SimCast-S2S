from typing import *

import torch
import torch.nn.functional as F
from cesm2.common import DataContainer, MetaData


class Detrender:

    def __init__(self, metadata: MetaData) -> None:
        self.metadata: MetaData = metadata

    def linear_regression(self, data: DataContainer) -> DataContainer:
        n_years: int = len(self.metadata.years)
        year_tensor: torch.Tensor = torch.tensor(data=self.metadata.years, dtype=torch.float)
        assert year_tensor.shape == (n_years,)
        X: torch.Tensor = torch.stack([torch.ones(n_years), year_tensor], dim=1)
        assert X.shape == (n_years, 2)
        mean_container: DataContainer = data.yearly_agg(reduce_func="mean")
        results: DataContainer = DataContainer(self.metadata)

        for var_name in self.metadata.var_names:
            for sim_id in self.metadata.sim_ids:
                mean_tensors: Dict[int: torch.Tensor] = mean_container.get(var_name=var_name, sim_id=sim_id, year=None)
                mean_tensor: torch.Tensor = torch.cat([tensor for tensor in mean_tensors.values()], dim=0) # along T
                assert mean_tensor.shape == (n_years, 192, 288)
                y: torch.Tensor = mean_tensor.reshape(n_years, 192 * 288)
                W: torch.Tensor = torch.linalg.lstsq(X, y).solution
                assert W.shape == (2, 192 * 288)
                y_bar: torch.Tensor = X.matmul(W)
                assert y_bar.shape == (n_years, 192 * 288)
                y_bar = y_bar.reshape(n_years, 192, 288)
                for i, year in enumerate(self.metadata.years):
                    results.set(var_name=var_name, sim_id=sim_id, year=year, value=y_bar[i])
        
        return results

    def __call__(self, input: DataContainer) -> DataContainer:
        trend_container: DataContainer = self.linear_regression(input)
        output: DataContainer = DataContainer(self.metadata)
        for var_name, sim_id, year in self.metadata.combinations:
            input_tensor: torch.Tensor = input.get(var_name=var_name, sim_id=sim_id, year=year)
            assert input_tensor.shape == (365, 192, 288)
            trend_tensor: torch.Tensor = trend_container.get(var_name=var_name, sim_id=sim_id, year=year)
            assert trend_tensor.shape == (192, 288)
            detrended: torch.Tensor = input_tensor - trend_tensor     # broadcast along T (dim=1)
            assert detrended.shape == (365, 192, 288)
            output.set(var_name=var_name, sim_id=sim_id, year=year, value=detrended)

        return output

class ClimatologyRemover:

    def __init__(self, metadata: MetaData, window_size: int):
        self.metadata: MetaData = metadata
        assert window_size % 2 == 1, "window size must be an odd number"
        self.window_size: int = window_size
        self.half_window: int = window_size // 2

    def __call__(self, input: DataContainer) -> DataContainer:
        output: DataContainer = DataContainer(self.metadata)
        for var_name in self.metadata.var_names:
            for sim_id in self.metadata.sim_ids:
                # retrieve input data (ensure ascending year)
                year_tensors: Dict[str, torch.Tensor] = input.get(var_name=var_name, sim_id=sim_id, year=None)
                year_tensors: List[torch.Tensor] = [year_tensors[k] for k in sorted(year_tensors)]
                assert all(tensor.shape == (365, 192, 288) for tensor in year_tensors)
                input_tensor: torch.Tensor = torch.stack(year_tensors, dim=0)
                input_tensor = input_tensor.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
                assert input_tensor.shape == (len(self.metadata.years), 192 * 288, 365)
                # compute climatology
                climatology = input_tensor.mean(dim=0, keepdim=True)
                assert climatology.shape == (1, 192 * 288, 365)
                climatology = F.pad(input=climatology, pad=(self.half_window, self.half_window), mode="replicate")
                assert climatology.shape == (1, 192 * 288, self.half_window + 365 + self.half_window)
                climatology = F.avg_pool1d(input=climatology, kernel_size=self.window_size, stride=1)
                assert climatology.shape == (1, 192 * 288, 365)
                # compute anomaly
                anomaly_tensor: torch.Tensor = input_tensor - climatology
                assert anomaly_tensor.shape == (len(self.metadata.years), 192 * 288, 365)
                # standardize
                std_tensor: torch.Tensor = anomaly_tensor.std(dim=0, keepdim=True)
                output_tensor: torch.Tensor = anomaly_tensor / std_tensor
                assert output_tensor.shape == (len(self.metadata.years), 192 * 288, 365)
                # asign output
                output_tensor = output_tensor.transpose(1, 2).reshape(len(self.metadata.years), 365, 192, 288)
                for i, year in enumerate(self.metadata.years):
                    output.set(var_name=var_name, sim_id=sim_id, year=year, value=output_tensor[i])
                
        return output
    