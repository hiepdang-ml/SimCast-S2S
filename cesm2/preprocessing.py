import os
from collections import defaultdict
from typing import *

import torch
import torch.nn.functional as F

from cesm2.container import DataContainer
from common.configs import MetaData


class _LinearRegressor:

    def __init__(self) -> None:
        self.W: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)  # sim_ids, var_names

    def fit(self, mean_container: DataContainer, train_metadata: MetaData) -> None:
        assert (not bool(self.W)) or (mean_container is None), "Linear regression is already fit"
        mean_container = mean_container.to(train_metadata.device)
        X: torch.Tensor = self.__get_X(metadata=train_metadata)
        assert X.shape == (train_metadata.n_years, 2)
        for sim_id in train_metadata.sim_ids:
            for var_name in train_metadata.var_names:
                mean_tensors: Dict[int, torch.Tensor] = mean_container.get(
                    sim_id=sim_id, var_name=var_name, year=None
                )
                y: torch.Tensor = torch.cat([mean_tensors[year] for year in train_metadata.years], dim=0)
                y = y.reshape(train_metadata.n_years, 192 * 288)
                W: torch.Tensor = torch.linalg.lstsq(X, y).solution
                assert W.shape == (2, 192 * 288)
                self.W[sim_id][var_name] = W

    def __call__(self, new_metadata: MetaData) -> DataContainer:
        X: torch.Tensor = self.__get_X(metadata=new_metadata)
        result: DataContainer = DataContainer(new_metadata)
        for sim_id in new_metadata.sim_ids:
            for var_name in new_metadata.var_names:
                y_bar: torch.Tensor = X.matmul(self.W[sim_id][var_name])
                y_bar = y_bar.reshape(new_metadata.n_years, 192, 288)
                for i, year in enumerate(new_metadata.years):
                    result.set(sim_id=sim_id, var_name=var_name, year=year, value=y_bar[i])
        return result

    @staticmethod
    def __get_X(metadata: MetaData) -> torch.Tensor:
        return torch.stack([
                torch.ones(metadata.n_years, dtype=torch.float, device=metadata.device),
                torch.tensor(metadata.years, dtype=torch.float, device=metadata.device)
            ], 
            dim=1
        )

    def load_state(self, train_metadata: MetaData) -> None:
        with torch.serialization.safe_globals([defaultdict]):
            # collect the state of all sim ids
            for sim_id in train_metadata.sim_ids:
                filename: str = f"{sim_id}_{train_metadata.start_year}_{train_metadata.end_year}.pt"
                self.W[sim_id] = torch.load(
                    f=train_metadata.detrender_state_directory.joinpath(filename), weights_only=False
                )

    def save_state(self, train_metadata: MetaData) -> None:
        for sim_id in train_metadata.sim_ids:
            filename: str = f"{sim_id}_{train_metadata.start_year}_{train_metadata.end_year}.pt"
            torch.save(obj=self.W[sim_id], f=train_metadata.detrender_state_directory.joinpath(filename))


class Detrender:

    def __init__(self, metadata: MetaData) -> None:
        self.metadata: MetaData = metadata
        os.makedirs(name=self.metadata.detrender_state_directory, exist_ok=True)

    def __call__(self, input_container: DataContainer, train_metadata: MetaData | None = None) -> None:
        """
        Inplace operation to save memory
        """
        input_container = input_container.to(device=self.metadata.device)
        mean_container: DataContainer = input_container.yearly_agg(reduce_func="mean")
        lr: _LinearRegressor = _LinearRegressor()
        if self.metadata.tp == "train":
            # fit new lr (should be during training)
            assert train_metadata is None
            lr.fit(mean_container=mean_container, train_metadata=self.metadata)
            lr.save_state(train_metadata=self.metadata)
        else:
            # load saved lr (should be during inference)
            assert isinstance(train_metadata, MetaData)
            lr.load_state(train_metadata=train_metadata)

        trend_container: DataContainer = lr(new_metadata=self.metadata)

        for sim_id, var_name, year in self.metadata.combinations:
            input_tensor: torch.Tensor = input_container.get(sim_id=sim_id, var_name=var_name, year=year)
            assert input_tensor.shape == (365, 192, 288)
            trend_tensor: torch.Tensor = trend_container.get(sim_id=sim_id, var_name=var_name, year=year)
            assert trend_tensor.shape == (192, 288)
            detrended: torch.Tensor = input_tensor - trend_tensor   # broadcast along T (dim=0)
            assert detrended.shape == (365, 192, 288)
            input_container.set(sim_id=sim_id, var_name=var_name, year=year, value=detrended)


class _ClimatologicalMean:

    def __init__(self):
        self.climatological_mean: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)  # sim_ids, var_names
        self.climatological_std : Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)  # sim_ids, var_names

    def fit(self, input_container: DataContainer, train_metadata: MetaData) -> None:
        assert not bool(self.climatological_mean), "climatological_mean is already computed"
        assert train_metadata.climatological_window_size % 2 == 1, "window size must be an odd number"
        window_size: int = train_metadata.climatological_window_size
        half_window: int = window_size // 2
        n_years: int = train_metadata.n_years
        input_container = input_container.to(device=train_metadata.device)
        
        for sim_id in train_metadata.sim_ids:
            for var_name in train_metadata.var_names:
                # retrieve input data (ensure ascending year)
                year_tensors: Dict[str, torch.Tensor] = input_container.get(sim_id=sim_id, var_name=var_name, year=None)
                year_tensors: List[torch.Tensor] = [year_tensors[k] for k in sorted(year_tensors)]
                assert all(tensor.shape == (365, 192, 288) for tensor in year_tensors)
                input_tensor: torch.Tensor = torch.stack(year_tensors, dim=0)
                input_tensor = input_tensor.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
                assert input_tensor.shape == (n_years, 192 * 288, 365)
                # compute climatological mean
                padded_input: torch.Tensor = F.pad(
                    input=input_tensor.mean(dim=0, keepdim=True), pad=(half_window, half_window), mode="replicate"
                )
                assert padded_input.shape == (1, 192 * 288, half_window + 365 + half_window)
                climatological_mean = F.avg_pool1d(input=padded_input, kernel_size=window_size, stride=1)
                assert climatological_mean.shape == (1, 192 * 288, 365)
                del padded_input
                climatological_mean = climatological_mean.reshape(192, 288, 365)
                climatological_mean = climatological_mean.permute(2, 0, 1)
                assert climatological_mean.shape == (365, 192, 288)
                self.climatological_mean[sim_id][var_name] = climatological_mean
                # compute climatological std
                padded_sq_input: torch.Tensor = F.pad(
                    input=(input_tensor ** 2).mean(dim=0, keepdim=True), pad=(half_window, half_window), mode="replicate"
                )
                assert padded_sq_input.shape == (1, 192 * 288, half_window + 365 + half_window)
                climatological_sq_mean: torch.Tensor = F.avg_pool1d(input=padded_sq_input, kernel_size=window_size, stride=1)
                assert climatological_sq_mean.shape == (1, 192 * 288, 365)
                del input_tensor, padded_sq_input
                climatological_sq_mean = climatological_sq_mean.reshape(192, 288, 365)
                climatological_sq_mean = climatological_sq_mean.permute(2, 0, 1)
                assert climatological_sq_mean.shape == (365, 192, 288)
                # Var(X) = E(X^2) - [E(X)]^2
                climatological_var: torch.Tensor = (climatological_sq_mean - climatological_mean ** 2)
                assert climatological_var.shape == (365, 192, 288)
                self.climatological_std[sim_id][var_name] = torch.sqrt(climatological_var.clamp(min=1e-12))
                del climatological_sq_mean, climatological_var

    def load_state(self, train_metadata: MetaData) -> None:
        with torch.serialization.safe_globals([defaultdict]):
            for sim_id in train_metadata.sim_ids:
                suffix: str = f"{sim_id}_{train_metadata.start_year}_{train_metadata.end_year}.pt"
                self.climatological_mean[sim_id] = torch.load(f=train_metadata.climatology_state_directory.joinpath(f"mean_{suffix}"), weights_only=False)
                self.climatological_std[sim_id] = torch.load(f=train_metadata.climatology_state_directory.joinpath(f"std_{suffix}"), weights_only=False)

    def save_state(self, train_metadata: MetaData) -> None:
        for sim_id in train_metadata.sim_ids:
            suffix: str = f"{sim_id}_{train_metadata.start_year}_{train_metadata.end_year}.pt"
            torch.save(obj=self.climatological_mean[sim_id], f=train_metadata.climatology_state_directory.joinpath(f"mean_{suffix}"))
            torch.save(obj=self.climatological_std[sim_id], f=train_metadata.climatology_state_directory.joinpath(f"std_{suffix}"))


class ClimatologyRemover:

    def __init__(self, metadata: MetaData):
        self.metadata: MetaData = metadata
        assert self.metadata.climatological_window_size % 2 == 1, "window size must be an odd number"
        os.makedirs(name=self.metadata.climatology_state_directory, exist_ok=True)

    def __call__(self, input_container: DataContainer, train_metadata: MetaData | None = None) -> None:
        """
        Inplace operation to save memory
        """
        cm: _ClimatologicalMean = _ClimatologicalMean()
        if self.metadata.tp == "train":
            # fit new cm (during training)
            assert train_metadata is None
            cm.fit(input_container=input_container, train_metadata=self.metadata)
            cm.save_state(train_metadata=self.metadata)
        else:
            # load saved cm (during inference)
            assert isinstance(train_metadata, MetaData)
            cm.load_state(train_metadata=train_metadata)

        for sim_id in self.metadata.sim_ids:
            for var_name in self.metadata.var_names:
                # compute standardized anomalies
                year_tensors: Dict[str, torch.Tensor] = input_container.get(sim_id=sim_id, var_name=var_name, year=None)
                year_tensors: List[torch.Tensor] = [year_tensors[k] for k in sorted(year_tensors)]
                assert all(tensor.shape == (365, 192, 288) for tensor in year_tensors)
                input_tensor: torch.Tensor = torch.stack(year_tensors, dim=0)
                assert input_tensor.shape == (self.metadata.n_years, 365, 192, 288)
                anomaly_tensor: torch.Tensor = input_tensor - cm.climatological_mean[sim_id][var_name]
                standardized_anomaly_tensor: torch.Tensor = anomaly_tensor / cm.climatological_std[sim_id][var_name]
                assert standardized_anomaly_tensor.shape == (self.metadata.n_years, 365, 192, 288)
                for i, year in enumerate(self.metadata.years):
                    input_container.set(sim_id=sim_id, var_name=var_name, year=year, value=standardized_anomaly_tensor[i])
                

