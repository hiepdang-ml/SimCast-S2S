import os
from collections import defaultdict
from typing import cast

import torch

from .container import VariableContainer
from common.configs import MetaData


class _LinearRegressor:

    def __init__(self, var_name: str, resolution: tuple[int, int]) -> None:
        self.var_name: str = var_name
        self.resolution: tuple[int, int] = resolution
        self.H, self.W = resolution
        self.lr_weight: dict[str, torch.Tensor] = {}  # sim_ids

    def fit(self, mean_container: VariableContainer, train_metadata: MetaData) -> None:
        assert self.var_name == mean_container.var_name
        assert (not bool(self.lr_weight)), "Linear regression is already fit"
        self.train_metadata: MetaData = train_metadata
        X: torch.Tensor = self.__get_X(metadata=train_metadata)
        assert X.shape == (train_metadata.n_years, 2)
        for sim_id in train_metadata.sim_ids:
            tensors: list[torch.Tensor] = [
                cast(torch.Tensor, mean_container.get(sim_id=sim_id, year=year)) for year in train_metadata.years
            ]
            y: torch.Tensor = torch.cat(tensors, dim=0)
            assert y.shape == (train_metadata.n_years, self.H, self.W)
            y = y.reshape(train_metadata.n_years, self.H * self.W).to(X.dtype)
            # torch.linalg.lstsq requires float64 to produce correct fit
            fit = torch.linalg.lstsq(X, y)
            assert fit.rank.item() == 2 # means the solver retains both coefficients, rank 1 means it produces a poor fit.
            W: torch.Tensor = fit.solution
            assert W.shape == (2, self.H * self.W)
            self.lr_weight[sim_id] = W

    def __call__(self, new_metadata: MetaData) -> VariableContainer:
        X: torch.Tensor = self.__get_X(metadata=new_metadata)
        assert X.shape == (new_metadata.n_years, 2)
        result: VariableContainer = VariableContainer(var_name=self.var_name, metadata=new_metadata)
        for sim_id in new_metadata.sim_ids:
            y_bar: torch.Tensor = X.matmul(self.lr_weight[sim_id])
            y_bar = y_bar.reshape(new_metadata.n_years, self.H, self.W)
            for i, year in enumerate(new_metadata.years):
                result.set(sim_id=sim_id, year=year, value=y_bar[i])
        return result

    @staticmethod
    def __get_X(metadata: MetaData) -> torch.Tensor:
        return torch.stack([
            torch.ones(metadata.n_years, dtype=torch.float64, device="cpu"),
                torch.tensor(metadata.years, dtype=torch.float64, device="cpu")
            ],
            dim=1
        )

    def load_state(self, train_metadata: MetaData) -> None:
        with torch.serialization.safe_globals([defaultdict]):
            # collect the state of all sim ids
            for sim_id in train_metadata.sim_ids:
                filename: str = (
                    f"{sim_id}_{self.var_name}_{train_metadata.start_year}_{train_metadata.end_year}.pt"
                )
                self.lr_weight[sim_id] = torch.load(
                    f=train_metadata.detrender_state_directory.joinpath(filename), weights_only=False
                )

    def save_state(self, train_metadata: MetaData) -> None:
        for sim_id in train_metadata.sim_ids:
            filename: str = (
                f"{sim_id}_{self.var_name}_{train_metadata.start_year}_{train_metadata.end_year}.pt"
            )
            torch.save(obj=self.lr_weight[sim_id], f=train_metadata.detrender_state_directory.joinpath(filename))


class Detrender:

    def __init__(self, metadata: MetaData) -> None:
        self.metadata: MetaData = metadata
        self.H, self.W = self.metadata.resolution
        os.makedirs(name=self.metadata.detrender_state_directory, exist_ok=True)

    def __call__(
        self, input_container: VariableContainer, train_metadata: MetaData | None = None,
        *, context_container: VariableContainer | None = None,
    ) -> None:
        """
        Inplace operation to save memory
        """
        lr: _LinearRegressor = _LinearRegressor(
            var_name=input_container.var_name, resolution=self.metadata.resolution
        )
        if self.metadata.tp == "train":
            # fit new lr (should be during training)
            assert train_metadata is None
            mean_container: VariableContainer = input_container.yearly_agg(reduce_func="mean")
            lr.fit(mean_container=mean_container, train_metadata=self.metadata)
            lr.save_state(train_metadata=self.metadata)
        else:
            # load saved lr (should be during inference)
            assert isinstance(train_metadata, MetaData)
            assert input_container.var_name in train_metadata.var_names
            lr.load_state(train_metadata=train_metadata)

        # Context is transformed with the same fit, using its actual calendar year.
        for container in (input_container, context_container):
            if container is None:
                continue
            assert container.var_name == input_container.var_name
            trend_container: VariableContainer = lr(new_metadata=container.metadata)
            for sim_id, year in container.metadata.combinations:
                input_tensor = cast(torch.Tensor, container.get(sim_id=sim_id, year=year))
                assert input_tensor.shape == (365, self.H, self.W)
                trend_tensor64 = cast(torch.Tensor, trend_container.get(sim_id=sim_id, year=year))
                assert trend_tensor64.dtype == torch.float64    # trend_container contains float64 tensors
                trend_tensor = trend_tensor64.to(input_tensor.dtype)
                assert trend_tensor.shape == self.metadata.resolution
                detrended = (input_tensor - trend_tensor[None, :, :])
                container.set(sim_id=sim_id, year=year, value=detrended)


class _ClimatologicalMean:

    def __init__(self, var_name: str, resolution: tuple[int, int]):
        self.var_name: str = var_name
        self.resolution: tuple[int, int] = resolution
        self.H, self.W = self.resolution
        self.climatological_mean: dict[str, torch.Tensor] = {}  # sim_ids
        self.climatological_std : dict[str, torch.Tensor] = {}  # sim_ids

    def fit(self, input_container: VariableContainer, train_metadata: MetaData) -> None:
        assert self.var_name == input_container.var_name
        assert input_container.var_name in train_metadata.var_names
        assert not bool(self.climatological_mean), "climatological_mean is already computed"
        n_years: int = train_metadata.n_years

        for sim_id in train_metadata.sim_ids:
            # retrieve input data (ensure ascending year)
            tensors_by_year: dict[int, torch.Tensor] = cast(dict[int, torch.Tensor], input_container.get(sim_id=sim_id, year=None))
            year_tensors: list[torch.Tensor] = [tensors_by_year[k] for k in train_metadata.years]
            assert all(tensor.shape == (365, self.H, self.W) for tensor in year_tensors)
            input_tensor: torch.Tensor = torch.stack(year_tensors, dim=0)
            assert input_tensor.shape == (n_years, 365, self.H, self.W)
            self.climatological_mean[sim_id] = input_tensor.mean(dim=0)
            self.climatological_std[sim_id] = input_tensor.var(dim=0).sqrt().clamp(min=1e-12)

    def load_state(self, train_metadata: MetaData) -> None:
        with torch.serialization.safe_globals([defaultdict]):
            for sim_id in train_metadata.sim_ids:
                suffix: str = f"{sim_id}_{self.var_name}_{train_metadata.start_year}_{train_metadata.end_year}.pt"
                self.climatological_mean[sim_id] = torch.load(f=train_metadata.climatology_state_directory.joinpath(f"mean_{suffix}"), weights_only=False)
                self.climatological_std[sim_id] = torch.load(f=train_metadata.climatology_state_directory.joinpath(f"std_{suffix}"), weights_only=False)

    def save_state(self, train_metadata: MetaData) -> None:
        for sim_id in train_metadata.sim_ids:
            suffix: str = f"{sim_id}_{self.var_name}_{train_metadata.start_year}_{train_metadata.end_year}.pt"
            torch.save(obj=self.climatological_mean[sim_id], f=train_metadata.climatology_state_directory.joinpath(f"mean_{suffix}"))
            torch.save(obj=self.climatological_std[sim_id], f=train_metadata.climatology_state_directory.joinpath(f"std_{suffix}"))


class ClimatologyRemover:

    def __init__(self, metadata: MetaData):
        self.metadata: MetaData = metadata
        self.H, self.W = self.metadata.resolution
        os.makedirs(name=self.metadata.climatology_state_directory, exist_ok=True)

    def __call__(
        self, input_container: VariableContainer, train_metadata: MetaData | None = None,
        *, context_container: VariableContainer | None = None,
    ) -> None:
        """
        Inplace operation to save memory
        """
        cm: _ClimatologicalMean = _ClimatologicalMean(
            var_name=input_container.var_name, resolution=self.metadata.resolution
        )
        if self.metadata.tp == "train":
            # fit new cm (during training)
            assert train_metadata is None
            cm.fit(input_container=input_container, train_metadata=self.metadata)
            cm.save_state(train_metadata=self.metadata)
        else:
            # load saved cm (during inference)
            assert isinstance(train_metadata, MetaData)
            assert input_container.var_name in train_metadata.var_names
            cm.load_state(train_metadata=train_metadata)

        for container in (input_container, context_container):
            if container is None:
                continue
            assert container.var_name == input_container.var_name
            for sim_id in container.metadata.sim_ids:
                tensors_by_year = cast(dict[int, torch.Tensor], container.get(sim_id=sim_id, year=None))
                year_tensors = [tensors_by_year[k] for k in container.metadata.years]
                assert all(tensor.shape == (365, self.H, self.W) for tensor in year_tensors)
                input_tensor = torch.stack(year_tensors, dim=0)
                anomaly_tensor = input_tensor - cm.climatological_mean[sim_id]
                standardized_anomaly_tensor = anomaly_tensor / cm.climatological_std[sim_id]
                for i, year in enumerate(container.metadata.years):
                    container.set(sim_id=sim_id, year=year, value=standardized_anomaly_tensor[i])
