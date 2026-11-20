import os
from collections import defaultdict
from pathlib import Path
from typing import TypedDict, cast

import torch

from .container import VariableContainer
from common.configs import MetaData


class TrendLine(TypedDict):
    window_start_year: int
    window_end_year: int
    a: torch.Tensor
    b: torch.Tensor


def detrender_state_path(metadata: MetaData, var_name: str, sim_id: str) -> Path:
    """Return the state file shared by all train/validation/test splits."""
    filename: str = (
        f"{sim_id}_{var_name}_"
        f"{metadata.detrend_start_year}_{metadata.detrend_end_year}_"
        f"window{metadata.detrend_window_n_years}.pt"
    )
    return metadata.detrender_state_directory.joinpath(filename)


class _TrendFromLinearRegression:

    def __init__(self, var_name: str, resolution: tuple[int, int]) -> None:
        self.var_name: str = var_name
        self.resolution: tuple[int, int] = resolution
        self.H, self.W = resolution
        # {sim_id, year} -> TrendLine
        self.trend_lines: dict[str, dict[int, TrendLine]] = {}

    def fit(self, mean_container: VariableContainer, metadata: MetaData) -> None:
        """
        NOTE:
        Before fitting, exportdata.py creates `metadata=full_metadata` by:
        ```
        full_metadata.start_year = first_train_year
        full_metadata.end_year = last_test_year
        full_metadata.years = train_years + val_years + test_years
        ```
        """
        assert self.var_name == mean_container.var_name
        assert not self.trend_lines, "Linear regression is already fit"
        years: list[int] = metadata.years
        detrend_window_n_years: int = metadata.detrend_window_n_years
        if years != list(range(metadata.start_year, metadata.end_year + 1)):
            raise ValueError("Detrending requires consecutive years in ascending order")
        if len(years) < detrend_window_n_years:
            raise ValueError(
                f"Cannot fit a {detrend_window_n_years}-year detrending window to only {len(years)} years"
            )

        # Define t
        positions: torch.Tensor = torch.arange(
            detrend_window_n_years, dtype=torch.float64, device="cpu",
        )
        # Compute (t-t̄)
        centered_positions: torch.Tensor = positions - positions.mean()
        # Compute ∑(t-t̄)²
        denominator: torch.Tensor = centered_positions.square().sum()
        assert denominator > 0

        for sim_id in metadata.sim_ids:
            tensors: list[torch.Tensor] = [
                cast(torch.Tensor, mean_container.get(sim_id=sim_id, year=year))
                for year in years
            ]
            if not all(tensor.shape == (1, self.H, self.W) for tensor in tensors):
                raise ValueError("Each yearly mean must have shape (1, H, W)")
            annual_means = torch.cat(tensors, dim=0).to(torch.float64)
            assert annual_means.shape == (metadata.n_years, self.H, self.W)

            # Keep rolling sums so every line is fit in O(H*W), rather than
            # solving a separate least-squares problem over the whole window.
            window_values: torch.Tensor = annual_means[:detrend_window_n_years]
            # Compute ∑yₜ
            rolling_sum: torch.Tensor = window_values.sum(dim=0)
            assert rolling_sum.shape == (self.H, self.W)
            # Compute ∑tyₜ
            rolling_weighted_sum: torch.Tensor = torch.einsum(
                "t,thw->hw", positions, window_values
            )
            assert rolling_weighted_sum.shape == (self.H, self.W)
            lines_by_window_start: list[TrendLine] = []
            n_window_starts: int = max(1, metadata.n_years - detrend_window_n_years)
            for window_start in range(n_window_starts):
                # Compute ∑(t-t̄)y = ∑(yt)-∑(yt̄)
                numerator: torch.Tensor = rolling_weighted_sum - positions.mean() * rolling_sum
                # Compute b=∑(t-t̄)y/∑(t-t̄)²
                b: torch.Tensor = numerator / denominator
                # Compute ȳ
                value_at_mean_year: torch.Tensor = rolling_sum / detrend_window_n_years
                # Compute
                mean_year: float = float(years[window_start]) + positions.mean().item()
                a: torch.Tensor = value_at_mean_year - mean_year * b
                lines_by_window_start.append({
                    "window_start_year": years[window_start],
                    "window_end_year": years[window_start + detrend_window_n_years - 1],
                    "a": a,
                    "b": b,
                })

                next_start: int = window_start + 1
                if next_start < n_window_starts:
                    outgoing: torch.Tensor = annual_means[window_start]
                    incoming: torch.Tensor = annual_means[window_start + detrend_window_n_years]
                    old_sum: torch.Tensor = rolling_sum
                    rolling_sum = old_sum - outgoing + incoming
                    rolling_weighted_sum = (
                        rolling_weighted_sum
                        - (old_sum - outgoing)
                        + (detrend_window_n_years - 1) * incoming
                    )

            # Example: With a 30-year window
            # Year 1->30    uses years 1--30
            # Year 31       uses years 1--30
            # Year 32       uses years 2--31
            # Year 33       uses years 3--32
            target_lines: dict[int, TrendLine] = {}
            for target_index, mean_year in enumerate(years):
                window_start = max(0, target_index - detrend_window_n_years)
                target_lines[mean_year] = lines_by_window_start[window_start]
            self.trend_lines[sim_id] = target_lines

    def transform(self, new_metadata: MetaData) -> VariableContainer:
        result: VariableContainer = VariableContainer(var_name=self.var_name, metadata=new_metadata)
        for sim_id in new_metadata.sim_ids:
            if sim_id not in self.trend_lines:
                raise KeyError(f"No detrending state for simulation {sim_id!r}")
            for year in new_metadata.years:
                if year not in self.trend_lines[sim_id]:
                    raise KeyError(f"No detrending line for target year {year}")
                line: TrendLine = self.trend_lines[sim_id][year]
                trend = line["a"] + year * line["b"]
                assert trend.shape == (self.H, self.W)
                result.set(sim_id=sim_id, year=year, value=trend)
        return result

    def load_state(self, metadata: MetaData) -> None:
        for sim_id in metadata.sim_ids:
            state = torch.load(
                f=detrender_state_path(metadata, self.var_name, sim_id),
                weights_only=True,
                map_location="cpu",
            )
            if set(state) != set(range(metadata.detrend_start_year, metadata.detrend_end_year + 1)):
                raise ValueError(f"Detrending state has unexpected target years for {sim_id!r}")
            self.trend_lines[sim_id] = state

    def save_state(self, metadata: MetaData) -> None:
        for sim_id in metadata.sim_ids:
            torch.save(
                obj=self.trend_lines[sim_id],
                f=detrender_state_path(metadata, self.var_name, sim_id),
            )


class Detrender:

    def __init__(self, metadata: MetaData) -> None:
        self.metadata: MetaData = metadata
        self.H, self.W = self.metadata.resolution
        os.makedirs(name=self.metadata.detrender_state_directory, exist_ok=True)

    def fit(self, mean_container: VariableContainer) -> None:
        """Fit and save target-year trend lines from full-dataset yearly means."""
        lr = _TrendFromLinearRegression(
            var_name=mean_container.var_name, resolution=self.metadata.resolution
        )
        lr.fit(mean_container=mean_container, metadata=self.metadata)
        lr.save_state(metadata=self.metadata)

    def transform(
        self, input_container: VariableContainer,
        *, context_container: VariableContainer | None = None,
    ) -> None:
        """
        Remove the saved target-year linear trend from daily fields in place.
        This method performs inplace operation to save memory.

        Args:
            input_container: Complete daily data for one variable and dataset
                split. Its tensors must have shape ``(365, H, W)`` and are
                replaced with detrended tensors.
            context_container: Optional data for the year preceding the start
                of a validation or test split. It is needed for samples near
                the beginning of the split.
                For example, an output beginning on January 1 may require
                84 input days plus a 14-day lead period extending into the
                previous year. It must contain the same variable as
                ``input_container`` and is also modified in place.
        """
        lr: _TrendFromLinearRegression = _TrendFromLinearRegression(
            var_name=input_container.var_name, resolution=self.metadata.resolution
        )
        lr.load_state(metadata=self.metadata)

        # Context is transformed with the same fit, using its actual calendar year.
        for container in (input_container, context_container):
            if container is None:
                continue
            assert container.var_name == input_container.var_name
            trend_container: VariableContainer = lr.transform(new_metadata=container.metadata)
            for sim_id, year in container.metadata.combinations:
                input_tensor = cast(torch.Tensor, container.get(sim_id=sim_id, year=year))
                assert input_tensor.shape == (365, self.H, self.W)
                trend_tensor = cast(
                    torch.Tensor, trend_container.get(sim_id=sim_id, year=year)
                ).to(input_tensor.dtype)
                assert trend_tensor.shape == self.metadata.resolution
                detrended: torch.Tensor = input_tensor - trend_tensor[None, :, :]
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
