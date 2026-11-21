import os
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, cast

from sympy.ntheory.digits import defaultdict
import torch
from torch.nn import functional as F

from .container import VariableContainer
from common.configs import FunctionalTransformName, MetaData


class FunctionalTransform:

    def __init__(self, metadata: MetaData) -> None:
        self.functions: dict[str, FunctionalTransformName] = metadata.functional_transforms

    def transform(self, input_tensor: torch.Tensor, *, var_name: str) -> torch.Tensor:
        """Apply the configured pointwise function to a raw daily tensor."""
        # Default to "identity"
        function: FunctionalTransformName = self.functions.get(var_name, "identity")
        if function == "identity":
            return input_tensor

        if function == "log":
            invalid_mask: torch.Tensor = input_tensor <= 0
            requirement: str = "positive"
            invalid_kind: str = "nonpositive"
            operation: Callable[[], torch.Tensor] = input_tensor.log
        else:
            invalid_mask = input_tensor < 0
            requirement = "nonnegative"
            invalid_kind = "negative"
            operation = input_tensor.sqrt

        if invalid_mask.any():
            invalid_count: int = int(invalid_mask.count_nonzero().item())
            raise ValueError(
                f"The {function} transform for {var_name!r} requires {requirement} values. "
                f"Found {invalid_count} {invalid_kind} values."
            )
        del invalid_mask
        return operation()

    def inverse_transform(self, input_tensor: torch.Tensor, *, var_name: str) -> torch.Tensor:
        """
        Args:
            input_tensor: Values in the output space of ``transform()``.
            var_name: Variable whose configured function to invert.
        """
        # Default to "identity"
        function: FunctionalTransformName = self.functions.get(var_name, "identity")
        if function == "log":
            return input_tensor.exp()
        if function == "sqrt":
            return input_tensor.square()
        return input_tensor

    @staticmethod
    def state_suffix(metadata: MetaData, var_name: str) -> str:
        """Identify the raw-data function used to fit saved preprocessing state."""
        function: FunctionalTransformName = metadata.functional_transforms.get(var_name, "identity")
        return function


class TrendLine(TypedDict):
    window_start_year: int
    window_end_year: int
    a: torch.Tensor
    b: torch.Tensor


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
                # Compute the mean calendar year (first year + mean position).
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
                f=Detrender.state_path(
                    metadata=metadata, var_name=self.var_name, sim_id=sim_id
                ),
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
                f=Detrender.state_path(
                    metadata=metadata, var_name=self.var_name, sim_id=sim_id
                ),
            )


class Detrender:

    def __init__(self, metadata: MetaData) -> None:
        self.metadata: MetaData = metadata
        self.H, self.W = self.metadata.resolution
        os.makedirs(name=self.metadata.detrender_state_directory, exist_ok=True)

    @staticmethod
    def state_path(metadata: MetaData, var_name: str, sim_id: str) -> Path:
        """Return the state file shared by all train/validation/test splits."""
        suffix: str = FunctionalTransform.state_suffix(metadata, var_name)
        filename: str = (
            f"trend_{sim_id}_{var_name}_"
            f"{metadata.detrend_start_year}_{metadata.detrend_end_year}_"
            f"window{metadata.detrend_window_n_years}y_{suffix}.pt"
        )
        return metadata.detrender_state_directory.joinpath(filename)

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
        self._apply_from_saved_state(input_container, context_container=context_container, operation=torch.sub)

    def inverse_transform(
        self, input_container: VariableContainer,
        *, context_container: VariableContainer | None = None,
    ) -> None:
        """Restore the saved target-year trend to detrended daily fields in place.

        Args:
            input_container: Detrended daily data for one variable and split.
                Each yearly tensor has shape ``(365, H, W)``.
            context_container: Optional detrended data for the same variable
                from the preceding year. Its trend is restored in place using
                its own calendar year.
        """
        self._apply_from_saved_state(input_container, context_container=context_container, operation=torch.add)

    def _apply_from_saved_state(
        self, input_container: VariableContainer,
        *, context_container: VariableContainer | None,
        operation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        lr = _TrendFromLinearRegression(
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
                transformed: torch.Tensor = operation(input_tensor, trend_tensor[None, :, :])
                container.set(sim_id=sim_id, year=year, value=transformed)


class SeasonalCycle:

    def __init__(self, metadata: MetaData) -> None:
        self.H, self.W = metadata.resolution
        self.seasonal_cycles: dict[str, dict[str, torch.Tensor]] = {}  # var_name, sim_id
        os.makedirs(name=metadata.seasonal_state_directory, exist_ok=True)

    @staticmethod
    def state_path(metadata: MetaData, var_name: str, sim_id: str) -> Path:
        """Return the saved seasonal cycle for this training period."""
        suffix: str = FunctionalTransform.state_suffix(metadata, var_name)
        filename: str = f"seasonal_{sim_id}_{var_name}_{metadata.start_year}_{metadata.end_year}_{suffix}.pt"
        return metadata.seasonal_state_directory.joinpath(filename)

    def fit(self, input_container: VariableContainer, train_metadata: MetaData) -> None:
        """Store the fitted seasonal cycle for this variable and each simulation."""
        assert input_container.var_name in train_metadata.var_names
        seasonal_cycles: dict[str, torch.Tensor] = {}
        n_years: int = train_metadata.n_years
        window_days: int = train_metadata.seasonal_smoothing_window_n_days
        half_window_days: int = window_days // 2

        for sim_id in train_metadata.sim_ids:
            # retrieve input data (ensure ascending year)
            tensors_by_year: dict[int, torch.Tensor] = cast(dict[int, torch.Tensor], input_container.get(sim_id=sim_id, year=None))
            year_tensors: list[torch.Tensor] = [tensors_by_year[k] for k in train_metadata.years]
            assert all(tensor.shape == (365, self.H, self.W) for tensor in year_tensors)

            # Smooth only for the seasonal mean. Use adjacent training years at Dec/Jan
            # and shorter windows at the ends of the training period.
            series: torch.Tensor = torch.cat(year_tensors, dim=0)
            assert series.shape == (n_years * 365, self.H, self.W)
            seasonal_sum: torch.Tensor = torch.zeros_like(year_tensors[0])
            for year_start in range(0, series.shape[0], 365):
                context_start: int = max(0, year_start - half_window_days)
                days: torch.Tensor = series[context_start:year_start + 365 + half_window_days]
                # NOTE: With a seven-day window and at least two training years, days
                # contains 371 days for interior years and 368 for the first and last years
                smoothed: torch.Tensor = F.avg_pool1d(
                    days.flatten(start_dim=1, end_dim=-1).T.unsqueeze(0),
                    kernel_size=window_days, stride=1, padding=half_window_days, count_include_pad=False,
                )
                start: int = year_start - context_start
                smoothed_year: torch.Tensor = smoothed[0, :, start:start + 365]
                assert smoothed_year.shape == (self.H * self.W, 365)
                seasonal_sum.add_(smoothed_year.T.reshape_as(seasonal_sum))
                del days
            seasonal_cycles[sim_id] = seasonal_sum / n_years
            del series
        self.seasonal_cycles[input_container.var_name] = seasonal_cycles

    def load_state(self, train_metadata: MetaData) -> None:
        """Load the seasonal cycles for all configured variables and simulations."""
        self.seasonal_cycles.clear()
        for var_name in train_metadata.var_names:
            cycles_by_sim: dict[str, torch.Tensor] = {}
            for sim_id in train_metadata.sim_ids:
                cycles_by_sim[sim_id] = torch.load(
                    f=SeasonalCycle.state_path(
                        metadata=train_metadata, var_name=var_name, sim_id=sim_id
                    ),
                    weights_only=True,
                )
            self.seasonal_cycles[var_name] = cycles_by_sim

    def save_state(self, train_metadata: MetaData) -> None:
        """Save the stored cycles using one file per variable and simulation."""
        for var_name, cycles_by_sim in self.seasonal_cycles.items():
            for sim_id, cycle in cycles_by_sim.items():
                assert cycle.shape == (365, self.H, self.W)
                torch.save(
                    obj=cycle,
                    f=SeasonalCycle.state_path(
                        metadata=train_metadata, var_name=var_name, sim_id=sim_id
                    )
                )

    def transform(
        self, input_container: VariableContainer,
        *, context_container: VariableContainer | None = None,
    ) -> None:
        """
        Subtract the stored seasonal cycle from detrended daily data in place for
        one variable in ``input_container`` and ``context_container``

        Call ``fit()`` or ``load_state()`` before calling this method.
        Each simulation uses its own fitted calendar-day cycle.

        Args:
            input_container: Unfiltered, detrended data for one variable.
                Each yearly tensor has shape ``(365, H, W)``.
            context_container: Optional detrended data for the same variable
                from the year preceding the split. It is transformed in place
                using the same seasonal cycles.
        """
        self._apply_from_saved_state(input_container, context_container=context_container, operation=torch.sub)

    def inverse_transform(
        self, input_container: VariableContainer,
        *, context_container: VariableContainer | None = None,
    ) -> None:
        """Restore the fitted calendar-day seasonal cycle to daily anomalies in place.

        Call ``fit()`` or ``load_state()`` before transforming a variable.

        Args:
            input_container: Daily anomalies for one variable, in
                ``(365, H, W)`` tensors.
            context_container: Optional anomalies for the same variable from
                the year preceding the split. The seasonal cycle is restored
                in place using the same fitted state.
        """
        self._apply_from_saved_state(input_container, context_container=context_container, operation=torch.add)

    def _apply_from_saved_state(
        self, input_container: VariableContainer,
        *, context_container: VariableContainer | None,
        operation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        var_name: str = input_container.var_name
        if var_name not in self.seasonal_cycles:
            raise RuntimeError(
                f"No fitted seasonal cycle for {var_name!r}. "
                "Call fit() or load_state() before applying the seasonal cycle."
            )
        seasonal_cycles: dict[str, torch.Tensor] = self.seasonal_cycles[var_name]

        for container in (input_container, context_container):
            if container is None:
                continue
            assert container.var_name == var_name
            for sim_id in container.metadata.sim_ids:
                tensors_by_year = cast(dict[int, torch.Tensor], container.get(sim_id=sim_id, year=None))
                year_tensors = [tensors_by_year[k] for k in container.metadata.years]
                assert all(tensor.shape == (365, self.H, self.W) for tensor in year_tensors)
                input_tensor: torch.Tensor = torch.stack(year_tensors, dim=0)
                transformed: torch.Tensor = operation(input_tensor, seasonal_cycles[sim_id])
                for i, year in enumerate(container.metadata.years):
                    container.set(sim_id=sim_id, year=year, value=transformed[i])


class Standardization:

    def __init__(self, metadata: MetaData) -> None:
        self.H, self.W = metadata.resolution
        self.standard_deviations: dict[str, dict[str, torch.Tensor]] = {}  # var_name, sim_id
        os.makedirs(name=metadata.standardization_state_directory, exist_ok=True)

    @staticmethod
    def state_path(metadata: MetaData, var_name: str, sim_id: str) -> Path:
        """Return the saved standard deviations for this training period and window."""
        suffix: str = FunctionalTransform.state_suffix(metadata, var_name)
        filename: str = (
            f"standardization_{sim_id}_{var_name}_{metadata.start_year}_{metadata.end_year}_"
            f"window{metadata.standardization_window_n_days}d_{suffix}.pt"
        )
        return metadata.standardization_state_directory.joinpath(filename)

    def fit(self, input_container: VariableContainer, train_metadata: MetaData) -> None:
        """Fit a sample standard deviation for each calendar day and grid cell."""
        assert input_container.var_name in train_metadata.var_names
        window_days: int = train_metadata.standardization_window_n_days
        half_window_days: int = window_days // 2
        wrap_start: int = 365 - half_window_days
        sample_count: int = window_days * train_metadata.n_years
        # A one-sample window has zero variance and uses the unit scale below.
        sample_correction: float = sample_count / max(1, sample_count - 1)
        std_by_sim: dict[str, torch.Tensor] = {}

        for sim_id in train_metadata.sim_ids:
            year_tensors: list[torch.Tensor] = [
                cast(torch.Tensor, input_container.get(sim_id=sim_id, year=year))
                for year in train_metadata.years
            ]
            assert all(tensor.shape == (365, self.H, self.W) for tensor in year_tensors)
            anomalies: torch.Tensor = torch.stack(year_tensors, dim=0)
            assert anomalies.shape == (train_metadata.n_years, 365, self.H, self.W)
            daily_variance: torch.Tensor
            daily_mean: torch.Tensor
            daily_variance, daily_mean = torch.var_mean(anomalies, dim=0, correction=0)
            del anomalies

            # Total variance combines variation within each calendar day and
            # variation between the day means. Wrap the 365-day calendar.
            assert daily_variance.shape == (365, self.H, self.W)
            wrapped_variance: torch.Tensor = torch.cat(
                tensors=(
                    daily_variance[wrap_start:], # shape: (half_window_days, self.H, self.W)
                    daily_variance, # shape: (365, self.H, self.W)
                    daily_variance[:half_window_days], # shape: (half_window_days, self.H, self.W)
                ),
                dim=0
            )
            wrapped_mean: torch.Tensor = torch.cat(
                tensors=(
                    daily_mean[wrap_start:], # shape: (half_window_days, self.H, self.W)
                    daily_mean, # shape: (365, self.H, self.W)
                    daily_mean[:half_window_days], # shape: (half_window_days, self.H, self.W)
                ),
                dim=0
            )
            # Compute within_day variance: \mathbb{E}_{k \in W_d}[\sigma_k^2]
            within_days_unfold: torch.Tensor = wrapped_variance.unfold(dimension=0, size=window_days, step=1)
            assert within_days_unfold.shape == (365, self.H, self.W, window_days)
            within_days_variance: torch.Tensor = within_days_unfold.mean(dim=-1)
            # Compute between_day variance: E_{k \in W_d}[(\mu_k - \bar{\mu}_d)^2]
            between_days_unfold: torch.Tensor = wrapped_mean.unfold(dimension=0, size=window_days, step=1)
            assert between_days_unfold.shape == (365, self.H, self.W, window_days)
            between_days_variance: torch.Tensor = between_days_unfold.var(dim=-1, correction=0)
            # Compute total variance: Var(X_d) = within_day variance + between_day variance
            sample_variance: torch.Tensor = (within_days_variance + between_days_variance) * sample_correction
            assert sample_variance.shape == (365, self.H, self.W)
            std_by_sim[sim_id] = sample_variance.sqrt()

        self.standard_deviations[input_container.var_name] = std_by_sim

    def load_state(self, train_metadata: MetaData) -> None:
        """Load standardization scales for all configured variables and simulations."""
        self.standard_deviations.clear()
        for var_name in train_metadata.var_names:
            std_by_sim: dict[str, torch.Tensor] = {}
            for sim_id in train_metadata.sim_ids:
                std_by_sim[sim_id] = cast(
                    torch.Tensor,
                    torch.load(
                        f=Standardization.state_path(
                            metadata=train_metadata, var_name=var_name, sim_id=sim_id
                        ),
                        weights_only=True
                    ),
                )
            self.standard_deviations[var_name] = std_by_sim

    def save_state(self, train_metadata: MetaData) -> None:
        """Save one tensor of calendar-day scales per variable and simulation."""
        for var_name, std_by_sim in self.standard_deviations.items():
            for sim_id, std in std_by_sim.items():
                assert std.shape == (365, self.H, self.W)
                torch.save(obj=std, f=Standardization.state_path(train_metadata, var_name, sim_id))

    def transform(
        self, input_container: VariableContainer,
        *, context_container: VariableContainer | None = None,
    ) -> None:
        """Standardize daily anomalies in place using fitted standard deviations.

        Call ``fit()`` or ``load_state()`` before transforming a variable.
        Each anomaly is divided by the deviation for its simulation,
        calendar day, and grid cell.

        Args:
            input_container: Detrended and deseasonalized data for one variable.
                Each yearly tensor has shape ``(365, H, W)``.
            context_container: Optional anomalies for the same variable from the
                year preceding the split. They are standardized in place using
                the same fitted deviations.
        """
        self._apply_from_saved_state(input_container, context_container=context_container, operation=torch.div)

    def inverse_transform(
        self, input_container: VariableContainer,
        *, context_container: VariableContainer | None = None,
    ) -> None:
        """Restore the anomaly scale for standardized daily data in place.

        Call ``fit()`` or ``load_state()`` first. Each standardized value is
        multiplied by the same calendar-day scale used by ``transform()``.
        The optional context container is restored in place using the same scales.
        """
        self._apply_from_saved_state(input_container, context_container=context_container, operation=torch.mul)

    def _apply_from_saved_state(
        self, input_container: VariableContainer,
        *, context_container: VariableContainer | None,
        operation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        var_name: str = input_container.var_name
        if var_name not in self.standard_deviations:
            raise RuntimeError(
                f"No fitted standard deviations for {var_name!r}. "
                "Call fit() or load_state() before applying standardization."
            )
        std_by_sim: dict[str, torch.Tensor] = self.standard_deviations[var_name]

        for container in (input_container, context_container):
            if container is None:
                continue
            assert container.var_name == var_name
            for sim_id in container.metadata.sim_ids:
                std: torch.Tensor = std_by_sim[sim_id]
                # for consistently dry grids, do not divide => set std = 1
                scale: torch.Tensor = torch.where(std == 0, 1.0, std)
                for year in container.metadata.years:
                    anomalies: torch.Tensor = cast(
                        torch.Tensor, container.get(sim_id=sim_id, year=year)
                    )
                    assert anomalies.shape == (365, self.H, self.W)
                    transformed: torch.Tensor = operation(anomalies, scale)
                    container.set(sim_id=sim_id, year=year, value=transformed)
