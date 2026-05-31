from argparse import ArgumentParser, Namespace
from pathlib import Path
from functools import cached_property
import datetime as dt
import random
from typing import Any
import numpy as np
from numpy.typing import NDArray
import xarray as xr
import torch

from common.configs import MetaData, ECMWFS2SConfig
from common.metrics import GeographicalMAE, GeographicalMSE
from common.utils import TorchDictIO
from datapipeline.readers.era5 import ERA5Utilities
from datapipeline.utils import SampleInfo


class ECMWFReader:

    VAR_NAME: str = "tp"

    def __init__(
        self,
        root: str, n_lead_days: int, n_output_days: int,
        target_resolution: tuple[int, int],
        from_year: int,
        to_year: int,
    ) -> None:
        self.root: Path = Path(root)
        filepaths: list[Path] = sorted(self.root.glob("*.grib"))
        filepaths = [
            filepath for filepath in filepaths
            if from_year <= int(filepath.stem[:4]) <= to_year
        ]
        self.filepaths: list[Path] = filepaths
        assert len(self.filepaths) > 0, f"No ECMWF GRIB files found in {self.root}"
        self.n_lead_days: int = n_lead_days
        self.n_output_days: int = n_output_days
        self.target_resolution: tuple[int, int] = target_resolution

    @cached_property
    def output_steps(self) -> list[np.timedelta64]:
        start_day: int = self.n_lead_days + 1
        end_day: int = start_day + self.n_output_days
        return [np.timedelta64(d, "D") for d in range(start_day, end_day)]

    @cached_property
    def boundary_steps(self) -> list[np.timedelta64]:
        return [np.timedelta64(self.n_lead_days, "D")] + self.output_steps

    @staticmethod
    def resize(da: xr.DataArray, target_resolution: tuple[int, int]) -> xr.DataArray:
        newlat: NDArray[np.float64] = np.linspace(
            da.latitude.min().item(), da.latitude.max().item(), target_resolution[0]
        )
        newlon: NDArray[np.float64] = np.linspace(
            da.longitude.min().item(), da.longitude.max().item(), target_resolution[1]
        )
        return da.interp(latitude=newlat, longitude=newlon, method="linear")

    @staticmethod
    def sample_forecast_indices(da: xr.DataArray) -> list[int]:
        valid_time: xr.DataArray = da.coords["valid_time"]
        assert valid_time.dims == ("time", "step")
        start_years: NDArray[np.int64] = valid_time.isel(step=0).dt.year.values.astype(np.int64)
        end_years: NDArray[np.int64] = valid_time.isel(step=-1).dt.year.values.astype(np.int64)
        init_years: NDArray[np.int64] = da.time.dt.year.values.astype(np.int64)
        flag1: NDArray[np.bool_] = start_years == end_years
        flag2: NDArray[np.bool_] = init_years == start_years
        eligible_indices: list[int] = np.flatnonzero(flag1 & flag2).tolist()
        indices: list[int] = random.sample(eligible_indices, k=min(len(eligible_indices), 2))
        return indices

    def accumulated_tp_to_daily_totals(self, da: xr.DataArray) -> xr.DataArray:
        cumulative: xr.DataArray = da.sel(step=self.boundary_steps)
        daily_totals: xr.DataArray = cumulative.diff(dim="step", label="upper")
        assert daily_totals.dims == ("number", "time", "step", "latitude", "longitude")
        valid_time: xr.DataArray = da["valid_time"].sel(step=self.output_steps)
        daily_totals = daily_totals.assign_coords(valid_time=valid_time)
        return daily_totals

    def read(self) -> xr.DataArray:
        monthly_outputs: list[xr.DataArray] = []
        for filepath in self.filepaths:
            ds: xr.Dataset = xr.open_dataset(filepath, engine="cfgrib")
            raw_tp: xr.DataArray = ds[self.VAR_NAME]
            output_window: xr.DataArray = raw_tp.sel(step=self.output_steps)
            sampled_time_indices: list[int] = self.sample_forecast_indices(output_window)
            if len(sampled_time_indices) == 0:
                # edge months may not have enough eligible indices -> skip
                continue
            da: xr.DataArray = raw_tp.isel(time=sampled_time_indices)
            da = self.accumulated_tp_to_daily_totals(da)
            da = da / 1000.0 / 86400.0  # kg m^-2 -> m, then daily total -> m s^-1
            da = ERA5Utilities.fliplatitude(da)
            monthly_outputs.append(self.resize(da, self.target_resolution))
        return xr.concat(objs=monthly_outputs, dim="time")


class ECMWFPreprocessor:

    def __init__(self, train_metadata: MetaData) -> None:
        assert train_metadata.dataset_name == "era5"
        assert train_metadata.tp == "train"
        self.train_metadata: MetaData = train_metadata
        self.var_name: str = train_metadata.output_vars[0]
        self.sim_id: str = train_metadata.sim_ids[0]
        self.H, self.W = train_metadata.resolution

        suffix: str = f"{self.sim_id}_{self.var_name}_{train_metadata.start_year}_{train_metadata.end_year}.pt"
        self.lr_weight: NDArray[np.float32] = torch.load(
            f=train_metadata.detrender_state_directory.joinpath(suffix),
            weights_only=False,
            map_location="cpu",
        ).numpy().astype(np.float32, copy=False)
        self.climatological_mean: NDArray[np.float32] = torch.load(
            f=train_metadata.climatology_state_directory.joinpath(f"mean_{suffix}"),
            weights_only=False,
            map_location="cpu",
        ).numpy().astype(np.float32, copy=False)
        self.climatological_std: NDArray[np.float32] = torch.load(
            f=train_metadata.climatology_state_directory.joinpath(f"std_{suffix}"),
            weights_only=False,
            map_location="cpu",
        ).numpy().astype(np.float32, copy=False)
        assert self.lr_weight.shape == (2, self.H * self.W)
        assert self.climatological_mean.shape == (365, self.H, self.W)
        assert self.climatological_std.shape == (365, self.H, self.W)

    def preprocess_predictions(self, da: xr.DataArray) -> xr.DataArray:
        required_dims: tuple[str, ...] = ("number", "time", "step", "latitude", "longitude")
        assert da.dims == required_dims, f"Expected dims {required_dims}, got {da.dims}"
        feb29_mask: NDArray[np.bool_] = (
            (da.coords["valid_time"].dt.month.values == 2) &
            (da.coords["valid_time"].dt.day.values == 29)
        )
        keep_mask: NDArray[np.bool_] = ~np.any(feb29_mask, axis=1)
        keep_indices: NDArray[np.int64] = np.flatnonzero(keep_mask).astype(np.int64, copy=False)
        da = da.isel(time=keep_indices)
        n_members: int = da.sizes["number"]
        n_samples: int = da.sizes["time"]
        n_output_days: int = da.sizes["step"]
        assert da.shape == (n_members, n_samples, n_output_days, self.H, self.W), (
            "Unexpected prediction shape: "
            f"expected {(n_members, n_samples, n_output_days, self.H, self.W)}, got {da.shape}"
        )
        valid_time: xr.DataArray = da.coords["valid_time"]
        years: NDArray[np.float32] = valid_time.dt.year.values.astype(np.float32)
        dayofyears: NDArray[np.int64] = valid_time.dt.dayofyear.values.astype(np.int64)
        assert valid_time.shape == years.shape == dayofyears.shape == (n_samples, n_output_days)
        flattened: xr.DataArray = da.transpose("time", "step", "number", "latitude", "longitude").stack(
            sample=("time", "step", "number")
        ).transpose("sample", "latitude", "longitude")
        assert flattened.shape == (n_samples * n_output_days * n_members, self.H, self.W)
        flat_valid_time: xr.DataArray = valid_time.expand_dims(number=da.coords["number"]).transpose(
            "time", "step", "number"
        ).stack(sample=("time", "step", "number"))
        assert flat_valid_time.shape == (n_samples * n_output_days * n_members,)
        processed: xr.DataArray = self._preprocess(da=flattened, valid_time=flat_valid_time)
        output: xr.DataArray = processed.unstack("sample").transpose("number", "time", "step", "latitude", "longitude")
        output = output.assign_coords(valid_time=da.coords["valid_time"])
        assert output.shape == (n_members, n_samples, n_output_days, self.H, self.W)
        assert output.coords["valid_time"].dims == ("time", "step")
        return output

    def preprocess_groundtruths(self, da: xr.DataArray) -> xr.DataArray:
        required_dims: tuple[str, ...] = ("valid_time", "latitude", "longitude")
        assert da.dims == required_dims, f"Expected dims {required_dims}, got {da.dims}"
        da = ERA5Utilities.dropfeb29(da)
        n_valid_times: int = da.sizes["valid_time"]
        assert da.shape == (n_valid_times, self.H, self.W), (
            f"Unexpected groundtruth shape: expected {(n_valid_times, self.H, self.W)}, got {da.shape}"
        )
        valid_time: xr.DataArray = da.coords["valid_time"]
        assert valid_time.shape == (n_valid_times,)
        flattened: xr.DataArray = da.rename(valid_time="sample")
        processed: xr.DataArray = self._preprocess(da=flattened, valid_time=valid_time)
        return processed.rename(sample="valid_time")

    def _preprocess(self, da: xr.DataArray, valid_time: xr.DataArray) -> xr.DataArray:
        required_dims: tuple[str, ...] = ("sample", "latitude", "longitude")
        assert da.dims == required_dims, f"Expected dims {required_dims}, got {da.dims}"
        n_samples: int = da.sizes["sample"]
        assert da.shape == (n_samples, self.H, self.W), (
            f"Unexpected flattened shape: expected {(n_samples, self.H, self.W)}, got {da.shape}"
        )
        assert valid_time.shape == (n_samples,), (
            f"Unexpected valid_time shape: expected {(n_samples,)}, got {valid_time.shape}"
        )
        months: NDArray[np.int64] = valid_time.dt.month.values.astype(np.int64)
        years: NDArray[np.float32] = valid_time.dt.year.values.astype(np.float32)
        dayofyears: NDArray[np.int64] = valid_time.dt.dayofyear.values.astype(np.int64)
        day_index: NDArray[np.int64] = dayofyears - 1 - (months > 2).astype(np.int64)
        assert np.all((0 <= day_index) & (day_index < 365))

        trend: NDArray[np.float32] = self._compute_trend(years=years)
        climatological_mean: NDArray[np.float32] = self.climatological_mean[day_index]
        climatological_std: NDArray[np.float32] = self.climatological_std[day_index]
        assert trend.shape == climatological_mean.shape == climatological_std.shape == (n_samples, self.H, self.W)

        input_array: NDArray[np.float32] = da.values.astype(np.float32)
        assert input_array.shape == (n_samples, self.H, self.W)
        standardized_anomaly: NDArray[np.float32] = (input_array - trend - climatological_mean) / climatological_std
        return da.copy(data=standardized_anomaly)

    def _compute_trend(self, years: NDArray[np.float32]) -> NDArray[np.float32]:
        assert years.ndim in {1, 2}
        X: NDArray[np.float32] = np.stack(
            [
                np.ones(years.shape, dtype=np.float32),
                years.astype(np.float32, copy=False),
            ],
            axis=-1,
        )
        trend: NDArray[np.float32] = X.reshape(-1, 2) @ self.lr_weight
        trend = trend.reshape(*years.shape, self.H, self.W)
        return trend


class ECMWFGenerator:

    def __init__(
        self,
        groundtruth_path: str | Path,
        prediction_path: str | Path,
        metadata: MetaData,
        train_metadata: MetaData,
        from_year: int,
        to_year: int,
    ) -> None:
        assert metadata.dataset_name == "era5"
        assert metadata.tp == "test"
        assert len(metadata.output_vars) == 1

        self.metadata: MetaData = metadata
        self.output_name: str = metadata.output_vars[0]
        self.groundtruth_path: Path = Path(groundtruth_path)
        self.prediction_path: Path = Path(prediction_path)
        self.from_year: int = from_year
        self.to_year: int = to_year
        self.preprocessor: ECMWFPreprocessor = ECMWFPreprocessor(train_metadata=train_metadata)
        self.model_name: str = "ECMWFS2S"
        self.tropical_lats: tuple[float, float] = (-25.0, 25.0) # fixed
        self.mse = GeographicalMSE(
            landmask_path=metadata.landmask_path.as_posix(),
            tropical_lats=self.tropical_lats,
        )
        self.mae = GeographicalMAE(
            landmask_path=metadata.landmask_path.as_posix(),
            tropical_lats=self.tropical_lats,
        )
        self.n_members, self.n_samples, self.n_output_days, self.H, self.W = self.prediction_tensor.shape
        assert (self.H, self.W) == metadata.resolution

    @staticmethod
    def _get_date_index(year: int) -> list[np.datetime64]:
        start_date: dt.date = dt.date(year, 1, 1)
        end_date: dt.date = dt.date(year, 12, 31)
        values: list[np.datetime64] = []
        current: dt.date = start_date
        while current <= end_date:
            if not (current.month == 2 and current.day == 29):
                values.append(np.datetime64(current.isoformat()))
            current += dt.timedelta(days=1)
        assert len(values) == 365
        return values

    @cached_property
    def sample_timearray_lookup(self) -> dict[int, NDArray[np.datetime64]]:
        valid_time: xr.DataArray = self.prediction_array.coords["valid_time"]
        assert valid_time.dims == ("time", "step"), f"Unexpected valid_time dims: {valid_time.dims}"
        values: NDArray[np.datetime64] = valid_time.values
        assert values.shape == (self.n_samples, self.n_output_days), (
            "Unexpected valid_time lookup shape: "
            f"expected {(self.n_samples, self.n_output_days)}, got {values.shape}"
        )
        return dict(enumerate(values))

    @cached_property
    def prediction_years(self) -> list[int]:
        years: NDArray[np.int64] = (
            self.prediction_array.coords["valid_time"].values
            .astype("datetime64[Y]").astype(np.int64).reshape(-1) + 1970
        )
        return sorted(set(years.tolist()))

    @cached_property
    def era5_truth(self) -> dict[np.datetime64, torch.Tensor]:
        result: dict[np.datetime64, torch.Tensor] = {}
        for year in self.prediction_years:
            filename: str = f"{self.output_name}/{self.output_name}_{year}.nc"
            filepath: Path = self.groundtruth_path.joinpath(filename)
            assert filepath.exists()
            da: xr.DataArray = xr.open_dataarray(filepath, engine="netcdf4").load()
            da = ERA5Utilities.convert_to_cesm2_definition(var_name=self.output_name, da=da)
            ERA5Utilities.validate_complete_data(da=da, var_name=self.output_name, year=year)
            da = self.preprocessor.preprocess_groundtruths(da=da)
            tensor: torch.Tensor = torch.from_numpy(da.values.astype(np.float32))
            assert tensor.shape == (365, self.H, self.W)
            for day_idx, date_value in enumerate(self._get_date_index(year)):
                record: torch.Tensor = tensor[day_idx]
                assert record.shape == (self.H, self.W)
                result[date_value] = record
        return result

    @cached_property
    def prediction_array(self) -> xr.DataArray:
        reader: ECMWFReader = ECMWFReader(
            root=self.prediction_path.as_posix(),
            n_lead_days=self.metadata.n_lead_days,
            n_output_days=self.metadata.n_output_days,
            target_resolution=self.metadata.resolution,
            from_year=self.from_year,
            to_year=self.to_year,
        )
        predictions: xr.DataArray = reader.read()
        predictions = self.preprocessor.preprocess_predictions(da=predictions)
        expected_shape: tuple[int, int, int, int, int] = (
            predictions.sizes["number"],
            predictions.sizes["time"],
            self.metadata.n_output_days,
            self.metadata.resolution[0],
            self.metadata.resolution[1],
        )
        assert predictions.shape == expected_shape, (
            "Unexpected raw ECMWF-S2S prediction shape: "
            f"expected {expected_shape}, got {predictions.shape}"
        )
        return predictions

    @cached_property
    def prediction_tensor(self) -> torch.Tensor:
        tensor: torch.Tensor = torch.from_numpy(self.prediction_array.values.astype(np.float32))
        assert tensor.ndim == 5 # n_members, n_samples, n_output_days, H, W
        return tensor

    @staticmethod
    def _to_datetime(value: np.datetime64) -> dt.datetime:
        timestamp_us: int = int(value.astype("datetime64[us]").astype(np.int64))
        return dt.datetime(1970, 1, 1) + dt.timedelta(microseconds=timestamp_us)

    def get_sample_info(self, sample_id: int) -> SampleInfo:
        prediction_time: NDArray[np.datetime64] = self.sample_timearray_lookup[sample_id]
        assert prediction_time.shape == (self.n_output_days,)
        init_time: np.datetime64 = self.prediction_array.coords["time"].values[sample_id]
        init_dt: dt.datetime = self._to_datetime(init_time)
        out_start_dt: dt.datetime = self._to_datetime(prediction_time[0])
        out_end_dt: dt.datetime = self._to_datetime(prediction_time[-1])
        return SampleInfo(
            sim_id="ecmwf",
            in_startdate=init_dt.strftime("%Y/%m/%d"),
            in_enddate=init_dt.strftime("%Y/%m/%d"),
            out_startdate=out_start_dt.strftime("%Y/%m/%d"),
            out_enddate=out_end_dt.strftime("%Y/%m/%d"),
        )

    def get_prediction(self, sample_id: int) -> torch.Tensor:
        return self.prediction_tensor[:, sample_id].mean(dim=1)

    def get_groundtruth(self, sample_id: int) -> torch.Tensor:
        prediction_time: NDArray[np.datetime64] = self.sample_timearray_lookup[sample_id]
        assert prediction_time.shape == (self.n_output_days,)
        tensors: list[torch.Tensor] = []
        for value in prediction_time:
            day_key: np.datetime64 = value.astype("datetime64[D]")
            if day_key not in self.era5_truth:
                raise KeyError(f"Missing ERA5 groundtruth for valid_time={day_key}")
            tensors.append(self.era5_truth[day_key])
        groundtruth: torch.Tensor = torch.stack(tensors, dim=0)
        assert groundtruth.shape == (self.n_output_days, self.H, self.W)
        return groundtruth.mean(dim=0)

    def _make_filename_prefix(self, sampleinfo: SampleInfo) -> str:
        prefix: str = (
            f"{self.model_name}_{self.output_name}_{sampleinfo.sim_id}_"
            f"{sampleinfo.in_startdate}{sampleinfo.in_enddate}_"
            f"{sampleinfo.out_startdate}{sampleinfo.out_enddate}"
        )
        return prefix.replace("/", "")

    def generate(self, target_dir: str | Path) -> None:
        torchio: TorchDictIO = TorchDictIO(dirpath=Path(target_dir).as_posix())
        for sample_id in range(self.n_samples):
            sampleinfo: SampleInfo = self.get_sample_info(sample_id=sample_id)
            prediction: torch.Tensor = self.get_prediction(sample_id=sample_id)
            groundtruth: torch.Tensor = self.get_groundtruth(sample_id=sample_id)
            assert prediction.shape == (self.n_members, self.H, self.W)
            assert groundtruth.shape == (self.H, self.W)
            prefix: str = self._make_filename_prefix(sampleinfo=sampleinfo)

            for member_idx in range(self.n_members):
                prediction_frame: torch.Tensor = prediction[member_idx]
                error_frame: torch.Tensor = groundtruth - prediction_frame
                global_mse_, tropical_mse_, extratropical_mse_ = self.mse(
                    prediction=prediction_frame, groundtruth=groundtruth
                )
                global_mse: float = global_mse_.item()
                tropical_mse: float = tropical_mse_.item()
                extratropical_mse: float = extratropical_mse_.item()
                global_rmse: float = global_mse ** 0.5
                tropical_rmse: float = tropical_mse ** 0.5
                extratropical_rmse: float = extratropical_mse ** 0.5
                global_mae_, tropical_mae_, extratropical_mae_ = self.mae(
                    prediction=prediction_frame, groundtruth=groundtruth
                )
                global_mae: float = global_mae_.item()
                tropical_mae: float = tropical_mae_.item()
                extratropical_mae: float = extratropical_mae_.item()
                suffix: str = f"ens_{member_idx:04d}.pt"
                result_object: dict[str, Any] = {
                    "groundtruth": groundtruth,
                    "prediction": prediction_frame,
                    "error_map": error_frame,
                    "global_mse": global_mse,
                    "tropical_mse": tropical_mse,
                    "extratropical_mse": extratropical_mse,
                    "global_rmse": global_rmse,
                    "tropical_rmse": tropical_rmse,
                    "extratropical_rmse": extratropical_rmse,
                    "global_mae": global_mae,
                    "tropical_mae": tropical_mae,
                    "extratropical_mae": extratropical_mae,
                    "model_name": self.model_name,
                    "output_name": self.output_name,
                    "sim_id": sampleinfo.sim_id,
                    "in_startdate": sampleinfo.in_startdate,
                    "in_enddate": sampleinfo.in_enddate,
                    "out_startdate": sampleinfo.out_startdate,
                    "out_enddate": sampleinfo.out_enddate,
                    "tropical_lats": self.tropical_lats,
                    "ensemble_size": self.n_members,
                    "ensemble_member": member_idx + 1,
                    "ensemble_stat": "member",
                    "prefix": prefix,
                    "suffix": suffix,
                }
                torchio.save(obj=result_object, filename=f"{prefix}_{suffix}")

            ensemble_mean: torch.Tensor = prediction.mean(dim=0)
            ensemble_var: torch.Tensor = prediction.var(dim=0, unbiased=True)
            ensemble_std: torch.Tensor = prediction.std(dim=0, unbiased=True)
            ensemble_q001: torch.Tensor = prediction.quantile(q=0.01, dim=0)
            ensemble_q005: torch.Tensor = prediction.quantile(q=0.05, dim=0)
            ensemble_q010: torch.Tensor = prediction.quantile(q=0.10, dim=0)
            ensemble_q025: torch.Tensor = prediction.quantile(q=0.25, dim=0)
            ensemble_q050: torch.Tensor = prediction.quantile(q=0.50, dim=0)
            ensemble_q075: torch.Tensor = prediction.quantile(q=0.75, dim=0)
            ensemble_q090: torch.Tensor = prediction.quantile(q=0.90, dim=0)
            ensemble_q095: torch.Tensor = prediction.quantile(q=0.95, dim=0)
            ensemble_q099: torch.Tensor = prediction.quantile(q=0.99, dim=0)
            error_mean_frame: torch.Tensor = groundtruth - ensemble_mean
            error_q050_frame: torch.Tensor = groundtruth - ensemble_q050
            global_mse_, tropical_mse_, extratropical_mse_ = self.mse(
                prediction=ensemble_mean, groundtruth=groundtruth
            )
            global_mse: float = global_mse_.item()
            tropical_mse: float = tropical_mse_.item()
            extratropical_mse: float = extratropical_mse_.item()
            global_rmse: float = global_mse ** 0.5
            tropical_rmse: float = tropical_mse ** 0.5
            extratropical_rmse: float = extratropical_mse ** 0.5
            global_mae_, tropical_mae_, extratropical_mae_ = self.mae(
                prediction=ensemble_mean, groundtruth=groundtruth
            )
            global_mae: float = global_mae_.item()
            tropical_mae: float = tropical_mae_.item()
            extratropical_mae: float = extratropical_mae_.item()
            suffix = "ens_aggregate.pt"
            result_object = {
                "groundtruth": groundtruth,
                "ensemble_mean": ensemble_mean,
                "ensemble_std": ensemble_std,
                "ensemble_var": ensemble_var,
                "ensemble_q001": ensemble_q001,
                "ensemble_q005": ensemble_q005,
                "ensemble_q010": ensemble_q010,
                "ensemble_q025": ensemble_q025,
                "ensemble_q050": ensemble_q050,
                "ensemble_q075": ensemble_q075,
                "ensemble_q090": ensemble_q090,
                "ensemble_q095": ensemble_q095,
                "ensemble_q099": ensemble_q099,
                "error_mean_frame": error_mean_frame,
                "error_q050_frame": error_q050_frame,
                "global_mse": global_mse,
                "tropical_mse": tropical_mse,
                "extratropical_mse": extratropical_mse,
                "global_rmse": global_rmse,
                "tropical_rmse": tropical_rmse,
                "extratropical_rmse": extratropical_rmse,
                "global_mae": global_mae,
                "tropical_mae": tropical_mae,
                "extratropical_mae": extratropical_mae,
                "model_name": self.model_name,
                "output_name": self.output_name,
                "sim_id": sampleinfo.sim_id,
                "in_startdate": sampleinfo.in_startdate,
                "in_enddate": sampleinfo.in_enddate,
                "out_startdate": sampleinfo.out_startdate,
                "out_enddate": sampleinfo.out_enddate,
                "tropical_lats": self.tropical_lats,
                "ensemble_size": self.n_members,
                "ensemble_member": -1,
                "ensemble_stat": "aggregate",
                "prefix": prefix,
                "suffix": suffix,
            }
            torchio.save(obj=result_object, filename=f"{prefix}_{suffix}")



if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--fromyear", type=int, required=True)
    parser.add_argument("--toyear", type=int, required=True)
    args: Namespace = parser.parse_args()
    assert args.fromyear <= args.toyear, "fromyear must be <= toyear"

    dataset: str = "era5"
    model_config: ECMWFS2SConfig = ECMWFS2SConfig()
    ecmwf_metadata: MetaData = MetaData(dataset_name=dataset, tp="test")
    era5_metadata: MetaData = MetaData(dataset_name=dataset, tp="train")
    self = ECMWFGenerator(
        groundtruth_path=model_config.groundtruth_path,
        prediction_path=model_config.prediction_path,
        metadata=ecmwf_metadata,
        train_metadata=era5_metadata,
        from_year=args.fromyear,
        to_year=args.toyear,
    )
    self.generate(target_dir=model_config.target_path.joinpath(f"{dataset}/tensors/"))
