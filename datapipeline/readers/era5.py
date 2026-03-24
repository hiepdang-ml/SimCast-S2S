from typing import Literal, Any, cast
from numpy.typing import NDArray
import yaml
from pathlib import Path
from collections import defaultdict
from functools import cached_property

import numpy as np
import xarray as xr
import torch


class Merger:

    def __init__(self, source_root: str, target_root: str) -> None:
        self.source_root: Path = Path(source_root)
        self.target_root: Path = Path(target_root)
        self.pressurelevels_root: Path = self.source_root.joinpath("pressurelevels")
        self.singlelevel_root: Path = self.source_root.joinpath("singlelevel")

    def _align_variables(self, year: int, tp: Literal["pressure_levels", "single_level"]) -> xr.Dataset:
        filedirs: list[Path]
        required_dims: list[str]
        if tp == "pressure_levels":
            filedirs = sorted(self.pressurelevels_root.glob(f"{year}*"), key=lambda x: x.name)
            required_dims = ["valid_time", "pressure_level", "latitude", "longitude"]
        else:
            filedirs = sorted(self.singlelevel_root.glob(f"{year}*"), key=lambda x: x.name)
            required_dims = ["valid_time", "latitude", "longitude"]

        records: dict[str, list[xr.DataArray]] = defaultdict(list)
        for filedir in filedirs:
            for ncfile in sorted(filedir.glob("*.nc"), key=lambda x: x.name):
                print(f"Loading: {ncfile}")
                da: xr.DataArray = xr.load_dataarray(ncfile, engine="netcdf4")
                var_name: str = cast(str, da.name)
                var_name = var_name.replace("_", "") # remove all "_" to ease later parsing
                records[var_name].append(da)

        concat_records: dict[str, xr.DataArray] = {
            var_name: xr.concat(records[var_name], dim="valid_time")
            for var_name in records.keys()
        }
        del records     # release memory
        ds: xr.Dataset = xr.Dataset(concat_records)
        del concat_records     # release memory
        assert set(required_dims) == set(ds.dims), f"required_dims: {set(required_dims)}, ds.dims: {set(ds.dims)}"
        return ds.transpose(*required_dims)

    def from_pressure_levels(self, year: int) -> xr.Dataset:
        ds: xr.Dataset = self._align_variables(year=year, tp="pressure_levels")
        flattened_vars: dict[str, xr.DataArray] = {}
        for var_name in ds.data_vars.keys():
            da: xr.DataArray = ds[var_name]
            for level in da.pressure_level.values:
                new_name: str = f"{var_name}{int(level)}"
                flattened_vars[new_name] = da.sel(pressure_level=level, drop=True)

        return xr.Dataset(flattened_vars)

    def from_single_level(self, year: int) -> xr.Dataset:
        return self._align_variables(year=year, tp="single_level")

    def to_netcdf(self, year: int, target_resolution: tuple[int, int]) -> None:
        ds: xr.Dataset = xr.merge(
            objects=[self.from_pressure_levels(year=year), self.from_single_level(year=year)],
        )
        # Preprocess ERA5 into CESM2 format
        ds = ERA5Utilities.dropfeb29(ds)
        ds = ERA5Utilities.fliplatitude(ds)
        # Resize
        newlat: NDArray[np.float64] = np.linspace(
            ds.latitude.min().item(), ds.latitude.max().item(), target_resolution[0]
        )
        newlon: NDArray[np.float64] = np.linspace(
            ds.longitude.min().item(), ds.longitude.max().item(), target_resolution[1]
        )
        ds = ds.interp(latitude=newlat, longitude=newlon, method="linear")
        # Transpose and sort
        ds = ds.transpose("valid_time", "latitude", "longitude")
        ds = ds.sortby(variables=["valid_time"], ascending=True)
        # Write
        for var_name in ds.data_vars.keys():
            target_path: Path = self.target_root.joinpath(cast(str, var_name))
            target_path.mkdir(parents=True, exist_ok=True)
            var_da: xr.DataArray = ds[var_name]
            var_da.to_netcdf(target_path.joinpath(f"{var_name}_{year}.nc"))
            print(f"Saved {var_name}_{year}.nc - Shape: {var_da.shape}")


class ERA5Utilities:

    @staticmethod
    def dropfeb29(data: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
        """
        ERA5 accounts for leaf years, CESM2 standardizes years into 365-day periods
        """
        assert "valid_time" in data.coords
        return data.sel(valid_time=~((data.valid_time.dt.month == 2) & (data.valid_time.dt.day == 29)))

    @staticmethod
    def fliplatitude(data: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
        """
        ERA5 stores latitude in descending order (90 -> -90), CESM2 stores it in ascending order
        """
        assert "latitude" in data.coords
        return data.sortby("latitude", ascending=True)

    @staticmethod
    def convert_to_cesm2_definition(var_name: str, da: xr.DataArray) -> xr.DataArray:
        if var_name == "avgtnlwrf":
            print(f"Convert {var_name} (flip sign)")
            return -da    # ECMWF convention assigns possitive downward
        if var_name == "tp":
            print(f"Convert {var_name} (divided by 3600)")
            return da / 3600  # precisely, da = da * 24 / 86400
        if var_name in {"z200", "z500", "z850"}:
            print(f"Convert {var_name} (divided by 9.80665)")
            return da / 9.80665   # geopotential -> geopotential height

        print(f"No conversion {var_name}")
        return da

    @staticmethod
    def validate_complete_data(da: xr.DataArray, var_name: str, year: int) -> None:
        # check var name
        assert da.name == var_name
        # expected (365, H, W)
        time: xr.DataArray = da.coords["valid_time"]
        found_years: set[int] = set(y.item() for y in time.dt.year.values)
        assert found_years == {year}, f"Found years: {found_years}"
        months: set[int] = set(m.item() for m in time.dt.month.values)
        assert len(months) == 12, f"Found months: {months}"
        days = set(d.item() for d in time.dt.floor("D").values.astype("datetime64[D]"))
        assert len(days) == 365, f"Found {len(days)} unique days"
        return


class DataReader:

    def __init__(self, target_resolution: tuple[int, int]) -> None:
        with open("./config.yaml", mode="r") as file:
            data_config: dict[str, Any] = yaml.safe_load(file)["era5"]
            self.raw_root: Path = Path(data_config["raw_root"])
            self.array_root: Path = Path(data_config["array_root"])
            self.target_resolution: tuple[int, int] = target_resolution

    # NOTE:
    # era5.DataReader has slightly different interface with cesm2.DataReader
    # due to the difference in raw data's structure. However, they both return
    # consistent output (365, H, W) because we want both datasets share a common
    # preprocessing pipeline. Difference in interfaces should only be taken care
    # in `exportdata.py`
    def get_tensor(self, var_name: str, year: int) -> torch.Tensor:
        filepath: Path = self.array_root.joinpath(f"{var_name}/{var_name}_{year}.nc")
        if not filepath.exists():
            # NOTE: miss one variable => miss all variables
            self.__merge(year)

        assert filepath.exists()
        # Load merged files from disk
        # NOTE: Force to load full data to RAM (not just memory references)
        da: xr.DataArray = xr.open_dataarray(filepath, engine="netcdf4").load()
        # Convert to ERA5's definition
        da = ERA5Utilities.convert_to_cesm2_definition(var_name=var_name, da=da)
        # Validate
        ERA5Utilities.validate_complete_data(da=da, var_name=var_name, year=year)
        tensor: torch.Tensor = torch.from_numpy(da.values.astype("float32"))
        return tensor

    def __merge(self, year: int) -> None:
        merger: Merger = Merger(
            source_root=self.raw_root.as_posix(), target_root=self.array_root.as_posix(),
        )
        merger.to_netcdf(year=year, target_resolution=self.target_resolution)


class LandmaskReader:

    def __init__(self, resolution: tuple[int, int]) -> None:
        self.resolution: tuple[int, int]= resolution
        self.H, self.W = self.resolution
        with open("./config.yaml", mode="r") as file:
            self.mask_directory: Path = Path(yaml.safe_load(file)["era5"]["landmask_root"])
            self.filepath: Path = next(self.mask_directory.glob("*.nc"))

    def __resize(self, da: xr.DataArray) -> xr.DataArray:
        # Resize
        newlat: NDArray[np.float64] = np.linspace(
            da.latitude.min().item(), da.latitude.max().item(), self.resolution[0]
        )
        newlon: NDArray[np.float64] = np.linspace(
            da.longitude.min().item(), da.longitude.max().item(), self.resolution[1]
        )
        da = da.interp(latitude=newlat, longitude=newlon, method="nearest")
        return da

    @cached_property
    def tensor(self) -> torch.Tensor:
        da: xr.DataArray = xr.load_dataarray(self.filepath, engine="netcdf4").isel(valid_time=0)
        da = ERA5Utilities.fliplatitude(da)
        da = self.__resize(da)
        value: torch.Tensor = torch.from_numpy(da.values).nan_to_num(0.0)
        landmask: torch.Tensor = (value != 0.).float()
        assert landmask.shape == (self.H, self.W)
        return landmask


class CoordinatesReader:

    def __init__(self, resolution: tuple[int, int]) -> None:
        self.resolution: tuple[int, int] = resolution
        self.H, self.W = self.resolution
        with open("./config.yaml", mode="r") as file:
            self.mask_directory: Path = Path(yaml.safe_load(file)["era5"]["landmask_root"])
            self.filepath: Path = next(self.mask_directory.glob("*.nc"))

    def __resize(self, input1d: torch.Tensor, HorW: Literal["H","W"]) -> torch.Tensor:
        assert input1d.ndim == 1
        min: float = input1d.min().item()
        max: float = input1d.max().item()
        return torch.linspace(start=min, end=max, steps=self.H if HorW == "H" else self.W)

    @cached_property
    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        ds: xr.Dataset = xr.open_dataset(self.filepath, engine="netcdf4")
        lat_tensor: torch.Tensor = torch.from_numpy(ds["latitude"].values).squeeze()
        lon_tensor: torch.Tensor = torch.from_numpy(ds["longitude"].values).squeeze()
        lat_tensor = self.__resize(input1d=lat_tensor, HorW="H")
        lon_tensor = self.__resize(input1d=lon_tensor, HorW="W")
        assert lat_tensor.shape == (self.H,)
        assert lon_tensor.shape == (self.W,)
        return lat_tensor, lon_tensor
