from typing import *
import yaml
import pathlib
from collections import defaultdict
from functools import cached_property

import numpy as np
import xarray as xr
from zipfile import ZipFile
import torch


class ZipExtractor:

    def __init__(self, zip_root: str, array_root: str) -> None:
        self.zip_root: pathlib.Path = pathlib.Path(zip_root)
        self.pressurelevels_zip_root: pathlib.Path = self.zip_root.joinpath("pressurelevels")
        self.singlelevel_zip_root: pathlib.Path = self.zip_root.joinpath("singlelevel")
        assert self.pressurelevels_zip_root.exists()
        assert self.singlelevel_zip_root.exists()
        self.array_root: pathlib.Path = pathlib.Path(array_root)
        self.array_root.mkdir(parents=True, exist_ok=True)

    def _extract(self, year: int, tp: Literal["pressure_levels", "single_level"]) -> xr.Dataset:
        if tp == "pressure_levels":
            filepaths: List[pathlib.Path] = sorted(self.pressurelevels_zip_root.glob(f"{year}*.zip"), key=lambda x: x.name)
            required_dims: List[str] = ["valid_time", "pressure_level", "latitude", "longitude"]
        else:
            filepaths: List[pathlib.Path] = sorted(self.singlelevel_zip_root.glob(f"{year}*.zip"), key=lambda x: x.name)
            required_dims: List[str] = ["valid_time", "latitude", "longitude"]

        records: Dict[str, List[xr.DataArray]] = defaultdict(list)
        for filepath in filepaths:
            print(f"Extracting: {filepath}")
            with ZipFile(filepath, mode="r") as zipfile:
                for ncfile in sorted(zipfile.filelist, key=lambda x: x.filename):
                    print(f"Loading: {ncfile}")
                    with zipfile.open(ncfile, mode="r") as file:
                        da: xr.DataArray = xr.load_dataarray(file)
                        records[da.name].append(da)

        records: Dict[str, xr.DataArray] = {
            var_name: xr.concat(records[var_name], dim="valid_time")
            for var_name in records.keys()
        }
        ds: xr.Dataset = xr.Dataset(records)
        assert set(required_dims) == set(ds.dims), f"set(required_dims): {set(required_dims)}, set(ds.dims): {set(ds.dims)}"
        return ds.transpose(*required_dims)

    def from_pressure_levels(self, year: int) -> xr.Dataset:
        ds: xr.Dataset = self._extract(year=year, tp="pressure_levels")
        flattened_vars: Dict[str, xr.DataArray] = {}
        for var_name in ds.data_vars.keys():
            da: xr.DataArray = ds[var_name]
            for level in da.pressure_level.values:
                new_name: str = f"{var_name}_{int(level)}"
                flattened_vars[new_name] = da.sel(pressure_level=level, drop=True)

        return xr.Dataset(flattened_vars)

    def from_single_level(self, year: int) -> xr.DataArray:
        return self._extract(year=year, tp="single_level")

    def to_netcdf(self, year: int) -> None:
        ds: xr.Dataset = xr.merge(
            objects=[self.from_pressure_levels(year=year), self.from_single_level(year=year)],
        )
        ds = ZipExtractor.__dropfeb29(ds)
        ds = ds.transpose("valid_time", "latitude", "longitude")
        ds = ds.sortby(variables=["valid_time"], ascending=True)
        for var_name in ds.data_vars.keys():
            path: pathlib.Path = self.array_root.joinpath(var_name)
            path.mkdir(parents=True, exist_ok=True)
            var_da: xr.DataArray = ds[var_name]
            var_da.to_netcdf(path.joinpath(f"{var_name}_{year}.nc"))
            print(f"Saved {var_name}_{year}.nc - Shape: {var_da.shape}")

    @staticmethod
    def __dropfeb29(ds: xr.Dataset) -> xr.Dataset:
        assert "valid_time" in ds.coords
        return ds.sel(valid_time=~((ds.valid_time.dt.month == 2) & (ds.valid_time.dt.day == 29)))


class DataReader:

    def __init__(self, resolution: Tuple[int, int] | None, device: str) -> None:
        self.resolution: Tuple[int, int] = resolution
        self.H, self.W = self.resolution
        self.device: torch.device = torch.device(device)
        with open("./config.yaml", mode="r") as file:
            data_config: Dict[str, Any] = yaml.safe_load(file)["era5"]
            self.array_directory: pathlib.Path = pathlib.Path(data_config["array_root"])
            self.zip_directory: pathlib.Path = pathlib.Path(data_config["zip_root"])

    # NOTE: 
    # era5.DataReader has slightly different interface with cesm2.DataReader 
    # due to the difference in raw data's structure. However, they both return 
    # consistent output (365, H, W) because we want both datasets share a common 
    # preprocessing pipeline. Difference in interfaces should only be taken care 
    # in `exportdata.py`
    def get_tensor(self, var_name: str, year: int) -> torch.Tensor:
        filepath: pathlib.Path = self.array_directory.joinpath(f"{var_name}/{var_name}_{year}.nc")
        if not filepath.exists():
            # zip file not extracted to netcdf (.nc) yet
            self.__extract(year)

        assert filepath.exists()
        # force to load full data to RAM (not just memory references)
        da: xr.DataArray = xr.open_dataarray(filepath, engine="netcdf4").load()
        # validate
        self.__validate_complete_data(da=da, var_name=var_name, year=year)

        tensor: torch.Tensor = torch.from_numpy(da.values).to(device=self.device)
        assert tensor.shape == (365, 721, 1440)
        if self.resolution:
            tensor = self.__resize(tensor)
            assert tensor.shape == (365, self.H, self.W)
        return tensor

    def __extract(self, year: int) -> None:
        zip_extractor: ZipExtractor = ZipExtractor(zip_root=str(self.zip_directory), array_root=str(self.array_directory))
        zip_extractor.to_netcdf(year=year)

    def __resize(self, input2d: torch.Tensor) -> torch.Tensor:
        assert input2d.ndim == 3    # (365, H, W)
        return torch.nn.functional.interpolate(
            input=input2d[:, None, :, :], size=self.resolution, mode="bilinear"
        ).squeeze(dim=1)
    
    def __validate_complete_data(self, da: xr.DataArray, var_name: str, year: int) -> None:
        # check var name
        assert da.name == var_name
        # expected (365, H, W)
        time: xr.DataArray = da.coords["valid_time"]
        years = np.unique(time.dt.year.values)
        assert len(years) == 1 and years[0] == year, f"Found years: {years}"
        months = np.unique(time.dt.month.values)
        assert len(months) == 12, f"Found months: {months}"
        days = np.unique(time.dt.floor("D").values.astype("datetime64[D]"))
        assert len(days) == 365, f"Found {len(days)} unique days"


class LandmaskReader:

    def __init__(self, resolution: Tuple[int, int], device: str) -> None:
        self.resolution: Tuple[int, int]= resolution
        self.H, self.W = self.resolution
        self.device: torch.device = torch.device(device)
        with open("./config.yaml", mode="r") as file:
            pathstring: str = yaml.safe_load(file)["era5"]["zip_root"]
            self.mask_directory: pathlib.Path = pathlib.Path(pathstring).parent.joinpath("landmask")
            self.filepath: pathlib.Path = next(self.mask_directory.glob("*.nc"))

    def __resize(self, input2d: torch.Tensor) -> torch.Tensor:
        """
        Era5-land data (which is used to extract landmask) have higher resolution than Era5-climate,
        this method approximates the Era5-climate's landmask from Era5-land.
        """
        assert input2d.ndim == 2
        input2d = torch.nn.functional.interpolate(
            input=input2d[None, None, :, :], size=self.resolution, mode="nearest"
        )
        return input2d.squeeze(0, 1)

    @cached_property
    def tensor(self) -> torch.Tensor:
        da: xr.DataArray = xr.load_dataarray(self.filepath, engine="netcdf4")
        value: torch.Tensor = torch.from_numpy(da.values).to(device=self.device).nan_to_num(0.0)
        landmask: torch.Tensor = (value != 0.).float()
        assert landmask.shape == (721, 1440)
        landmask = LandmaskReader.__resize(landmask)
        assert landmask.shape == (self.H, self.W)
        return landmask


class CoordinatesReader:

    def __init__(self, resolution: Tuple[int, int], device: str) -> None:
        self.resolution: Tuple[int, int] = resolution
        self.H, self.W = self.resolution
        self.device: torch.device = torch.device(device)

    @cached_property
    def tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        lat_tensor: torch.Tensor = torch.arange(start=-90., end=90.01, step=180 / (self.H - 1))
        lon_tensor: torch.Tensor = torch.arange(start=0., end=360., step=360 / self.W)
        assert lat_tensor.shape == (self.H,)
        assert lon_tensor.shape == (self.W,)
        return lat_tensor, lon_tensor

