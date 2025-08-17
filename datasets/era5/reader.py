from typing import *
import yaml
import pathlib
from collections import defaultdict
from functools import cached_property
import datetime as dt

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

    def _extract(self, year: int, tp: Literal["pressure_levels", "single_level"]) -> xr.DataArray:
        if tp == "pressure_levels":
            filepaths: List[pathlib.Path] = sorted(self.pressurelevels_zip_root.glob(f"{year}*.zip"), key=lambda x: x.name)
            required_dims: List[str] = ["valid_time", "pressure_level", "variable", "latitude", "longitude"]
        else:
            filepaths: List[pathlib.Path] = sorted(self.singlelevel_zip_root.glob(f"{year}*.zip"), key=lambda x: x.name)
            required_dims: List[str] = ["valid_time", "variable", "latitude", "longitude"]

        records: Dict[str, List[xr.DataArray]] = defaultdict(list)
        for filepath in filepaths:
            print(f"Extracting: {filepath}")
            with ZipFile(filepath, mode="r") as zipfile:
                for ncfile in sorted(zipfile.filelist, key=lambda x: x.filename):
                    print(f"Loading: {ncfile}")
                    with zipfile.open(ncfile, mode="r") as file:
                        da: xr.DataArray = xr.load_dataarray(file)
                        records[da.name].append(da)

        var_names: List[str] = sorted(list(records.keys()))
        arrays: List[xr.DataArray] = [xr.concat(records[key], dim="valid_time") for key in var_names]
        da: xr.DataArray = xr.concat(arrays, dim="variable").assign_coords(variable=var_names)
        assert set(required_dims) == set(da.dims), f"set(required_dims): {set(required_dims)}, set(da.dims): {set(da.dims)}"
        return da.transpose(*required_dims)

    def from_pressure_levels(self, year: int) -> xr.DataArray:
        da: xr.DataArray = self._extract(year=year, tp="pressure_levels")
        # Flatten, and avoid "ValueError: cannot create a new dimension with the same name as an existing dimension"
        da = da.rename({"variable": "variable_temp"})
        da = da.stack(variable=["pressure_level", "variable_temp"])
        coords: List[str] = [
            f"{int(pl)}_{var}" for pl, var in da.coords["variable"].to_index().values
        ]
        da = da.drop_vars(["pressure_level", "variable_temp"])
        da = da.assign_coords(variable=coords)
        return da

    def from_single_level(self, year: int) -> xr.DataArray:
        return self._extract(year=year, tp="single_level")

    def to_netcdf(self, year: int) -> None:
        da: xr.DataArray = xr.concat(
            objs=[self.from_pressure_levels(year=year), self.from_single_level(year=year)],
            dim="variable",
        )
        da = ZipExtractor.__dropfeb29(da)
        da = da.rename({"valid_time": "time"})
        da = da.transpose("time", "variable", "latitude", "longitude")
        da = da.sortby(variables=["variable", "time"], ascending=True)
        da.to_netcdf(self.array_root.joinpath(f"{year}.nc"))
        print(f"Saved {year}.nc - Shape: {da.shape}")

    @staticmethod
    def __dropfeb29(da: xr.DataArray) -> xr.Dataset:
        assert "valid_time" in da.coords
        return da.sel(valid_time=~((da.valid_time.dt.month == 2) & (da.valid_time.dt.day == 29)))


class DataReader:

    def __init__(self, year: int, resolution: Tuple[int, int] | None, device: str) -> None:
        self.year: int = year
        self.resolution: Tuple[int, int] = resolution
        self.H, self.W = self.resolution
        self.device: torch.device = torch.device(device)
        with open("./config.yaml", mode="r") as file:
            data_config: Dict[str, Any] = yaml.safe_load(file)["era5"]
            self.array_directory: pathlib.Path = pathlib.Path(data_config["array_root"])
            self.zip_directory: pathlib.Path = pathlib.Path(data_config["zip_root"])

        self.filepath: pathlib.Path = self.array_directory.joinpath(f"{self.year}.nc")

    @cached_property
    def xrarray(self) -> xr.DataArray:
        if not self.filepath.exists():
            # zip file not extracted to netcdf (.nc) yet
            self.__extract()

        assert self.filepath.exists()
        # force to load full data to RAM (not just memory references)
        da: xr.DataArray = xr.open_dataarray(self.filepath, engine="netcdf4").load()
        # validate
        self.__validate_complete_data(da)
        return da

    # NOTE: 
    # era5.DataReader has slightly different interface with cesm2.DataReader 
    # due to the difference in raw data's structure. However, they both return 
    # consistent output (365, H, W) because we want both datasets share a common 
    # preprocessing pipeline. Difference in interfaces should only be taken care 
    # in `exportdata.py`
    def get_tensor(self, var_name: str) -> torch.Tensor:
        assert var_name in self.xrarray.coords["variable"]
        tensor: torch.Tensor = torch.from_numpy(self.xrarray.sel(variable=var_name).values).to(device=self.device)
        assert tensor.shape == (365, 721, 1440)
        if self.resolution:
            tensor = self.__resize(tensor)
            assert tensor.shape == (365, self.H, self.W)
        return tensor

    def __extract(self) -> None:
        zip_extractor: ZipExtractor = ZipExtractor(zip_root=self.zip_directory, array_root=self.array_directory)
        zip_extractor.to_netcdf(year=self.year)

    def __resize(self, input2d: torch.Tensor) -> torch.Tensor:
        assert input2d.ndim == 3    # (365, H, W)
        return torch.nn.functional.interpolate(
            input=input2d[:, None, :, :], size=self.resolution, mode="bilinear"
        ).squeeze(dim=1)
    
    def __validate_complete_data(self, da: xr.DataArray) -> None:
        time: xr.DataArray = da.coords["time"]  # or "valid_time" if that's the name
        years = np.unique(time.dt.year.values)
        assert len(years) == 1 and years[0] == self.year, f"Found years: {years}"
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


if __name__ == "__main__":
    year: int = 2024
    # self = ZipExtractor(zip_root="/scratch/zgp2ps/era5/raw", array_root="/scratch/zgp2ps/era5/arrays")
    # self.to_netcdf(year=year)

    self = DataReader(year=year, resolution=(192, 288), device="cpu")
    a = self.get_tensor(var_name="500_w")

