from typing import *
import yaml
import pathlib
from collections import defaultdict
from functools import cached_property
import datetime as dt

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

    def save_xr_dataarray(self, year: int) -> None:
        da: xr.DataArray = xr.concat(
            objs=[self.from_pressure_levels(year=year), self.from_single_level(year=year)],
            dim="variable",
        )
        da = da.rename({"valid_time": "time"})
        da = da.transpose("time", "variable", "latitude", "longitude")
        da = da.sortby(variables=["time", "variable"])
        da.to_netcdf(self.array_root.joinpath(f"{year}.nc"))
        print(f"Saved year: {year} - Shape: {da.shape}")


class DataReader:

    def __init__(self, year: int, spatial_resolution: Tuple[int, int], device: str) -> None:
        self.year: int = year
        self.spatial_resolution: Tuple[int, int] = spatial_resolution
        self.device: torch.device = torch.device(device)
        with open("./config.yaml", mode="r") as file:
            self.array_directory: pathlib.Path = pathlib.Path(yaml.safe_load(file)["era5"]["array_root"])

        self.filepath: pathlib.Path = self.array_directory.joinpath(f"{self.year}.nc")
        assert self.filepath.exists()

    def __resize(self, input2d: torch.Tensor) -> torch.Tensor:
        """
        For pre-trained models
        """
        assert input2d.ndim == 4    # (n_days, n_variables, H, W)
        return torch.nn.functional.interpolate(input=input2d, size=(self.spatial_resolution))

    @property   # Only access once -> No cache
    def tensor(self) -> torch.Tensor:
        da: xr.DataArray = xr.open_dataarray(self.filepath, engine="netcdf4")
        tensor: torch.Tensor = torch.from_numpy(da.values).to(device=self.device)
        # valid if 
        if self.year == 2025:
            n_year_days: int = 181
        elif self.year % 4 == 0:
            n_year_days: int = 366
        else:
            n_year_days: int = 365
        assert tensor.shape == (n_year_days, 16, 721, 1440)
        tensor = self.__resize(tensor)
        assert tensor.shape == (n_year_days, 16, self.spatial_resolution[0], self.spatial_resolution[1])
        return self.__resize(tensor)


class LandmaskReader:

    def __init__(self, spatial_resolution: Tuple[int, int], device: str) -> None:
        self.spatial_resolution: Tuple[int, int] = spatial_resolution
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
            input=input2d[None, None, :, :], size=self.spatial_resolution, mode="nearest"
        )
        input2d = input2d.squeeze(0, 1)
        assert input2d.shape == self.spatial_resolution
        return input2d

    @cached_property
    def tensors(self) -> torch.Tensor:
        da: xr.DataArray = xr.load_dataarray(self.filepath, engine="netcdf4")
        value: torch.Tensor = torch.from_numpy(da.values).to(device=self.device).nan_to_num(0.0)
        landmask: torch.Tensor = (value != 0.)
        landmask = LandmaskReader.__resize(landmask)
        assert landmask.shape == self.spatial_resolution
        return landmask


class CoordinatesReader:

    def __init__(self, spatial_resolution: Tuple[int, int], device: str) -> None:
        self.spatial_resolution: Tuple[int, int] = spatial_resolution
        self.device: torch.device = torch.device(device)

    @cached_property
    def tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        lat, lon = self.spatial_resolution
        lat_tensor: torch.Tensor = torch.arange(start=-90., end=90.01, step=180 / lat)
        lon_tensor: torch.Tensor = torch.arange(start=0., end=360., step=360 / lon)
        assert lat_tensor.shape == (lat,) 
        assert lon_tensor.shape == (lon,)
        return lat_tensor, lon_tensor


if __name__ == "__main__":
    year: int = 2022
    self = ZipExtractor(zip_root="/scratch/zgp2ps/era5/raw", array_root="/scratch/zgp2ps/era5/arrays")
    self.save_xr_dataarray(year=year)

    self = DataReader(year=year, device="cpu")
    a = self.tensor

