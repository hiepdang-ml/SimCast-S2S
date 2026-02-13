import datetime as dt
import pathlib
from functools import cached_property

import yaml
import torch
import xarray as xr


class DataReader:

    """
    Read from .nc to tensor
    """

    def __init__(self, var_name: str, sim_id: str, year: int) -> None:
        self.var_name: str = var_name
        self.sim_id: str = sim_id
        self.year: int = year
        with open("./config.yaml", mode="r") as file:
            self.root_directory: pathlib.Path = pathlib.Path(yaml.safe_load(file)["cesm2"]["root"])

    @cached_property
    def filepath(self) -> pathlib.Path:
        for filepath in pathlib.Path(self.root_directory, self.var_name).glob(f"*{self.sim_id}*.nc"):
            start_year, end_year = (int(part[:4]) for part in filepath.name.split(".")[-2].split("-")[:6])
            if start_year <= self.year <= end_year:
                return filepath
        raise FileNotFoundError(f"No file found for sim_id {self.sim_id} year {self.year}")

    @staticmethod
    def __compute_week(t: dt.date) -> int:
        return t.isocalendar().week

    @property   # Only access once -> No cache
    def ds(self) -> xr.Dataset:
        ds: xr.Dataset = xr.open_dataset(self.filepath, engine="netcdf4")[[self.var_name]]
        ds = ds.assign_coords(
            year=("time", ds.time.dt.year.data),
            week=("time", [DataReader.__compute_week(t=dt.date(year=t.year, month=t.month, day=t.day)) for t in ds.time.values])
        )
        ds = ds.sel(time=ds.time.dt.year == self.year)
        return ds.sortby(["time"]).transpose("time", "lat", "lon")

    @property   # Only access once -> No cache
    def tensor(self) -> torch.Tensor:
        tensor: torch.Tensor = torch.from_numpy(self.ds.to_array().values).squeeze()
        assert tensor.shape == (365, 192, 288)
        return tensor


class LandMaskReader:

    """
    Read from .nc to tensor
    """

    def __init__(self) -> None:
        with open("./config.yaml", mode="r") as file:
            pathstring: str = yaml.safe_load(file)["cesm2"]["root"]
            self.mask_directory: pathlib.Path = pathlib.Path(pathstring).parent.joinpath("landmask")
            self.filepath: pathlib.Path = next(self.mask_directory.glob("*.nc"))

    @cached_property
    def tensor(self) -> torch.Tensor:
        xrarray: xr.DataArray = xr.open_dataset(self.filepath, engine="netcdf4")["landmask"]
        tensor: torch.Tensor = torch.from_numpy(xrarray.values).squeeze()
        tensor = torch.nan_to_num(tensor, nan=0.0)
        assert tensor.shape == (192, 288)
        return tensor


class CoordinatesReader:

    """
    Read from .nc to tensor
    """

    def __init__(self) -> None:
        with open("./config.yaml", mode="r") as file:
            pathstring: str = yaml.safe_load(file)["cesm2"]["root"]
            self.mask_directory: pathlib.Path = pathlib.Path(pathstring).parent.joinpath("landmask")
            self.filepath: pathlib.Path = next(self.mask_directory.glob("*.nc"))

    @cached_property
    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        ds: xr.Dataset = xr.open_dataset(self.filepath, engine="netcdf4")
        lat_tensor: torch.Tensor = torch.from_numpy(ds["lat"].values).squeeze()
        lon_tensor: torch.Tensor = torch.from_numpy(ds["lon"].values).squeeze()
        assert lat_tensor.shape == (192,)
        assert lon_tensor.shape == (288,)
        return lat_tensor, lon_tensor
