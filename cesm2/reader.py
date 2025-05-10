import datetime as dt
import pathlib
from functools import cached_property

import yaml
import torch
import xarray as xr


class DataReader:

    def __init__(self, var_name: str, sim_id: str, year: int, device: str = "cpu") -> None:
        self.var_name: str = var_name
        self.sim_id: str = sim_id
        self.year: int = year
        self.device: str = device
        with open("./config.yaml", mode="r") as file:
            self.root_directory: pathlib.Path = pathlib.Path(yaml.safe_load(file)["dataset"]["root"])

    @cached_property
    def filepath(self) -> pathlib.Path:
        for filepath in pathlib.Path(self.root_directory, self.var_name).glob("*.nc"):
            start_year, end_year = (int(part[:4]) for part in filepath.name.split(".")[-2].split("-")[:2])
            if start_year <= self.year <= end_year:
                return filepath
        raise FileNotFoundError(f"No file found for year {self.year} from year {start_year} to {end_year}")

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
        tensor: torch.Tensor = torch.from_numpy(self.ds.to_array().values).squeeze().to(device=self.device)
        assert tensor.shape == (365, 192, 288)
        return tensor
    