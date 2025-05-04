import datetime as dt
import pathlib
from functools import cached_property

import torch
import xarray as xr


class DataReader:

    """
    Light-weight, memory-efficient data reader, not prioritizing speed -> no cache
    """

    root_directory: pathlib.Path = pathlib.Path("./data")

    def __init__(self, var_name: str, sim_id: str, year: int) -> None:
        self.var_name: str = var_name
        self.sim_id: str = sim_id
        self.year: int = year

    @cached_property
    def filepath(self) -> pathlib.Path:
        for filepath in pathlib.Path(DataReader.root_directory, self.var_name).glob("*.nc"):
            start_year, end_year = (int(part[:4]) for part in filepath.name.split(".")[-2].split("-")[:2])
            if start_year <= self.year <= end_year:
                return filepath
        raise FileNotFoundError(f"No file found for year {self.year}")

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