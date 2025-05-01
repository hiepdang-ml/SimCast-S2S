from __future__ import annotations

import os
from typing import List, Set, Tuple, Dict, TypedDict, NewType, Callable, Optional, Literal
import pathlib
from functools import cached_property, cache
from collections import defaultdict, namedtuple
import datetime as dt
import json

import xarray as xr
import torch
from torch.utils.data import Dataset


class DataContainer:

    """
    Universal data container in place of xarray
    """

    def __init__(self, var_names: List[str], sim_ids: List[str], years: List[int]) -> None:
        self.var_names: str = var_names
        self.sim_ids: List[str] = sim_ids
        self.years: List[int] = years
        
        self.__container: Dict[str, Dict[str, Dict[int, Optional[torch.Tensor]]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for var_name in self.var_names:
            for sim_id in self.sim_ids:
                for year in self.years:
                    self.set(var_name, sim_id, year, None)

    def get(self, var_name: str, sim_id: str, year: int | None) -> torch.Tensor| Dict[int, torch.Tensor] | None:
        if year is None:
            return self.__container[var_name][sim_id]
        return self.__container[var_name][sim_id][year]
    
    def set(self, var_name: str, sim_id: str, year: int | None, value: torch.Tensor | Dict[int, torch.Tensor] | None) -> None:
        if year is None:
            assert isinstance(value, dict)
            self.__container[var_name][sim_id] = value
        else:
            assert isinstance(value, (torch.Tensor, type(None)))
            self.__container[var_name][sim_id][year] = value

    @property
    def is_completed(self) -> bool:
        return all(
            self.get(var_name=var_name, sim_id=sim_id, year=year) is not None
            for var_name in self.var_names
            for sim_id in self.sim_ids
            for year in self.years
        )
                
    def to_json(self) -> None:
        """
        For debugging only
        """
        with open("./datacontainer.json", mode="w") as file:
            json.dump(
                obj={
                    var_name: {
                        sim_id: {
                            str(year): tensor.tolist() if tensor is not None else None
                            for year, tensor in years.items()
                        }
                        for sim_id, years in sim_ids.items()
                    }
                    for var_name, sim_ids in self.__container.items()
                }, 
                fp=file
            )

    def __add__(self, other) -> DataContainer:
        assert isinstance(other, DataContainer)
        assert self.is_completed
        assert other.is_completed

        result: DataContainer = DataContainer(self.var_names, self.sim_ids, self.years)
        for var_name in self.var_names:
            for sim_id in self.sim_ids:
                for year in self.years:
                    result.set(
                        var_name, sim_id, year,
                        value=(
                            self.get(var_name=var_name, sim_id=sim_id, year=year)
                            + other.get(var_name=var_name, sim_id=sim_id, year=year)
                        )
                    )
        return result

    def __sub__(self, other) -> DataContainer:
        return self + (-other)

    def __neg__(self) -> DataContainer:
        assert self.is_completed

        result: DataContainer = DataContainer(self.var_names, self.sim_ids, self.years)
        for var_name in self.var_names:
            for sim_id in self.sim_ids:
                for year in self.years:
                    result.set(
                        var_name, sim_id, year,
                        value= -self.get(var_name=var_name, sim_id=sim_id, year=year)
                    )
        return result

    def yearly_agg(self, reduce_func: Literal["sum", "mean", "std"]) -> DataContainer:
        assert self.is_completed

        func: Callable[[torch.Tensor], torch.Tensor] = {
            "sum": lambda x: x.sum(dim=1, keepdim=True),
            "mean": lambda x: x.mean(dim=1, keepdim=True),
            "std": lambda x: x.std(dim=1, keepdim=True)
        }.get(reduce_func)

        result: DataContainer = DataContainer(self.var_names, self.sim_ids, self.years)
        for var_name in self.var_names:
            for sim_id in self.sim_ids:
                for year in self.years:
                    result.set(
                        var_name, sim_id, year,
                        value=func(self.get(var_name=var_name, sim_id=sim_id, year=year))
                    )
        return result
    

class UnitaryDataReader:

    """
    Light-weight, memory-efficient data reader, not prioritizing speed -> no cache
    """

    def __init__(self, var_name: str, sim_id: str, year: int) -> None:
        self.root: str = os.getenv("DATAROOT")
        self.var_name: str = var_name
        self.sim_id: str = sim_id
        self.year: int = year

    @cached_property
    def filepath(self) -> pathlib.Path:
        for filepath in pathlib.Path(self.root, self.var_name).glob("*.nc"):
            start_year, end_year = (int(part[:4]) for part in filepath.name.split(".")[-2].split("-")[:2])
            if start_year <= self.year <= end_year:
                return filepath
        raise FileNotFoundError(f"No file found for year {self.year}")

    @staticmethod
    def __compute_week(t: dt.date) -> int:
        return t.isocalendar().week

    #nocache
    @property
    def ds(self) -> xr.Dataset:
        ds: xr.Dataset = xr.open_dataset(self.filepath, engine="netcdf4")[[self.var_name]]
        ds = ds.assign_coords(
            year=("time", [t.year for t in ds.time.values]),
            week=("time", [UnitaryDataReader.__compute_week(t=dt.date(year=t.year, month=t.month, day=t.day)) for t in ds.time.values])
        )
        ds = ds.where(ds.coords["year"]==self.year, drop=True)
        ds = ds.expand_dims(sim=[self.sim_id])
        return ds.sortby(["time"]).transpose("sim", "time", "lat", "lon")

    #nocache
    @property
    def tensor(self) -> torch.Tensor:
        tensor: torch.Tensor = torch.from_numpy(self.ds.to_array().transpose(..., "variable").values)
        assert tensor.shape == (1, 365, 192, 288, 1)    # (S, T, H, W, E)
        return tensor
    

class Detrender:

    def __init__(self) -> None:
        self.var_names: List[str] = ["FLUT", "TS", "PRECT"]
        self.root: str = os.getenv("DATAROOT")
        self.years, self.sim_ids = self.get_sims_and_years()
        
    def get_sims_and_years(self) -> Tuple[List[int], List[str]]:
        year_ranges: Dict[str, Set[str]] = defaultdict(set)
        sim_ranges: Dict[str, Set[str]] = defaultdict(set)
        for var_name in self.var_names:
            for filepath in pathlib.Path(self.root, var_name).glob("*.nc"):
                assert var_name in filepath.name
                year_ranges[var_name].add(filepath.name.split(".")[-2])
                sim_ranges[var_name].add(filepath.name.split("-")[1][:8])

        ref_years: Set[str] = year_ranges[self.var_names[0]]
        ref_sims: Set[str] = sim_ranges[self.var_names[0]]
        assert all(year_ranges[v] == ref_years for v in self.var_names)
        assert all(sim_ranges[v] == ref_sims for v in self.var_names)

        min_year: int = min(int(y.split("-")[0][:4]) for y in ref_years)
        max_year: int = max(int(y.split("-")[1][:4]) for y in ref_years)
        return list(range(min_year, max_year + 1)), sorted(list(ref_sims))
    
    @staticmethod
    @cache
    def get_data_reader(var_name: str, sim_id: str, year: int) -> UnitaryDataReader:
        return UnitaryDataReader(var_name, sim_id, year)

    def linear_regression(self, data: DataContainer) -> DataContainer:
        n_years: int = len(self.years)
        year_tensor: torch.Tensor = torch.tensor(data=self.years, dtype=torch.float)
        assert year_tensor.shape == (n_years,)
        X: torch.Tensor = torch.stack([torch.ones(n_years), year_tensor], dim=1)
        assert X.shape == (n_years, 2)
        mean_container: DataContainer = data.yearly_agg(reduce_func="mean")
        results: DataContainer = DataContainer(self.var_names, self.sim_ids, self.years)

        for var_name in self.var_names:
            for sim_id in self.sim_ids:
                mean_tensors: Dict[int: torch.Tensor] = mean_container.get(var_name=var_name, sim_id=sim_id, year=None)
                mean_tensor: torch.Tensor = torch.cat([tensor for tensor in mean_tensors.values()], dim=1) # along T
                assert mean_tensor.shape == (1, n_years, 192, 288, 1)
                y: torch.Tensor = mean_tensor.reshape(n_years, 192 * 288)
                W: torch.Tensor = torch.linalg.lstsq(X, y).solution
                assert W.shape == (2, 192 * 288)
                y_bar: torch.Tensor = X.matmul(W)
                assert y_bar.shape == (n_years, 192 * 288)
                y_bar = y_bar.reshape(n_years, 1, 1, 192, 288, 1)
                for i, year in enumerate(self.years):
                    results.set(
                        var_name=var_name, sim_id=sim_id, year=year, 
                        value=y_bar[i]
                    )
        
        return results

    def __call__(self, data: DataContainer) -> DataContainer:
        trend: DataContainer = self.linear_regression(data)
        result: DataContainer = DataContainer(self.var_names, self.sim_ids, self.years)
        for var_name in self.var_names:
            for sim_id in self.sim_ids:
                for year in self.years:
                    a: torch.Tensor = data.get(var_name=var_name, sim_id=sim_id, year=year)
                    b: torch.Tensor = trend.get(var_name=var_name, sim_id=sim_id, year=year)
                    assert a.shape == (1, 365, 192, 288, 1)
                    assert b.shape == (1, 1, 192, 288, 1)
                    print(f"Writing year {year}")
                    result.set(
                        # broadcast along T (dim=1)
                        var_name=var_name, sim_id=sim_id, year=year, value=a - b,
                    )

        return result


class SeasonalityRemover:
    # TODO: implement
    pass

class SampleWriter:
    # TODO: implement (write to disk, each sample is a .pt file)
    pass

class CESM2(Dataset):
    # TODO: implement (simply read from files) => write once, use multiple times
    pass

        


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # test
    container = DataContainer(var_names=["PRECT","FLUT","TS"], sim_ids=["1181.010", "1161.009"], years=list(range(2000, 2015)))
    for var_name in container.var_names:
        for sim_id in container.sim_ids:
            for year in container.years:
                container.set(var_name, sim_id, year, value=UnitaryDataReader(var_name, sim_id, year).tensor)

    detrender = Detrender()
    detrended_data: DataContainer = detrender(container)
    a: torch.Tensor = detrended_data.get(var_name="PRECT", sim_id="1181.010", year=2010)


