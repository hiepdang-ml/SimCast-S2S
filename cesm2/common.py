from __future__ import annotations

from collections import defaultdict
import yaml
import pathlib
from itertools import product
from typing import *
import torch


class MetaData:

    """ Singleton """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        with open("./config.yaml", mode="r") as file:
            self.__config: Dict[str, Any] = yaml.safe_load(file)

        self.load_metadata_from_config()
        self.var_names: List[str] = sorted(set(self.input_vars + self.output_vars))
        self.years: List[int] = list(range(self.start_year, self.end_year + 1))
        self.combinations: List[Tuple[str, str, int]] = list(product(self.var_names, self.sim_ids, self.years))

    def load_metadata_from_config(self) -> None:
        self.input_vars: List[str] = self.__config["data"]["input_vars"]
        self.output_vars: List[str] = self.__config["data"]["output_vars"]
        self.sim_ids: List[str] = self.__config["data"]["sim_ids"]
        self.start_year: int = self.__config["data"]["start_year"]
        self.end_year: int = self.__config["data"]["end_year"]
        self.write_directory: pathlib.Path = pathlib.Path(self.__config["data"]["write_directory"])
        self.saved_metadata_path: pathlib.Path = pathlib.Path(self.__config["data"]["saved_metadata_path"])
        self.n_input_days: int = self.__config["data"]["n_input_days"]
        self.n_lead_days: int = self.__config["data"]["n_lead_days"]
        self.n_output_days: int = self.__config["data"]["n_output_days"]
        self.n_step_days: int = self.__config["data"]["n_step_days"]
        self.need_daily_predictions: bool = self.__config["data"]["need_daily_predictions"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_vars": self.input_vars,
            "output_vars": self.output_vars,
            "sim_ids": self.sim_ids,
            "years": self.years,
            "n_input_days": self.n_input_days,
            "n_lead_days": self.n_lead_days,
            "n_output_days": self.n_output_days,
            "n_step_days": self.n_step_days,
            "need_daily_predictions": self.need_daily_predictions,
        }


class DataContainer:

    """
    Universal data container in place of xarray
    """

    def __init__(self, metadata: MetaData) -> None:
        self.metadata: MetaData = metadata
        self.__container: Dict[str, Dict[str, Dict[int, Optional[torch.Tensor]]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for var_name, sim_id, year in self.metadata.combinations:
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
            for var_name in self.metadata.var_names
            for sim_id in self.metadata.sim_ids
            for year in self.metadata.years
        )

    def __add__(self, other) -> DataContainer:
        assert isinstance(other, DataContainer)
        assert self.is_completed
        assert other.is_completed

        output: DataContainer = DataContainer(self.metadata)
        for var_name, sim_id, year in self.metadata.combinations:
            output.set(
                var_name, sim_id, year,
                value=(
                    self.get(var_name=var_name, sim_id=sim_id, year=year)
                    + other.get(var_name=var_name, sim_id=sim_id, year=year)
                )
            )
        return output

    def __sub__(self, other) -> DataContainer:
        return self + (-other)

    def __neg__(self) -> DataContainer:
        assert self.is_completed

        output: DataContainer = DataContainer(self.metadata)
        for var_name, sim_id, year in self.metadata.combinations:
            output.set(
                var_name, sim_id, year,
                value= -self.get(var_name=var_name, sim_id=sim_id, year=year)
            )
        return output

    def yearly_agg(self, reduce_func: Literal["sum", "mean", "std"]) -> DataContainer:
        assert self.is_completed

        func: Callable[[torch.Tensor], torch.Tensor] = {
            "sum": lambda x: x.sum(dim=0, keepdim=True),
            "mean": lambda x: x.mean(dim=0, keepdim=True),
            "std": lambda x: x.std(dim=0, keepdim=True)
        }.get(reduce_func)

        output: DataContainer = DataContainer(self.metadata)
        for var_name, sim_id, year in self.metadata.combinations:
            output.set(
                var_name, sim_id, year,
                value=func(self.get(var_name=var_name, sim_id=sim_id, year=year))
            )
        return output


