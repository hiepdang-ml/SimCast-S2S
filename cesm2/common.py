from __future__ import annotations

from collections import defaultdict
import yaml
import pathlib
from itertools import product
from typing import *

import torch

class MetaData:

    def __init__(self, tp: Literal["train", "val", "test"]):
        with open("./config.yaml", mode="r") as file:
            self.__config: Dict[str, Any] = yaml.safe_load(file)

        self.tp: Literal["train", "val", "test"] = tp
        self.load_metadata_from_config()
        self.var_names: List[str] = sorted(set(self.input_vars + self.output_vars))
        self.years: List[int] = list(range(self.start_year, self.end_year + 1))
        self.combinations: List[Tuple[str, str, int]] = list(product(self.var_names, self.sim_ids, self.years))
        self.n_years: int = len(self.years)

    def load_metadata_from_config(self) -> None:
        self.device: str = self.__config["dataset"]["device"]
        self.input_vars: List[str] = self.__config["dataset"]["input_vars"]
        self.output_vars: List[str] = self.__config["dataset"]["output_vars"]
        self.sim_ids: List[str] = self.__config["dataset"]["sim_ids"]

        if self.tp == "train":
            self.start_year: int = self.__config["dataset"]["train_start_year"]
            self.end_year: int = self.__config["dataset"]["train_end_year"]
            self.write_directory: pathlib.Path = pathlib.Path(self.__config["dataset"]["train_write_directory"])
            self.saved_metadata_path: pathlib.Path = pathlib.Path(self.__config["dataset"]["train_metadata_path"])
        elif self.tp == "val":
            self.start_year: int = self.__config["dataset"]["val_start_year"]
            self.end_year: int = self.__config["dataset"]["val_end_year"]
            self.write_directory: pathlib.Path = pathlib.Path(self.__config["dataset"]["val_write_directory"])
            self.saved_metadata_path: pathlib.Path = pathlib.Path(self.__config["dataset"]["val_metadata_path"])
        elif self.tp == "test":
            self.start_year: int = self.__config["dataset"]["test_start_year"]
            self.end_year: int = self.__config["dataset"]["test_end_year"]
            self.write_directory: pathlib.Path = pathlib.Path(self.__config["dataset"]["test_write_directory"])
            self.saved_metadata_path: pathlib.Path = pathlib.Path(self.__config["dataset"]["test_metadata_path"])
        else:
            raise ValueError(f"Invalid tp for MetaData, expected one of ['train', 'val', 'test'], get: '{self.tp}'")
        
        self.n_input_days: int = self.__config["dataset"]["n_input_days"]
        self.n_lead_days: int = self.__config["dataset"]["n_lead_days"]
        self.n_output_days: int = self.__config["dataset"]["n_output_days"]
        self.n_step_days: int = self.__config["dataset"]["n_step_days"]
        self.climatological_window_size: int = self.__config["dataset"]["climatological_window_size"]
        self.need_daily_predictions: bool = self.__config["dataset"]["need_daily_predictions"]

        self.detrender_state_directory: pathlib.Path = pathlib.Path(self.__config["dataset"]["detrender_state_directory"])
        self.climatology_state_directory: pathlib.Path = pathlib.Path(self.__config["dataset"]["climatology_state_directory"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.tp,
            "device": self.device,
            "input_vars": self.input_vars,
            "output_vars": self.output_vars,
            "sim_ids": self.sim_ids,
            "years": self.years,
            "n_input_days": self.n_input_days,
            "n_lead_days": self.n_lead_days,
            "n_output_days": self.n_output_days,
            "n_step_days": self.n_step_days,
            "climatological_window_size": self.climatological_window_size,
            "need_daily_predictions": self.need_daily_predictions,
        }

class DataContainer:

    """
    Universal dataset container in place of xarray
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

    def to(self, device: torch.device) -> DataContainer:
        assert self.is_completed, "DataContainer must be completed to be sent to new device"
        for var_name, sim_id, year in self.metadata.combinations:
            value = self.get(var_name, sim_id, year)
            if isinstance(value, torch.Tensor):
                self.set(var_name, sim_id, year, value.to(device=device))
        return self

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


