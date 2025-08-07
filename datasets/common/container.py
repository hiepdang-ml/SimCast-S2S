from __future__ import annotations

from collections import defaultdict
from typing import *

import torch
from common.configs import MetaData


class DataContainer:

    """
    Universal dataset container in place of xarray
    """

    def __init__(self, metadata: MetaData) -> None:
        self.metadata: MetaData = metadata
        self.__container: Dict[str, Dict[str, Dict[int, Optional[torch.Tensor]]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for sim_id, var_name, year in self.metadata.combinations:
            self.set(sim_id=sim_id, var_name=var_name, year=year, value=None)

    def get(self, sim_id: str, var_name: str, year: int | None) -> torch.Tensor| Dict[int, torch.Tensor] | None:
        if year is None:
            return self.__container[sim_id][var_name]
        return self.__container[sim_id][var_name][year]

    def set(self, sim_id: str, var_name: str, year: int | None, value: torch.Tensor | Dict[int, torch.Tensor] | None) -> None:
        if year is None:
            assert isinstance(value, dict)
            self.__container[sim_id][var_name] = value
        else:
            assert isinstance(value, (torch.Tensor, type(None)))
            self.__container[sim_id][var_name][year] = value

    @property
    def is_completed(self) -> bool:
        return all(
            self.get(sim_id=sim_id, var_name=var_name, year=year) is not None
            for sim_id in self.metadata.sim_ids
            for var_name in self.metadata.var_names
            for year in self.metadata.years
        )

    def to(self, device: torch.device) -> DataContainer:
        assert self.is_completed, "DataContainer must be completed to be sent to new device"
        for sim_id, var_name, year in self.metadata.combinations:
            value = self.get(sim_id=sim_id, var_name=var_name, year=year)
            if isinstance(value, torch.Tensor):
                self.set(sim_id=sim_id, var_name=var_name, year=year, value=value.to(device=device))
        return self

    def yearly_agg(self, reduce_func: Literal["sum", "mean", "std"]) -> DataContainer:
        assert self.is_completed

        func: Callable[[torch.Tensor], torch.Tensor] = {
            "sum": lambda x: x.sum(dim=0, keepdim=True),
            "mean": lambda x: x.mean(dim=0, keepdim=True),
            "std": lambda x: x.std(dim=0, keepdim=True)
        }.get(reduce_func)

        output: DataContainer = DataContainer(self.metadata)
        for sim_id, var_name, year in self.metadata.combinations:
            output.set(
                sim_id=sim_id, var_name=var_name, year=year,
                value=func(self.get(sim_id=sim_id, var_name=var_name, year=year))
            )
        return output


