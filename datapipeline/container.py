from __future__ import annotations

from collections import defaultdict
from typing import Callable, Literal

import torch
from common.configs import MetaData


class VariableContainer:

    """
    Universal data container for one single variable.
    """

    def __init__(self, var_name: str, metadata: MetaData) -> None:
        assert var_name in metadata.var_names
        self.var_name: str = var_name
        self.metadata: MetaData = metadata
        self.__container: dict[str, dict[int, torch.Tensor | None]] = defaultdict(dict)  # sim_id, year
        for sim_id, year in self.metadata.combinations:
            self.set(sim_id=sim_id, year=year, value=None)

    def get(self, sim_id: str, year: int | None) -> torch.Tensor | dict[int, torch.Tensor | None] | None:
        if year is None:
            return self.__container[sim_id]
        return self.__container[sim_id][year]

    def set(self, sim_id: str, year: int | None, value: torch.Tensor | dict[int, torch.Tensor | None] | None) -> None:
        if year is None:
            assert isinstance(value, dict)
            self.__container[sim_id] = value
        else:
            assert isinstance(value, (torch.Tensor, type(None)))
            self.__container[sim_id][year] = value

    @property
    def is_completed(self) -> bool:
        return all(
            self.get(sim_id=sim_id, year=year) is not None
            for sim_id, year in self.metadata.combinations
        )

    def to(self, device: torch.device | str) -> VariableContainer:
        assert self.is_completed, "VariableContainer must be completed before moving to new device"
        for sim_id, year in self.metadata.combinations:
            value = self.get(sim_id=sim_id, year=year)
            if isinstance(value, torch.Tensor):
                self.set(sim_id=sim_id, year=year, value=value.to(device=device))
        return self

    def yearly_agg(self, reduce_func: Literal["sum", "mean", "std"]) -> VariableContainer:
        assert self.is_completed

        func: Callable[[torch.Tensor], torch.Tensor] = {
            "sum": lambda x: x.sum(dim=0, keepdim=True),
            "mean": lambda x: x.mean(dim=0, keepdim=True),
            "std": lambda x: x.std(dim=0, keepdim=True)
        }.get(reduce_func)

        output: VariableContainer = VariableContainer(var_name=self.var_name, metadata=self.metadata)
        for sim_id, year in self.metadata.combinations:
            _input: torch.Tensor = self.get(sim_id=sim_id, year=year)
            output.set(sim_id=sim_id, year=year, value=func(_input))

        return output
