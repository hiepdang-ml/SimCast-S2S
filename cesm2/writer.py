import datetime as dt
import json
import os
import pathlib
from typing import *

import torch
from cesm2.common import DataContainer, MetaData


class DataWriter:

    def __init__(self, metadata: MetaData):
        """
        |---------------||---------------||---------------|
           n_input_days     n_lead_days     n_output_days

         ............... |---------------||---------------||---------------|
           n_step_days      n_input_days     n_lead_days     n_output_days

        |--------------------------------------------------------------------------->
        0                                                                           365
        """
        self.metadata: MetaData = metadata
        self.__datestrings: List[str] = [
            # choose 2025 since not a leap year
            (dt.datetime(2025, 1, 1) + dt.timedelta(days=i)).strftime("%m%d")
            for i in range(365)
        ]
        os.makedirs(name=self.metadata.write_directory, exist_ok=True)
        os.makedirs(name=self.metadata.saved_metadata_path.parent, exist_ok=True)
        self.__save_config()

    def __save_config(self) -> None:
        with open(file=pathlib.Path(self.metadata.saved_metadata_path), mode="w") as file:
            metadata: Dict[str, Any] = {
                "input_vars": self.metadata.input_vars,
                "output_vars": self.metadata.output_vars,
                "sim_ids": self.metadata.sim_ids,
                "years": self.metadata.years,
                "n_input_days": self.metadata.n_input_days,
                "n_lead_days": self.metadata.n_lead_days,
                "n_output_days": self.metadata.n_output_days,
                "n_step_days": self.metadata.n_step_days,
                "need_daily_predictions": self.metadata.need_daily_predictions,
            }
            json.dump(obj=metadata, fp=file)

    def __construct_file_name(
        self,
        sim_id: str, year: int, input_indices: List[int], output_indices: List[int]
    ) -> str:
        return (
            f"{sim_id.replace('.','')}_{year}__"
            f"{'_'.join([self.__datestrings[i] for i in input_indices])}__"
            f"{'_'.join([self.__datestrings[i] for i in output_indices])}.pt"
        )

    def __write_one_sample(self, input: DataContainer, sim_id: str, year: int, sample_index: int) -> None:
        # Input
        input_indices: torch.Tensor = torch.tensor(
            range(sample_index, sample_index + self.metadata.n_input_days),
            dtype=torch.int
        )
        input_tensors: List[str] = []
        for var_name in self.metadata.input_vars:
            tensor: torch.Tensor = input.get(var_name, sim_id, year)
            assert tensor.shape == (365, 192, 288)
            var_tensor: torch.Tensor = tensor[input_indices]
            assert var_tensor.shape == (self.metadata.n_input_days, 192, 288)
            input_tensors.append(var_tensor)

        input_tensor: torch.Tensor = torch.stack(tensors=input_tensors, dim=3)
        assert input_tensor.shape == (self.metadata.n_input_days, 192, 288, len(self.metadata.input_vars))

        # Output
        output_indices: torch.Tensor = torch.tensor(
            range(
                sample_index + self.metadata.n_input_days + self.metadata.n_lead_days,
                sample_index + self.metadata.n_input_days + self.metadata.n_lead_days + self.metadata.n_output_days,
            ),
            dtype=torch.int
        )
        output_tensors: List[str] = []
        for var_name in self.metadata.output_vars:
            tensor: torch.Tensor = input.get(var_name, sim_id, year)
            assert tensor.shape == (365, 192, 288)
            var_tensor: torch.Tensor = tensor[output_indices]
            assert var_tensor.shape == (self.metadata.n_output_days, 192, 288)
            output_tensors.append(var_tensor)

        output_tensor: torch.Tensor = torch.stack(tensors=output_tensors, dim=3)
        assert output_tensor.shape == (self.metadata.n_output_days, 192, 288, len(self.metadata.output_vars))

        if not self.metadata.need_daily_predictions:
            output_tensor = output_tensor.mean(dim=0)
            assert output_tensor.shape == (192, 288, len(self.metadata.output_vars))

        torch.save(
            obj=(input_indices, output_indices, input_tensor, output_tensor),
            f=pathlib.Path(
                self.metadata.write_directory,
                self.__construct_file_name(
                    sim_id=sim_id, year=year,
                    input_indices=input_indices, output_indices=output_indices
                )
            )
        )

    def __call__(self, input: DataContainer) -> None:
        for sim_id in self.metadata.sim_ids:
            bound: int = 365 - self.metadata.n_input_days - self.metadata.n_lead_days - self.metadata.n_output_days
            step: int = self.metadata.n_step_days
            for year in self.metadata.years:
                for i in range(0, bound, step):
                    self.__write_one_sample(input, sim_id=sim_id, year=year, sample_index=i)

