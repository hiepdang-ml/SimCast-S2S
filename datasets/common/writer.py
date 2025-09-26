import datetime as dt
import json
import pathlib
from typing import Literal, Any

import torch
from datasets.common.container import VariableContainer
from common.configs import MetaData


class DataWriter:

    _SAMPLE_COUNTER: int = 0

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
        self.__datestrings: list[str] = [
            # choose 2025 since not a leap year
            (dt.datetime(2025, 1, 1) + dt.timedelta(days=i)).strftime("%m%d")
            for i in range(365)
        ]
        self.metadata_path: pathlib.Path = self.metadata.write_directory.joinpath("metadata")
        self.metadata_path.mkdir(parents=True, exist_ok=True)

        self.input_path: pathlib.Path = self.metadata.write_directory.joinpath("input")
        self.output_path: pathlib.Path = self.metadata.write_directory.joinpath("output")
        self.input_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)

        for var_name in self.metadata.input_vars:
            self.input_path.joinpath(var_name).mkdir(exist_ok=True)

        for var_name in self.metadata.output_vars:
            self.output_path.joinpath(var_name).mkdir(exist_ok=True)
        
        self.__save_metadata()

    def __save_metadata(self) -> None:
        filepath: pathlib.Path = pathlib.Path(self.metadata_path.joinpath("metadata.json"))
        d: dict[str, Any] = self.metadata.to_dict()
        with open(file=filepath, mode="w") as file:
            json.dump(obj=d, fp=file)

    def __construct_file_name(
        self,
        sim_id: str, var_name: str, year: int, 
        input_or_output: Literal["input", "output"],
        sample_index: int, 
        yearday_indices: list[int]
    ) -> str:
        sample_id: str = f"{sample_index:06d}"
        prefix: str = f"{sample_id}.{self.metadata.tp}.{input_or_output}__{sim_id}_{var_name}_{year}__"
        suffix: str = f"{'_'.join([self.__datestrings[i] for i in yearday_indices])}"
        return f"{prefix}{suffix}.pt"

    def __write_one_sample(
        self, 
        var_container: VariableContainer,
        sim_id: str, year: int,
        yearday_index: int, sample_index: int,
    ) -> None:
        # Get yearday_indices
        input_yearday_indices: torch.Tensor = torch.tensor(
            range(yearday_index, yearday_index + self.metadata.n_input_days),
            dtype=torch.int,
            device=self.metadata.device,
        )
        output_yearday_indices: torch.Tensor = torch.tensor(
            range(
                yearday_index + self.metadata.n_input_days + self.metadata.n_lead_days,
                yearday_index + self.metadata.n_input_days + self.metadata.n_lead_days + self.metadata.n_output_days,
            ),
            dtype=torch.int,
            device=self.metadata.device,
        )
        if var_container.var_name in self.metadata.input_vars:
            # Input: Get data
            year_tensor: torch.Tensor = var_container.get(sim_id=sim_id, year=year)
            assert year_tensor.shape == (365, 192, 288)
            input_tensor: torch.Tensor = year_tensor[input_yearday_indices]
            assert input_tensor.shape == (self.metadata.n_input_days, 192, 288)
            # Input: Write to .pt
            filename: str = self.__construct_file_name(
                sim_id=sim_id, var_name=var_container.var_name, year=year, 
                input_or_output="input",
                sample_index=sample_index, 
                yearday_indices=input_yearday_indices.cpu().tolist(),
            )
            torch.save(
                obj=(input_yearday_indices, input_tensor),
                f=self.input_path.joinpath(f"{var_container.var_name}/{filename}"),
            )

        if var_container.var_name in self.metadata.output_vars:
            # Output: Get data
            year_tensor: torch.Tensor = var_container.get(sim_id=sim_id, year=year)
            assert year_tensor.shape == (365, 192, 288)
            output_tensor: torch.Tensor = year_tensor[output_yearday_indices]
            assert output_tensor.shape == (self.metadata.n_output_days, 192, 288)
            output_tensor = output_tensor.mean(dim=0, keepdim=True)
            assert output_tensor.shape == (1, 192, 288)
            # Output: Write to .pt
            filename: str = self.__construct_file_name(
                sim_id=sim_id, var_name=var_container.var_name, year=year,
                input_or_output="output",
                sample_index=sample_index,
                yearday_indices=output_yearday_indices.cpu().tolist(),
            )
            torch.save(
                obj=(output_yearday_indices, output_tensor),
                f=self.output_path.joinpath(f"{var_container.var_name}/{filename}"),
            )

    def __call__(self, var_container: VariableContainer) -> None:
        bound: int = 365 - self.metadata.n_input_days - self.metadata.n_lead_days - self.metadata.n_output_days
        step: int = self.metadata.n_step_days
        for sim_id in self.metadata.sim_ids:
            for year in self.metadata.years:
                for t in range(0, bound, step):
                    self.__write_one_sample(
                        var_container=var_container, sim_id=sim_id, year=year,
                        yearday_index=t, sample_index=DataWriter._SAMPLE_COUNTER,
                    )
                    DataWriter._SAMPLE_COUNTER += 1

        # Reset counter
        DataWriter._SAMPLE_COUNTER = 0



