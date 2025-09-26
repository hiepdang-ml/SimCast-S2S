import json
import pathlib
from typing import Any
from functools import cached_property

import torch
from torch.utils.data import Dataset
from datasets.common.utils import SampleInfo, DataBatch
from common.configs import MetaData


class CESM2(Dataset):

    def __init__(self, metadata: MetaData):
        super().__init__()
        assert metadata.dataset_name == "cesm2"
        self.metadata: MetaData = metadata
        meta_dict: dict[str, Any] = self.metadata.to_dict()
        # validate consistent write/read metadata
        with open(self.metadata.write_directory.joinpath(f"metadata/metadata.json"), mode="r") as file:
            loaded_dict: dict[str, Any] = json.load(fp=file)
            loaded_dict["resolution"] = tuple(loaded_dict["resolution"])
            loaded_dict.pop("input_vars")
            for k in loaded_dict.keys():
                if meta_dict[k] != loaded_dict[k]:
                    raise RuntimeError(
                        f"Inconsistent value for '{k}': "
                        f"loaded .pt file has {k} = {loaded_dict[k]!r}, but metadata has {k} = {meta_dict[k]!r}."
                    )

        # Pick any input var_name to count n samples
        self.n_samples: int = len(list(self.metadata.write_directory.glob(f"input/{self.metadata.var_names[0]}/*.pt")))
        # Preload filepaths for efficiency
        self.__input_filepaths: dict[str, list[pathlib.Path]] = {
            var_name: sorted(self.metadata.write_directory.glob(f"input/{var_name}/*.pt"), key=lambda x: x.name[:6])
            for var_name in self.metadata.input_vars
        }
        self.__output_filepaths: dict[str, list[pathlib.Path]] = {
            var_name: sorted(self.metadata.write_directory.glob(f"output/{var_name}/*.pt"), key=lambda x: x.name[:6])
            for var_name in self.metadata.output_vars
        }

    @cached_property
    def indices_by_context_group(self) -> dict[str, list[int]]:
        # Precompute variable indices by context_group
        result: dict[str, list[int]] = {}   # context_group: [indices]
        for context_group, var_names in MetaData.VAR_LOOKUP_TABLE[type(self).__name__.lower()].items():
            result[context_group] = [i for i, name in enumerate(self.metadata.input_vars) if name in var_names]
        return result
    
    @staticmethod
    def _get_sample_info(input_filename: str, output_filename) -> SampleInfo:
        input_parts: list[str] = input_filename.removesuffix(".pt").split("__")
        output_parts: list[str] = output_filename.removesuffix(".pt").split("__")
        assert input_parts[0][:6] == output_parts[0][:6]  # sample_id
        assert input_parts[0].split(".")[1] == output_parts[0].split(".")[1] # tp ("train", "val", "test")
        assert input_parts[0].split(".")[-1] == "input" and output_parts[0].split(".")[-1] == "output"

        year: str = input_parts[1][-4:]
        input_days: list[str] = input_parts[2].split("_")
        output_days: list[str] = output_parts[2].split("_")
        return SampleInfo(
            sim_id=input_parts[1][:8],
            in_startdate=f"{year}/{input_days[0][:2]}/{input_days[0][-2:]}",
            in_enddate=f"{year}/{input_days[-1][:2]}/{input_days[-1][-2:]}",
            out_startdate=f"{year}/{output_days[0][:2]}/{output_days[0][-2:]}",
            out_enddate=f"{year}/{output_days[-1][:2]}/{output_days[-1][-2:]}",
        )

    def __getitem__(self, idx: int) -> tuple[SampleInfo, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_var_tensors: list[torch.Tensor] = []
        for var_name in self.metadata.input_vars:
            # Note: input_yearday_indices remains the same regardless of var_name
            input_path: pathlib.Path = self.__input_filepaths[var_name][idx]
            input_yearday_indices, input_var_tensor = torch.load(
                f=input_path, weights_only=True, map_location="cpu"
            )
            assert input_yearday_indices.shape == (self.metadata.n_input_days,)
            assert input_var_tensor.shape == (self.metadata.n_input_days, 192, 288)
            input_var_tensors.append(input_var_tensor)
        
        output_var_tensors: list[torch.Tensor] = []
        for var_name in self.metadata.output_vars:
            # Note: output_yearday_indices remains the same regardless of var_name
            output_path: pathlib.Path = self.__output_filepaths[var_name][idx]
            output_yearday_indices, output_var_tensor = torch.load(
                f=output_path, weights_only=True, map_location="cpu"
            )
            assert output_yearday_indices.shape == (self.metadata.n_output_days,)
            assert output_var_tensor.shape == (1, 192, 288)
            output_var_tensors.append(output_var_tensor)
    
        sampleinfo: SampleInfo = CESM2._get_sample_info(
            input_filename=input_path.name, output_filename=output_path.name
        )
        input_tensor: torch.Tensor = torch.stack(tensors=input_var_tensors, dim=-1)
        output_tensor: torch.Tensor = torch.stack(tensors=output_var_tensors, dim=-1)
        # validate before return
        assert input_yearday_indices.shape == (self.metadata.n_input_days,)
        assert output_yearday_indices.shape == (self.metadata.n_output_days,)
        assert input_tensor.shape == (self.metadata.n_input_days, 192, 288, len(self.metadata.input_vars))
        assert output_tensor.shape == (1, 192, 288, len(self.metadata.output_vars))
        return sampleinfo, input_yearday_indices, output_yearday_indices, input_tensor, output_tensor

    def __len__(self) -> int:
        return self.n_samples

    @staticmethod
    def collate_fn(
        batch: list[tuple[SampleInfo, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> DataBatch:
        # get iterables of objects of the same kind
        sampleinfos, input_indices, output_indices, input_tensors, output_tensors = zip(*batch)
        return (
            list(sampleinfos),
            torch.stack(input_indices, dim=0),
            torch.stack(output_indices, dim=0),
            torch.stack(input_tensors, dim=0),
            torch.stack(output_tensors, dim=0),
        )


if __name__ == "__main__":
    metadata = MetaData(dataset_name="cesm2", tp="test")
    dataset = CESM2(metadata)

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=CESM2.collate_fn)
    batch = next(iter(dataloader))


