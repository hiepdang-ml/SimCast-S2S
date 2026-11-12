import re
import json
import pathlib
from typing import Any
from functools import cached_property

import torch
from torch.utils.data import Dataset
from .utils import SampleInfo, DataBatch
from common.configs import MetaData


class _GenericDataset(Dataset):

    def __init__(self, metadata: MetaData):
        super().__init__()
        assert metadata.dataset_name in {"cesm2", "era5"}
        self.metadata: MetaData = metadata
        meta_dict: dict[str, Any] = self.metadata.to_dict()
        self.H, self.W = self.metadata.resolution
        # validate consistent write/read metadata
        with open(self.metadata.write_directory.joinpath("metadata/metadata.json"), mode="r") as file:
            loaded_dict: dict[str, Any] = json.load(fp=file)
            loaded_dict["resolution"] = tuple(loaded_dict["resolution"])
            loaded_dict.pop("sim_ids"); loaded_dict.pop("input_vars"); loaded_dict.pop("output_vars") # noqa: E702
            for k in loaded_dict.keys():
                if meta_dict[k] != loaded_dict[k]:
                    raise RuntimeError(
                        f"Inconsistent value for '{k}': "
                        f"loaded .pt file has {k} = {loaded_dict[k]!r}, but metadata has {k} = {meta_dict[k]!r}."
                    )

        # Preload filepath lookups for efficiency
        self.input_filepaths: dict[str, list[pathlib.Path]] = self.__collect_filepaths(
            kind="input", var_names=self.metadata.input_vars,
        )
        self.output_filepaths: dict[str, list[pathlib.Path]] = self.__collect_filepaths(
            kind="output", var_names=self.metadata.output_vars,
        )
        self.n_samples: int = len(self.output_filepaths[self.metadata.output_vars[0]])
        self.__check_data_integrity(input_files=self.input_filepaths, output_files=self.output_filepaths)

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

        sim_id, var_name, year = input_parts[1].split("_")
        output_sim_id, _, output_year = output_parts[1].split("_")
        assert sim_id == output_sim_id and year == output_year
        input_days: list[str] = input_parts[2].split("_")
        output_days: list[str] = output_parts[2].split("_")

        def full_date(value: str) -> str:
            if len(value) != 8 or not value.isdigit():
                raise ValueError(f"Invalid date in tensor filename: {value}")
            return f"{value[:4]}/{value[4:6]}/{value[6:]}"

        return SampleInfo(
            sim_id=sim_id,
            in_startdate=full_date(input_days[0]),
            in_enddate=full_date(input_days[-1]),
            out_startdate=full_date(output_days[0]),
            out_enddate=full_date(output_days[-1]),
        )

    def __getitem__(self, idx: int) -> tuple[SampleInfo, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_path: pathlib.Path | None = None
        output_path: pathlib.Path | None = None
        input_yearday_indices: torch.Tensor | None = None
        output_yearday_indices: torch.Tensor | None = None
        input_var_tensors: list[torch.Tensor] = []
        for var_name in self.metadata.input_vars:
            # Note: input_yearday_indices remains the same regardless of var_name
            input_path = self.input_filepaths[var_name][idx]
            input_yearday_indices, input_var_tensor = torch.load(
                f=input_path, weights_only=True, map_location="cpu"
            )
            assert isinstance(input_yearday_indices, torch.Tensor)
            assert input_yearday_indices.shape == (self.metadata.n_input_days,)
            assert input_var_tensor.shape == (self.metadata.n_input_days, *self.metadata.resolution)
            input_var_tensors.append(input_var_tensor)

        output_var_tensors: list[torch.Tensor] = []
        for var_name in self.metadata.output_vars:
            # Note: output_yearday_indices remains the same regardless of var_name
            output_path = self.output_filepaths[var_name][idx]
            output_yearday_indices, output_var_tensor = torch.load(
                f=output_path, weights_only=True, map_location="cpu"
            )
            assert isinstance(output_yearday_indices, torch.Tensor)
            assert output_yearday_indices.shape == (self.metadata.n_output_days,)
            assert output_var_tensor.shape == (self.metadata.n_output_days, self.H, self.W)
            output_var_tensors.append(output_var_tensor)

        if input_path is None or output_path is None or input_yearday_indices is None or output_yearday_indices is None:
            raise ValueError("A sample requires at least one input variable and one output variable")
        sampleinfo: SampleInfo = self._get_sample_info(
            input_filename=input_path.name, output_filename=output_path.name
        )
        input_tensor: torch.Tensor = torch.stack(tensors=input_var_tensors, dim=-1)
        output_tensor: torch.Tensor = torch.stack(tensors=output_var_tensors, dim=-1)
        # validate before return
        assert input_yearday_indices.shape == (self.metadata.n_input_days,)
        assert output_yearday_indices.shape == (self.metadata.n_output_days,)
        assert input_tensor.shape == (self.metadata.n_input_days, self.H, self.W, len(self.metadata.input_vars))
        assert output_tensor.shape == (self.metadata.n_output_days, self.H, self.W, len(self.metadata.output_vars))
        return sampleinfo, input_yearday_indices, output_yearday_indices, input_tensor, output_tensor

    def __len__(self) -> int:
        return self.n_samples

    def __collect_filepaths(self, kind: str, var_names: list[str]) -> dict[str, list[pathlib.Path]]:
        """
        Collect and sort .pt files for each variable by sample id
        """
        simid_pattern: re.Pattern[str] = re.compile("|".join(map(re.escape, self.metadata.sim_ids)))
        results: dict[str, list[pathlib.Path]] = {}
        base_dir: pathlib.Path = self.metadata.write_directory.joinpath(kind)
        for var_name in var_names:
            var_dir: pathlib.Path = base_dir.joinpath(var_name)
            results[var_name] = sorted(
                (f for f in var_dir.glob("*.pt") if simid_pattern.search(f.name)),
                key=lambda f: f.name[:6],
            )
        return results

    def __check_data_integrity(
        self,
        input_files: dict[str, list[pathlib.Path]],
        output_files: dict[str, list[pathlib.Path]],
    ) -> None:
        # Check missing files
        lengths: set[int] = set([self.n_samples])
        for var_name in self.metadata.input_vars:
            lengths.add(len(input_files[var_name]))
            assert len(lengths) == 1, f"{var_name} has different number of input files"

        for var_name in self.metadata.output_vars:
            lengths.add(len(output_files[var_name]))
            assert len(lengths) == 1, f"{var_name} has different number of output files"

        # Check identical indices
        for i in range(self.n_samples):
            indices: set[int] = set()
            for var_name in self.metadata.input_vars:
                indices.add(int(input_files[var_name][i].name[:6]))
                assert len(indices) == 1, f"Unmatched input file at index {i}"
            for var_name in self.metadata.output_vars:
                indices.add(int(output_files[var_name][i].name[:6]))
                assert len(indices) == 1, f"Unmatched output file at index {i}"

    @staticmethod
    def collate_fn(
        batch: list[tuple[SampleInfo, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> DataBatch:
        # get iterables of objects of the same kind
        sampleinfos, input_indices, output_indices, input_tensors, output_tensors = zip(*batch)
        databatch: DataBatch = (
            list(sampleinfos),
            torch.stack(input_indices, dim=0),
            torch.stack(output_indices, dim=0),
            torch.stack(input_tensors, dim=0),
            torch.stack(output_tensors, dim=0),
        )
        return databatch


class CESM2(_GenericDataset):
    def __init__(self, metadata: MetaData):
        super().__init__(metadata)
        assert self.metadata.dataset_name == "cesm2"


class ERA5(_GenericDataset):
    def __init__(self, metadata: MetaData):
        super().__init__(metadata)
        assert self.metadata.dataset_name == "era5"


if __name__ == "__main__":
    metadata = MetaData(dataset_name="cesm2", tp="test")
    dataset = CESM2(metadata)

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=CESM2.collate_fn)
    batch = next(iter(dataloader))
