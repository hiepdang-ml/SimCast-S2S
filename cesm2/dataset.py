import json
import pathlib
from functools import cached_property
from typing import *

import torch
from torch.utils.data import Dataset
from cesm2.utils import SampleInfo, DataBatch
from common.configs import MetaData


class CESM2(Dataset):

    def __init__(self, metadata: MetaData):
        super().__init__()
        self.metadata: MetaData = metadata
        meta_dict: Dict[str, Any] = self.metadata.to_dict()
        # validate consistent write/read metadata
        for sim_id in metadata.sim_ids:
            with open(self.metadata.write_directory.joinpath(f"metadata/{sim_id}.json"), mode="r") as file:
                loaded_dict: Dict[str, Any] = json.load(fp=file)
                loaded_dict.pop("sim_ids")
                for k in loaded_dict.keys():
                    if meta_dict[k] != loaded_dict[k]:
                        raise RuntimeError(
                            f"Inconsistent value for '{k}': "
                            f"loaded .pt file has {k} = {loaded_dict[k]!r}, but metadata has {k} = {meta_dict[k]!r}."
                        )

    @cached_property
    def filepaths(self) -> List[pathlib.Path]:
        return sorted(self.metadata.write_directory.glob("*.pt"))
    
    @staticmethod
    def _get_sample_info(filename: str) -> SampleInfo:
        parts: List[str] = filename.removesuffix(".pt").split("__")
        year: str = parts[0][-4:]
        in_parts: List[str] = parts[1].split("_")
        out_parts: List[str] = parts[2].split("_")
        return SampleInfo(
            sim_id=filename[:8],
            in_startdate=f"{year}/{in_parts[0][:2]}/{in_parts[0][-2:]}",
            in_enddate=f"{year}/{in_parts[-1][:2]}/{in_parts[-1][-2:]}",
            out_startdate=f"{year}/{out_parts[0][:2]}/{out_parts[0][-2:]}",
            out_enddate=f"{year}/{out_parts[-1][:2]}/{out_parts[-1][-2:]}",
        )

    def __getitem__(self, idx: int) -> Tuple[SampleInfo, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        filepath: pathlib.Path = self.filepaths[idx]
        filename: str = filepath.name
        sampleinfo: SampleInfo = CESM2._get_sample_info(filename=filename)
        input_indices, output_indices, input_tensor, output_tensor = torch.load(
            filepath, weights_only=True, map_location=self.metadata.device
        )
        # validate before return
        assert input_indices.shape == (self.metadata.n_input_days,)
        assert output_indices.shape == (self.metadata.n_output_days,)
        assert input_tensor.shape == (self.metadata.n_input_days, 192, 288, len(self.metadata.input_vars))
        assert output_tensor.shape == (1, 192, 288, len(self.metadata.output_vars))
        return sampleinfo, input_indices, output_indices, input_tensor, output_tensor

    def __len__(self) -> int:
        return len(self.filepaths)

    @staticmethod
    def collate_fn(
        batch: List[Tuple[SampleInfo, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
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
    metadata = MetaData(tp="test")
    dataset = CESM2(metadata)

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=CESM2.collate_fn)
    batch = next(iter(dataloader))


