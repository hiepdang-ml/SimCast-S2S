import json
import pathlib
from functools import cached_property
from typing import *

import torch
from torch.utils.data import Dataset

from cesm2.reader import DataReader
from cesm2.common import MetaData


class CESM2(Dataset):

    def __init__(self, metadata: MetaData):
        super().__init__()
        self.metadata: MetaData = metadata
        # validate consistent write/read metadata
        with open(self.metadata.saved_metadata_path, mode="r") as file:
            if self.metadata.to_dict() != json.load(fp=file):
                raise RuntimeError("Written .pt files do not match new config, need to rewrite")

    @cached_property
    def filepaths(self) -> List[pathlib.Path]:
        return sorted(self.metadata.write_directory.glob("*.pt"))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_indices, output_indices, input_tensor, output_tensor = torch.load(self.filepaths[idx], weights_only=True)
        # validate before return
        assert input_indices.shape == (self.metadata.n_input_days,)
        assert output_indices.shape == (self.metadata.n_output_days,)
        assert input_tensor.shape == (self.metadata.n_input_days, 192, 288, len(self.metadata.input_vars))
        if self.metadata.need_daily_predictions:
            assert output_tensor.shape == (self.metadata.n_output_days, 192, 288, len(self.metadata.output_vars))
        else:
            assert output_tensor.shape == (192, 288, len(self.metadata.output_vars))
        return input_indices, output_indices, input_tensor, output_tensor

    def __len__(self) -> int:
        return len(self.filepaths)
    


if __name__ == "__main__":

    from cesm2.common import DataContainer
    from cesm2.preprocessing import Detrender, ClimatologyRemover
    from cesm2.writer import DataWriter

    # test
    metadata: MetaData = MetaData()
    input = DataContainer(metadata=metadata)
    for var_name, sim_id, year in metadata.combinations:
        input.set(
            var_name=var_name, sim_id=sim_id, year=year, 
            value=DataReader(var_name=var_name, sim_id=sim_id, year=year).tensor
        )

    detrender = Detrender(metadata=metadata)
    climatology_remover: ClimatologyRemover = ClimatologyRemover(metadata=metadata, window_size=15)
    writer = DataWriter(metadata=metadata)
    writer(climatology_remover(detrender(input)))

    cesm2 = CESM2(metadata)
    dataloader = torch.utils.data.DataLoader(dataset=cesm2, batch_size=32, shuffle=True)
    input_indices, output_indices, input_tensor, output_tensor = next(iter(dataloader))
