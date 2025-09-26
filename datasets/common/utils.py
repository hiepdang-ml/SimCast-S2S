from typing import NewType
from dataclasses import dataclass
from functools import cache

import torch

@dataclass
class SampleInfo:
    sim_id: str
    in_startdate: str
    in_enddate: str
    out_startdate: str
    out_enddate: str

DataBatch = NewType("DataBatch", tuple[list[SampleInfo], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor])


