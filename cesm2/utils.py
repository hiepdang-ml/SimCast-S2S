from typing import *
from dataclasses import dataclass

import torch

@dataclass
class SampleInfo:
    sim_id: str
    in_startdate: str
    in_enddate: str
    out_startdate: str
    out_enddate: str

DataBatch = NewType("DataBatch", Tuple[List[SampleInfo], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor])

