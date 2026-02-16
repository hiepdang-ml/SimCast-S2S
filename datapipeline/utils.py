from typing import NewType
from functools import cached_property
import datetime as dt
import torch


class SampleInfo:

    def __init__(
        self,
        sim_id: str,
        in_startdate: str, in_enddate: str,
        out_startdate: str, out_enddate: str
    ):
        self.sim_id: str = sim_id
        self.in_startdate: str = in_startdate
        self.in_enddate: str = in_enddate
        self.out_startdate: str = out_startdate
        self.out_enddate: str = out_enddate

    @cached_property
    def in_dates(self) -> list[str]:
        return SampleInfo.find_date_list(self.in_startdate, self.in_enddate)

    @cached_property
    def out_dates(self) -> list[str]:
        return SampleInfo.find_date_list(self.out_startdate, self.out_enddate)

    @staticmethod
    def find_date_list(startdate: str, enddate: str) -> list[str]:
        in_startdate: dt.datetime = dt.datetime.strptime(startdate, "%Y/%m/%d")
        in_enddate: dt.datetime = dt.datetime.strptime(enddate, "%Y/%m/%d")
        daterange: list[dt.datetime] = [in_startdate]
        while daterange[-1] < in_enddate:
            d: dt.datetime = daterange[-1] + dt.timedelta(days=1)
            daterange.append(d)

        return [d.strftime("%Y/%m/%d") for d in daterange if not (d.month == 2 and d.day == 29)]


DataBatch = NewType("DataBatch", tuple[list[SampleInfo], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor])
