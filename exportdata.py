import argparse
from typing import *

from datasets.common.container import DataContainer
from datasets.common.preprocessing import Detrender, ClimatologyRemover
from datasets.cesm2.reader import DataReader
from datasets.cesm2.writer import DataWriter
from common.configs import MetaData


def export_cesm2() -> None:

    # train dataset
    train_metadata: MetaData = MetaData(tp="train")
    container: DataContainer = DataContainer(metadata=train_metadata)
    for sim_id, var_name, year in train_metadata.combinations:
        container.set(
            sim_id=sim_id, var_name=var_name, year=year, 
            value=DataReader(sim_id=sim_id, var_name=var_name, year=year, device=train_metadata.device).tensor
        )
    detrender: Detrender = Detrender(metadata=train_metadata)
    climatology_remover: ClimatologyRemover = ClimatologyRemover(metadata=train_metadata)
    writer: DataWriter = DataWriter(metadata=train_metadata)
    detrender(container)
    climatology_remover(container)
    writer(container)
    del container

    # val & test dataset
    for tp in ["val", "test"]:
        metadata: MetaData = MetaData(tp=tp)
        container: DataContainer = DataContainer(metadata=metadata)
        for sim_id, var_name, year in metadata.combinations:
            container.set(
                sim_id=sim_id, var_name=var_name, year=year, 
                value=DataReader(sim_id=sim_id, var_name=var_name, year=year, device=metadata.device).tensor
            )
        detrender: Detrender = Detrender(metadata=metadata)
        climatology_remover: ClimatologyRemover = ClimatologyRemover(metadata=metadata)
        writer: DataWriter = DataWriter(metadata=metadata)
        detrender(container, train_metadata=train_metadata)
        climatology_remover(container, train_metadata=train_metadata)
        writer(container)
        del container


# TODO: implement
def export_era5() -> None:
    pass


def main(dataset: Literal["cesm2", "era5"]):
    export_cesm2() if dataset == "cesm2" else export_era5()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["cesm2", "era5"], required=True)
    args: argparse.Namespace = parser.parse_args()
    main(dataset=args.dataset)


