import argparse
from typing import *

from datasets.common.container import DataContainer
from datasets.common.preprocessing import Detrender, ClimatologyRemover
from datasets.cesm2.reader import DataReader as CESM2_DataReader
from datasets.era5.reader import DataReader as ERA5_DataReader
from datasets.common.writer import DataWriter
from common.configs import MetaData


def main(dataset: Literal["cesm2", "era5"]) -> None:
    
    # train dataset
    train_metadata: MetaData = MetaData(dataset_name=dataset, tp="train")
    container: DataContainer = DataContainer(metadata=train_metadata)
    if dataset == "cesm2":
        # require H200
        for sim_id, var_name, year in train_metadata.combinations:
            reader: CESM2_DataReader = CESM2_DataReader(
                sim_id=sim_id, var_name=var_name, year=year, device=train_metadata.device,
            )
            container.set(sim_id=sim_id, var_name=var_name, year=year, value=reader.tensor)
            print(f"Loaded to container: {(sim_id, var_name, year)}")
    else:
        assert len(train_metadata.sim_ids) == 1
        sim_id: str = train_metadata.sim_ids[0]
        assert sim_id == "reanalysis"   # only one "simulation"
        # require H200
        for year in train_metadata.years:
            reader: ERA5_DataReader = ERA5_DataReader(
                year=year, resolution=train_metadata.resolution, device=train_metadata.device,
            )
            for var_name in train_metadata.var_names:
                container.set(
                    sim_id=sim_id, var_name=var_name, year=year, 
                    value=reader.get_tensor(var_name=var_name),
                )
                print(f"Loaded to container: {(sim_id, var_name, year)}")

    detrender: Detrender = Detrender(metadata=train_metadata)
    climatology_remover: ClimatologyRemover = ClimatologyRemover(metadata=train_metadata)
    writer: DataWriter = DataWriter(metadata=train_metadata)
    detrender(container)
    climatology_remover(container)
    writer(container)
    del container

    # val & test dataset
    for tp in ["val", "test"]:
        metadata: MetaData = MetaData(dataset_name=dataset, tp=tp)
        container: DataContainer = DataContainer(metadata=metadata)
        if dataset == "cesm2":
            for sim_id, var_name, year in metadata.combinations:
                reader: CESM2_DataReader = CESM2_DataReader(
                    sim_id=sim_id, var_name=var_name, year=year, device=metadata.device
                )
                container.set(sim_id=sim_id, var_name=var_name, year=year, value=reader.tensor)
        else:
            for sim_id, var_name, year in metadata.combinations:
                assert sim_id == "reanalysis"   # only one "simulation"
                reader: ERA5_DataReader = ERA5_DataReader(
                    year=year, resolution=metadata.resolution, device=metadata.device
                )
                container.set(
                    sim_id=sim_id, var_name=var_name, year=year, 
                    value=reader.get_tensor(var_name=var_name),
                )
        detrender: Detrender = Detrender(metadata=metadata)
        climatology_remover: ClimatologyRemover = ClimatologyRemover(metadata=metadata)
        writer: DataWriter = DataWriter(metadata=metadata)
        detrender(container, train_metadata=train_metadata)
        climatology_remover(container, train_metadata=train_metadata)
        writer(container)
        del container


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["cesm2", "era5"], required=True)
    args: argparse.Namespace = parser.parse_args()
    main(dataset=args.dataset)


