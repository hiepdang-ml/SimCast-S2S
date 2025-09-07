import argparse
from typing import *

from datasets.common.container import VariableContainer
from datasets.common.preprocessing import Detrender, ClimatologyRemover
from datasets.cesm2.reader import DataReader as CESM2_DataReader
from datasets.era5.reader import DataReader as ERA5_DataReader
from datasets.common.writer import DataWriter
from common.configs import MetaData


def main(dataset: Literal["cesm2", "era5"]) -> None:
    
    # train dataset
    train_metadata: MetaData = MetaData(dataset_name=dataset, tp="train")
    detrender: Detrender = Detrender(metadata=train_metadata)
    climatology_remover: ClimatologyRemover = ClimatologyRemover(metadata=train_metadata)
    writer: DataWriter = DataWriter(metadata=train_metadata)

    if dataset == "cesm2":
        for var_name in train_metadata.var_names:
            var_container: VariableContainer = VariableContainer(var_name=var_name, metadata=train_metadata)
            for sim_id, year in train_metadata.combinations:
                reader: CESM2_DataReader = CESM2_DataReader(
                    var_name=var_name, sim_id=sim_id, year=year, device=train_metadata.device,
                )
                var_container.set(sim_id=sim_id, year=year, value=reader.tensor)
                print(f"Loaded to var_container: {dataset}.train.{sim_id}.{var_name}.{year}")

            print(f"Fully loaded {dataset}.train.{var_name} to var_container")
            detrender(input_container=var_container)
            print(f"Detrened {dataset}.train.{var_name}")
            climatology_remover(input_container=var_container)
            print(f"Climatology removed {dataset}.train.{var_name}")
            writer(var_container=var_container)
            print(f"Saved all .pt files {dataset}.train.{var_name}")
            del var_container   # release memory

    else:
        assert len(train_metadata.sim_ids) == 1
        sim_id: str = train_metadata.sim_ids[0]
        assert sim_id == "reanalysis"   # only one "simulation"
        for var_name in train_metadata.var_names:
            var_container: VariableContainer = VariableContainer(var_name=var_name, metadata=train_metadata)
            reader: ERA5_DataReader = ERA5_DataReader(
                resolution=train_metadata.resolution, device=train_metadata.device,
            )
            for sim_id, year in train_metadata.combinations:
                var_container.set(
                    sim_id=sim_id, year=year, 
                    value=reader.get_tensor(var_name=var_name, year=year),
                )
                print(f"Loaded to var_container: {dataset}.train.{sim_id}.{var_name}.{year}")

            print(f"Fully loaded {dataset}.train.{var_name} to var_container")
            detrender(input_container=var_container)
            print(f"Detrened {dataset}.train.{var_name}")
            climatology_remover(input_container=var_container)
            print(f"Climatology removed {dataset}.train.{var_name}")
            writer(var_container=var_container)
            print(f"Saved all .pt files {dataset}.train.{var_name}")
            del var_container   # release memory

    # val & test dataset
    for tp in ["val", "test"]:
        metadata: MetaData = MetaData(dataset_name=dataset, tp=tp)
        detrender: Detrender = Detrender(metadata=metadata)
        climatology_remover: ClimatologyRemover = ClimatologyRemover(metadata=metadata)
        writer: DataWriter = DataWriter(metadata=metadata)

        if dataset == "cesm2":
            for var_name in metadata.var_names:
                var_container: VariableContainer = VariableContainer(var_name=var_name, metadata=metadata)
                for sim_id, year in metadata.combinations:
                    reader: CESM2_DataReader = CESM2_DataReader(
                        var_name=var_name, sim_id=sim_id, year=year, device=metadata.device
                    )
                    var_container.set(sim_id=sim_id, year=year, value=reader.tensor)
                    print(f"Loaded to var_container: {dataset}.{tp}.{sim_id}.{var_name}.{year}")

                print(f"Fully loaded {dataset}.{tp}.{var_name} to var_container")
                detrender(var_container, train_metadata=train_metadata)
                print(f"Detrened {dataset}.{tp}.{var_name}")
                climatology_remover(var_container, train_metadata=train_metadata)
                print(f"Climatology removed {dataset}.{tp}.{var_name}")
                writer(var_container)
                print(f"Saved all .pt files {dataset}.{tp}.{var_name}")
                del var_container   # release memory

        else:
            for var_name in metadata.var_names:
                var_container: VariableContainer = VariableContainer(var_name=var_name, metadata=metadata)
                for sim_id, year in metadata.combinations:
                    assert sim_id == "reanalysis"   # only one "simulation"
                    reader: ERA5_DataReader = ERA5_DataReader(
                        resolution=metadata.resolution, device=metadata.device
                    )
                    var_container.set(
                        sim_id=sim_id, year=year, 
                        value=reader.get_tensor(var_name=var_name, year=year),
                    )
                    print(f"Loaded to var_container: {dataset}.{tp}.{sim_id}.{var_name}.{year}")

                print(f"Fully loaded {dataset}.{tp}.{var_name} to var_container")
                detrender(var_container, train_metadata=train_metadata)
                print(f"Detrened {dataset}.{tp}.{var_name}")
                climatology_remover(var_container, train_metadata=train_metadata)
                print(f"Climatology removed {dataset}.{tp}.{var_name}")
                writer(var_container)
                print(f"Saved all .pt files {dataset}.{tp}.{var_name}")
                del var_container   # release memory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["cesm2", "era5"], required=True)
    args: argparse.Namespace = parser.parse_args()
    main(dataset=args.dataset)


