import argparse
from typing import Literal

from datapipeline.container import VariableContainer
from datapipeline.preprocessing import Detrender, ClimatologyRemover
from datapipeline.readers.cesm2 import DataReader as CESM2_DataReader
from datapipeline.readers.era5 import DataReader as ERA5_DataReader
from datapipeline.writer import DataWriter
from common.configs import MetaData


def main(dataset: Literal["cesm2", "era5"], fresh: bool) -> None:

    # train dataset
    train_metadata: MetaData = MetaData(dataset_name=dataset, tp="train")
    detrender: Detrender = Detrender(metadata=train_metadata)
    climatology_remover: ClimatologyRemover = ClimatologyRemover(metadata=train_metadata)
    with DataWriter(metadata=train_metadata, fresh=fresh) as writer:
        if dataset == "cesm2":
            for var_name in train_metadata.var_names:
                var_container: VariableContainer = VariableContainer(var_name=var_name, metadata=train_metadata)
                for sim_id, year in train_metadata.combinations:
                    reader: CESM2_DataReader = CESM2_DataReader(var_name=var_name, sim_id=sim_id, year=year)
                    var_container.set(sim_id=sim_id, year=year, value=reader.tensor)
                    print(f"Loaded to var_container: {dataset}.train.{sim_id}.{var_name}.{year}")

                print(f"Fully loaded {dataset}.train.{var_name} to var_container")
                detrender(input_container=var_container)
                print(f"Detrended {dataset}.train.{var_name}")
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
                reader: ERA5_DataReader = ERA5_DataReader(target_resolution=train_metadata.resolution)
                for sim_id, year in train_metadata.combinations:
                    var_container.set(
                        sim_id=sim_id, year=year,
                        value=reader.get_tensor(var_name=var_name, year=year),
                    )
                    print(f"Loaded to var_container: {dataset}.train.{sim_id}.{var_name}.{year}")

                print(f"Fully loaded {dataset}.train.{var_name} to var_container")
                detrender(input_container=var_container)
                print(f"Detrended {dataset}.train.{var_name}")
                climatology_remover(input_container=var_container)
                print(f"Climatology removed {dataset}.train.{var_name}")
                writer(var_container=var_container)
                print(f"Saved all .pt files {dataset}.train.{var_name}")
                del var_container   # release memory

    # val & test dataset
    for tp in ["val", "test"]:
        tp: Literal["val", "test"]
        metadata: MetaData = MetaData(dataset_name=dataset, tp=tp)
        detrender: Detrender = Detrender(metadata=metadata)
        climatology_remover: ClimatologyRemover = ClimatologyRemover(metadata=metadata)
        with DataWriter(metadata=metadata, fresh=fresh) as writer:
            if dataset == "cesm2":
                for var_name in metadata.var_names:
                    var_container: VariableContainer = VariableContainer(var_name=var_name, metadata=metadata)
                    for sim_id, year in metadata.combinations:
                        reader: CESM2_DataReader = CESM2_DataReader(var_name=var_name, sim_id=sim_id, year=year)
                        var_container.set(sim_id=sim_id, year=year, value=reader.tensor)
                        print(f"Loaded to var_container: {dataset}.{tp}.{sim_id}.{var_name}.{year}")

                    print(f"Fully loaded {dataset}.{tp}.{var_name} to var_container")
                    detrender(var_container, train_metadata=train_metadata)
                    print(f"Detrended {dataset}.{tp}.{var_name}")
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
                        reader: ERA5_DataReader = ERA5_DataReader(target_resolution=metadata.resolution)
                        var_container.set(
                            sim_id=sim_id, year=year,
                            value=reader.get_tensor(var_name=var_name, year=year),
                        )
                        print(f"Loaded to var_container: {dataset}.{tp}.{sim_id}.{var_name}.{year}")

                    print(f"Fully loaded {dataset}.{tp}.{var_name} to var_container")
                    detrender(var_container, train_metadata=train_metadata)
                    print(f"Detrended {dataset}.{tp}.{var_name}")
                    climatology_remover(var_container, train_metadata=train_metadata)
                    print(f"Climatology removed {dataset}.{tp}.{var_name}")
                    writer(var_container)
                    print(f"Saved all .pt files {dataset}.{tp}.{var_name}")
                    del var_container   # release memory


if __name__ == "__main__":

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    group: argparse._MutuallyExclusiveGroup = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("--dataset", type=str, choices=["cesm2", "era5"], required=True)
    group.add_argument(
        "--fresh",
        action="store_true",
        dest="fresh",
        help="Write from scratch (reset counter)",
    )
    group.add_argument(
        "--resume",
        action="store_true",
        dest="resume",
        help="Resume from last write (resume counter)",
    )
    args: argparse.Namespace = parser.parse_args()

    if args.fresh:
        main(dataset=args.dataset, fresh=True)
    elif args.resume:
        main(dataset=args.dataset, fresh=False)
