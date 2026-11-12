import argparse
from copy import copy
from typing import Literal, cast

from datapipeline.container import VariableContainer
from datapipeline.preprocessing import Detrender, ClimatologyRemover
from datapipeline.readers.cesm2 import DataReader as CESM2_DataReader
from datapipeline.readers.era5 import DataReader as ERA5_DataReader
from datapipeline.writer import DataWriter
from common.configs import MetaData


def _get_context_metadata(metadata: MetaData) -> MetaData | None:
    """Only validation/test need history from outside their split."""
    if metadata.tp == "train":
        # no additional year needs loading outside the training split.
        return None

    year: int = metadata.start_year - 1
    context: MetaData = copy(metadata)
    context.start_year = year
    context.end_year = year
    context.years = [year]
    context.n_years = 1
    context.sim_ids = metadata.sim_ids.copy()
    context.combinations = [
        (sim_id, year) for sim_id in context.sim_ids
    ]
    return context


def _load_variable(var_name: str, metadata: MetaData) -> VariableContainer:
    container: VariableContainer = VariableContainer(var_name=var_name, metadata=metadata)
    era5_reader: ERA5_DataReader | None = ERA5_DataReader() if metadata.dataset_name == "era5" else None
    for sim_id, year in metadata.combinations:
        if metadata.dataset_name == "era5":
            assert sim_id == "reanalysis"
            value = cast(ERA5_DataReader, era5_reader).get_tensor(var_name=var_name, year=year)
        else:
            value = CESM2_DataReader(var_name=var_name, sim_id=sim_id, year=year).tensor
        assert value.shape == (365, *metadata.resolution)
        container.set(sim_id=sim_id, year=year, value=value)
        print(f"Loaded to var_container: {metadata.dataset_name}.{metadata.tp}.{sim_id}.{var_name}.{year}")
    return container


def export_tensors(dataset: Literal["cesm2", "era5"], fresh: bool) -> None:
    train_metadata = MetaData(dataset_name=dataset, tp="train")
    types: tuple[Literal["train", "val", "test"], ...] = ("train", "val", "test")
    metadata: MetaData
    fit_metadata: MetaData | None
    for tp in types:
        if tp == "train":
            metadata = train_metadata
            fit_metadata = None
        else:
            metadata = MetaData(dataset_name=dataset, tp=tp)
            fit_metadata = train_metadata

        context_metadata: MetaData | None = _get_context_metadata(metadata)
        detrender: Detrender = Detrender(metadata=metadata)
        climatology_remover: ClimatologyRemover = ClimatologyRemover(metadata=metadata)
        with DataWriter(metadata=metadata, fresh=fresh) as writer:
            for var_name in metadata.var_names:
                var_container = _load_variable(var_name=var_name, metadata=metadata)
                context_container = _load_variable(var_name=var_name, metadata=context_metadata) if context_metadata is not None else None
                print(f"Fully loaded {dataset}.{tp}.{var_name} to var_container")
                detrender(var_container, train_metadata=fit_metadata, context_container=context_container)
                print(f"Detrended {dataset}.{tp}.{var_name}")
                climatology_remover(var_container, train_metadata=fit_metadata, context_container=context_container)
                print(f"Climatology removed {dataset}.{tp}.{var_name}")
                writer(var_container, context_container=context_container)
                print(f"Saved all .pt files {dataset}.{tp}.{var_name}")
                del var_container, context_container   # release memory


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["cesm2", "era5"], required=True)
    group: argparse._MutuallyExclusiveGroup = parser.add_mutually_exclusive_group(required=True)
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
        export_tensors(dataset=args.dataset, fresh=True)
    elif args.resume:
        export_tensors(dataset=args.dataset, fresh=False)


if __name__ == "__main__":
    main()
