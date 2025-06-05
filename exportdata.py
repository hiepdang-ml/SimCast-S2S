import argparse
from typing import List, Dict, Any, Optional
import yaml

from cesm2.container import DataContainer
from cesm2.preprocessing import Detrender, ClimatologyRemover
from cesm2.reader import DataReader
from cesm2.writer import DataWriter
from common.configs import MetaData


def main() -> None:
    """
    Main function to write tensors with a metadata.

    Parameters:
        config (Dict[str, Any]): Configuration dictionary.
    """

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

if __name__ == "__main__":
    main()

