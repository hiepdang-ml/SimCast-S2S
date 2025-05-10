import argparse
from typing import List, Dict, Any, Optional
import yaml

from cesm2.common import DataContainer, MetaData
from cesm2.preprocessing import Detrender, ClimatologyRemover
from cesm2.reader import DataReader
from cesm2.writer import DataWriter


def main() -> None:
    """
    Main function to write tensors with a metadata.

    Parameters:
        config (Dict[str, Any]): Configuration dictionary.
    """

    # train dataset
    train_metadata: MetaData = MetaData(tp="train")
    container: DataContainer = DataContainer(metadata=train_metadata)
    for var_name, sim_id, year in train_metadata.combinations:
        container.set(
            var_name=var_name, sim_id=sim_id, year=year, 
            value=DataReader(var_name=var_name, sim_id=sim_id, year=year, device=train_metadata.device).tensor
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
        for var_name, sim_id, year in metadata.combinations:
            container.set(
                var_name=var_name, sim_id=sim_id, year=year, 
                value=DataReader(var_name=var_name, sim_id=sim_id, year=year, device=metadata.device).tensor
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

