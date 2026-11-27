import argparse
from copy import copy

import torch
from datapipeline.container import VariableContainer
from datapipeline.preprocessing import Detrender, FunctionalTransform, PrecipitationMask, SeasonalCycle, Standardization
from datapipeline.readers.cesm2 import DataReader as CESM2_DataReader
from datapipeline.readers.era5 import DataReader as ERA5_DataReader
from datapipeline.writer import DataWriter
from common.configs import DatasetName, MetaData, Split


class TensorExporter:

    SPLITS: tuple[Split, ...] = ("train", "val", "test")

    def __init__(self, dataset_name: DatasetName, fresh: bool) -> None:
        self.dataset_name: DatasetName = dataset_name
        self.fresh: bool = fresh
        self.metadata_by_split: dict[Split, MetaData] = {
            split: MetaData(dataset_name=dataset_name, split=split)
            for split in self.SPLITS
        }
        self.train_metadata: MetaData = self.metadata_by_split["train"]
        self.full_metadata: MetaData = self._build_full_metadata()
        self.functional_transform = FunctionalTransform(metadata=self.train_metadata)
        self.precipitation_mask: PrecipitationMask = PrecipitationMask(self.train_metadata)

    def _build_full_metadata(self) -> MetaData:
        """Combine the split timelines for fitting rolling trend lines."""
        ordered: list[MetaData] = [
            self.metadata_by_split[split] for split in self.SPLITS
        ]
        reference: MetaData = ordered[0]
        for metadata in ordered[1:]:
            reference.validate_split_settings(metadata)

        years: list[int] = [
            year for metadata in ordered for year in metadata.years
        ]
        expected_years: list[int] = list(range(years[0], years[-1] + 1))
        if years != expected_years:
            raise ValueError(
                "Train/validation/test years must be consecutive and non-overlapping"
            )
        if years[0] != reference.detrend_start_year or years[-1] != reference.detrend_end_year:
            raise ValueError(
                "Configured detrending range does not cover the full dataset"
            )

        full_metadata: MetaData = copy(reference)
        full_metadata.start_year = years[0]
        full_metadata.end_year = years[-1]
        full_metadata.years = years
        full_metadata.n_years = len(years)
        full_metadata.combinations = [
            (sim_id, year)
            for sim_id in full_metadata.sim_ids
            for year in years
        ]
        return full_metadata

    @staticmethod
    def _get_context_metadata(metadata: MetaData) -> MetaData | None:
        """Build metadata for the year preceding validation or test."""
        if metadata.split == "train":
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

    def _read_variable_year(self, var_name: str, sim_id: str, year: int) -> torch.Tensor:
        """Read one raw calendar year in the ERA5 convention."""
        if self.dataset_name == "era5":
            return ERA5_DataReader().get_tensor(var_name=var_name, year=year)
        return CESM2_DataReader(var_name=var_name, sim_id=sim_id, year=year).tensor

    def _load_transformed_year(self, var_name: str, sim_id: str, year: int) -> torch.Tensor:
        """Load daily values and apply their configured function before detrending."""
        raw_tensor: torch.Tensor = self._read_variable_year(var_name=var_name, sim_id=sim_id, year=year)
        return self.functional_transform.transform(raw_tensor, var_name=var_name)

    def _load_variable(self, var_name: str, metadata: MetaData) -> VariableContainer:
        """Load all daily tensors for one variable and metadata period."""
        assert metadata.dataset_name == self.dataset_name
        container = VariableContainer(var_name=var_name, metadata=metadata)
        for sim_id, year in metadata.combinations:
            value: torch.Tensor = self._load_transformed_year(var_name=var_name, sim_id=sim_id, year=year)
            assert value.shape == (365, *metadata.resolution)
            container.set(sim_id=sim_id, year=year, value=value)
            print(
                f"Loaded to var_container: "
                f"{metadata.dataset_name}.{metadata.split}.{sim_id}.{var_name}.{year}"
            )
        return container

    def _load_yearly_means(
        self, var_name: str, metadata: MetaData,
    ) -> VariableContainer:
        """Compute annual means of the functionally transformed daily values."""
        assert metadata.dataset_name == self.dataset_name
        mean_container: VariableContainer = VariableContainer(var_name=var_name, metadata=metadata)

        def read_and_record(sim_id: str, year: int) -> torch.Tensor:
            raw: torch.Tensor = self._read_variable_year(var_name, sim_id, year)
            assert raw.shape == (365, *metadata.resolution)
            value: torch.Tensor = self.functional_transform.transform(raw, var_name=var_name)
            mean_container.set(sim_id=sim_id, year=year, value=value.mean(dim=0, keepdim=True))
            print(f"Loaded yearly mean: {self.dataset_name}.{sim_id}.{var_name}.{year}")
            return raw

        recorded_years: set[int] = set()
        if var_name == self.precipitation_mask.var_name:
            self.precipitation_mask.fit(read_and_record)
            self.precipitation_mask.save_state()
            recorded_years.update(self.train_metadata.years)
        for sim_id, year in metadata.combinations:
            if year not in recorded_years:
                read_and_record(sim_id, year)
        return mean_container

    def _fit_detrenders(self) -> None:
        """Fit each variable over the complete train/validation/test period."""
        for var_name in self.full_metadata.var_names:
            mean_container: VariableContainer = self._load_yearly_means(
                var_name=var_name, metadata=self.full_metadata
            )
            Detrender(metadata=self.full_metadata).fit(mean_container=mean_container)
            print(f"Saved rolling detrending state: {self.dataset_name}.{var_name}")
            del mean_container

    def _export_split(self, split: Split) -> None:
        """Preprocess and write every variable in one dataset split."""
        metadata: MetaData = self.metadata_by_split[split]
        context_metadata: MetaData | None = self._get_context_metadata(metadata)
        detrender = Detrender(metadata=metadata)
        seasonal_cycle = SeasonalCycle(metadata=metadata)
        standardization: Standardization = Standardization(metadata=metadata)
        if split != "train":
            seasonal_cycle.load_state(train_metadata=self.train_metadata)
            standardization.load_state(train_metadata=self.train_metadata)

        with DataWriter(metadata=metadata, fresh=self.fresh) as writer:
            for var_name in metadata.var_names:
                var_container: VariableContainer = self._load_variable(var_name=var_name, metadata=metadata)
                context_container: VariableContainer | None = (
                    self._load_variable(var_name=var_name, metadata=context_metadata)
                    if context_metadata is not None
                    else None
                )
                print(f"Fully loaded {self.dataset_name}.{split}.{var_name} to var_container")
                detrender.transform(var_container, context_container=context_container)
                print(f"Detrended {self.dataset_name}.{split}.{var_name}")
                if split == "train":
                    seasonal_cycle.fit(var_container, train_metadata=self.train_metadata)
                seasonal_cycle.transform(input_container=var_container, context_container=context_container)
                print(f"Removed seasonal cycle from {self.dataset_name}.{split}.{var_name}")
                if split == "train":
                    standardization.fit(var_container, train_metadata=self.train_metadata)
                standardization.transform(input_container=var_container, context_container=context_container)
                if var_name == self.precipitation_mask.var_name:
                    self.precipitation_mask.transform(var_container, context_container=context_container)
                print(f"Standardized anomalies for {self.dataset_name}.{split}.{var_name}")
                writer(var_container, context_container=context_container)
                print(f"Saved all .pt files {self.dataset_name}.{split}.{var_name}")
                del var_container, context_container
            if split == "train":
                seasonal_cycle.save_state(train_metadata=self.train_metadata)
                standardization.save_state(train_metadata=self.train_metadata)

    def run(self) -> None:
        if not self.fresh:
            for metadata in self.metadata_by_split.values():
                DataWriter.resume_counter(metadata)
        self._fit_detrenders()
        for split in self.SPLITS:
            self._export_split(split)


def main() -> None:
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

    dataset: DatasetName = args.dataset
    TensorExporter(dataset_name=dataset, fresh=args.fresh).run()


if __name__ == "__main__":
    main()
