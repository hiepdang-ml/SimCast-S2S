from pathlib import Path
from typing import Any, Literal

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import torch
from common.configs import MetaData
from datapipeline.readers import ERA5_CoordinatesReader, ERA5_LandmaskReader
from matplotlib.colors import LinearSegmentedColormap


SIMCAST_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/finetune/diffusion_v23_cosine_eta100_100steps_100members_guidancescale200")
ECMWF_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/ecmwfs2s_28/")
TARGET_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/")


class ProbabilitySkillScoreMapFigureBuilder:

    MAX_GROUPS: int = 70
    CATEGORY_PERCENTILES: tuple[float, float] = (33.3333333333, 66.6666666667)
    UPPER_EVENT_PERCENTILE: float = 90.0
    DATA_CMAP = LinearSegmentedColormap.from_list(
        "reference_soft_blue_orange_red",
        [
            (0.00, "#2f5597"),
            (0.08, "#5379bf"),
            (0.22, "#9bb7e5"),
            (0.38, "#d6e3f6"),
            (0.50, "#f7f4ee"),
            (0.62, "#f5d8c2"),
            (0.78, "#e26f55"),
            (0.92, "#bd252c"),
            (1.00, "#7f0d21"),
        ],
    )
    DATA_CMAP.set_bad(color="#ffffff", alpha=0.0)
    DATA_VMIN: float = -1
    DATA_VMAX: float = 1
    DIFF_VMIN: float = -1
    DIFF_VMAX: float = 1
    MIN_GROUNDTRUTH_VARIANCE: float = 1e-6
    BOOTSTRAP_SAMPLES: int = 1000
    BOOTSTRAP_CONFIDENCE: float = 0.975
    BOOTSTRAP_SEED: int = 7341
    STIPPLE_STRIDE: int = 4
    MIN_EVENT_COUNT: int = 10
    MIN_NON_EVENT_COUNT: int = 10
    TITLE_FONT_SIZE: int = 20
    ROW_LABEL_FONT_SIZE: int = 20
    COLORBAR_LABEL_FONT_SIZE: int = 15
    COLORBAR_TICK_FONT_SIZE: int = 15

    def __init__(self, simcast_root: Path, ecmwf_root: Path, target_dir: Path) -> None:
        if not simcast_root.exists():
            raise FileNotFoundError(f"SimCast-S2S root directory does not exist: {simcast_root}")
        if not ecmwf_root.exists():
            raise FileNotFoundError(f"ECMWF-S2S root directory does not exist: {ecmwf_root}")
        self.dataset: Literal["cesm2", "era5"] = "era5"
        self.simcast_root: Path = simcast_root
        self.ecmwf_root: Path = ecmwf_root
        self.target_dir: Path = target_dir
        self.output_path: Path = self.target_dir.joinpath("fig7f.png")

    @staticmethod
    def _sample_key(payload: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(payload["output_name"]),
            str(payload["out_startdate"]),
            str(payload["out_enddate"]),
        )

    def _load_member_payloads(self, root: Path) -> list[dict[str, Any]]:
        tensor_dir: Path = root.joinpath(self.dataset, "tensors")
        if not tensor_dir.exists():
            raise FileNotFoundError(f"Missing tensor directory: {tensor_dir}")

        payloads: list[dict[str, Any]] = []
        for path in sorted(tensor_dir.glob("*_ens_*.pt")):
            if path.name.endswith("_ens_aggregate.pt"):
                continue
            payloads.append(torch.load(path, map_location="cpu"))
        if not payloads:
            raise FileNotFoundError(f"No ensemble-member tensor files found in: {tensor_dir}")
        return payloads

    def _load_grouped_samples(self, root: Path) -> dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]]:
        payloads = self._load_member_payloads(root=root)
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for payload in payloads:
            grouped.setdefault(self._sample_key(payload), []).append(payload)

        samples: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for key, sample_payloads in grouped.items():
            members = sorted(sample_payloads, key=lambda item: int(item["ensemble_member"]))
            prediction_tensor = torch.stack(
                [torch.as_tensor(item["prediction"], dtype=torch.float32) for item in members],
                dim=0,
            )
            groundtruth = torch.as_tensor(members[0]["groundtruth"], dtype=torch.float32)
            assert prediction_tensor.ndim == 3
            assert groundtruth.shape == prediction_tensor.shape[1:]
            samples[key] = (prediction_tensor, groundtruth)

        if not samples:
            raise ValueError(f"No grouped samples found under {root}")
        return samples

    def _load_all_samples(
        self,
        root: Path,
        label: str,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        samples = self._load_grouped_samples(root=root)
        selected_keys = sorted(samples)[: self.MAX_GROUPS]
        if not selected_keys:
            raise ValueError(f"No {label} samples found")
        print(f"[fig7f] {label} samples: {len(selected_keys)}")

        predictions: list[torch.Tensor] = []
        groundtruths: list[torch.Tensor] = []
        for key in selected_keys:
            prediction, groundtruth = samples[key]
            predictions.append(prediction)
            groundtruths.append(groundtruth)
        return predictions, torch.stack(groundtruths, dim=0)

    def _load_common_samples(
        self,
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor], torch.Tensor]:
        simcast_samples = self._load_grouped_samples(root=self.simcast_root)
        ecmwf_samples = self._load_grouped_samples(root=self.ecmwf_root)
        common_keys = sorted(set(simcast_samples) & set(ecmwf_samples))[: self.MAX_GROUPS]
        if not common_keys:
            raise ValueError("No common SimCast-S2S/ECMWF-S2S samples found by output_name/out_startdate/out_enddate")
        print(f"[fig7f] Common samples for RPSS/CRPSS: {len(common_keys)}")

        simcast_predictions: list[torch.Tensor] = []
        simcast_groundtruths: list[torch.Tensor] = []
        ecmwf_predictions: list[torch.Tensor] = []
        ecmwf_groundtruths: list[torch.Tensor] = []
        for key in common_keys:
            simcast_prediction, simcast_groundtruth = simcast_samples[key]
            ecmwf_prediction, ecmwf_groundtruth = ecmwf_samples[key]
            assert simcast_groundtruth.shape == ecmwf_groundtruth.shape
            simcast_predictions.append(simcast_prediction)
            simcast_groundtruths.append(simcast_groundtruth)
            ecmwf_predictions.append(ecmwf_prediction)
            ecmwf_groundtruths.append(ecmwf_groundtruth)

        return (
            simcast_predictions,
            torch.stack(simcast_groundtruths, dim=0),
            ecmwf_predictions,
            torch.stack(ecmwf_groundtruths, dim=0),
        )

    @classmethod
    def _percentile_threshold_map(
        cls,
        samples: torch.Tensor,
        percentile: float,
    ) -> torch.Tensor:
        assert samples.ndim == 3
        return torch.quantile(samples, q=percentile / 100.0, dim=0)

    def _load_historical_output_climatology(self) -> torch.Tensor:
        train_metadata = MetaData(dataset_name=self.dataset, tp="train")
        output_name = train_metadata.output_vars[0]
        tensor_dir = train_metadata.write_directory.joinpath("output", output_name)
        if not tensor_dir.exists():
            raise FileNotFoundError(f"Missing historical climatology tensor directory: {tensor_dir}")

        climatology_samples: list[torch.Tensor] = []
        for path in sorted(tensor_dir.glob("*.pt")):
            _, output_tensor = torch.load(path, map_location="cpu", weights_only=False)
            output_tensor = torch.as_tensor(output_tensor, dtype=torch.float32)
            assert output_tensor.shape == (train_metadata.n_output_days, *train_metadata.resolution)
            climatology_samples.append(output_tensor.mean(dim=0))

        if not climatology_samples:
            raise FileNotFoundError(f"No historical climatology output tensors found in: {tensor_dir}")
        climatology = torch.stack(climatology_samples, dim=0)
        print(f"[fig7f] Historical climatology samples: {climatology.shape[0]}")
        return climatology

    @staticmethod
    def _category_index(values: torch.Tensor, lower_threshold: torch.Tensor, upper_threshold: torch.Tensor) -> torch.Tensor:
        categories = torch.zeros_like(values, dtype=torch.long)
        categories = torch.where(values > lower_threshold, torch.ones_like(categories), categories)
        categories = torch.where(values > upper_threshold, torch.full_like(categories, 2), categories)
        return categories

    @classmethod
    def _category_probabilities(
        cls,
        predictions: list[torch.Tensor],
        lower_threshold: torch.Tensor,
        upper_threshold: torch.Tensor,
    ) -> torch.Tensor:
        probabilities: list[torch.Tensor] = []
        for prediction in predictions:
            categories = cls._category_index(
                values=prediction,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
            )
            member_probabilities = torch.stack(
                [(categories == category_idx).to(dtype=torch.float32).mean(dim=0) for category_idx in range(3)],
                dim=-1,
            )
            probabilities.append(member_probabilities)
        return torch.stack(probabilities, dim=0)

    @classmethod
    def _observed_categories(
        cls,
        groundtruths: torch.Tensor,
        lower_threshold: torch.Tensor,
        upper_threshold: torch.Tensor,
    ) -> torch.Tensor:
        categories = cls._category_index(
            values=groundtruths,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
        )
        return torch.stack(
            [(categories == category_idx).to(dtype=torch.float32) for category_idx in range(3)],
            dim=-1,
        )

    @staticmethod
    def _ranked_probability_score(probabilities: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        assert probabilities.shape == observations.shape
        assert probabilities.shape[-1] == 3
        probability_cdf = probabilities.cumsum(dim=-1)[..., :-1]
        observation_cdf = observations.cumsum(dim=-1)[..., :-1]
        return torch.sum((probability_cdf - observation_cdf) ** 2, dim=-1)

    @classmethod
    def _rpss_components(
        cls,
        predictions: list[torch.Tensor],
        groundtruths: torch.Tensor,
        lower_threshold: torch.Tensor,
        upper_threshold: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(predictions) == groundtruths.shape[0]
        forecast_probabilities = cls._category_probabilities(
            predictions=predictions,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
        )
        observations = cls._observed_categories(
            groundtruths=groundtruths,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
        )
        climatology_probabilities = torch.full_like(observations, 1.0 / 3.0)
        rps = cls._ranked_probability_score(probabilities=forecast_probabilities, observations=observations)
        reference_rps = cls._ranked_probability_score(probabilities=climatology_probabilities, observations=observations)
        valid_cells = (
            (groundtruths.var(dim=0, unbiased=False) > cls.MIN_GROUNDTRUTH_VARIANCE)
            & (upper_threshold > lower_threshold)
            & (reference_rps.mean(dim=0) > 0)
        )
        return rps, reference_rps, valid_cells

    @staticmethod
    def _rpss_from_components(rps: torch.Tensor, reference_rps: torch.Tensor, valid_cells: torch.Tensor) -> torch.Tensor:
        mean_rps = rps.mean(dim=0)
        mean_reference_rps = reference_rps.mean(dim=0)
        skill = 1.0 - mean_rps / mean_reference_rps
        return torch.where(valid_cells, skill, torch.full_like(skill, torch.nan))

    @staticmethod
    def _event_indicator(values: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        return (values > threshold).to(dtype=torch.float32)

    @classmethod
    def _bss_components(
        cls,
        predictions: list[torch.Tensor],
        groundtruths: torch.Tensor,
        threshold: torch.Tensor,
        percentile: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(predictions) == groundtruths.shape[0]
        event_probability = torch.stack(
            [cls._event_indicator(prediction, threshold=threshold).mean(dim=0) for prediction in predictions],
            dim=0,
        )
        event_observation = cls._event_indicator(groundtruths, threshold=threshold)
        climatology_probability = 1.0 - percentile / 100.0
        brier_score = (event_probability - event_observation) ** 2
        reference_score = (climatology_probability - event_observation) ** 2
        event_count = event_observation.sum(dim=0)
        non_event_count = event_observation.shape[0] - event_count
        valid_cells = (
            (groundtruths.var(dim=0, unbiased=False) > cls.MIN_GROUNDTRUTH_VARIANCE)
            & (event_count >= cls.MIN_EVENT_COUNT)
            & (non_event_count >= cls.MIN_NON_EVENT_COUNT)
            & (reference_score.mean(dim=0) > 0)
        )
        return brier_score, reference_score, valid_cells

    @staticmethod
    def _bss_from_components(
        score: torch.Tensor,
        reference_score: torch.Tensor,
        valid_cells: torch.Tensor,
    ) -> torch.Tensor:
        mean_score = score.mean(dim=0)
        mean_reference_score = reference_score.mean(dim=0)
        skill = 1.0 - mean_score / mean_reference_score
        return torch.where(valid_cells, skill, torch.zeros_like(skill))

    @staticmethod
    def _pairwise_abs_mean(sorted_predictions: torch.Tensor) -> torch.Tensor:
        n_members = sorted_predictions.shape[0]
        weights = (2 * torch.arange(1, n_members + 1, device=sorted_predictions.device) - n_members - 1).to(
            dtype=sorted_predictions.dtype
        )
        weighted_sum = (weights[:, None, None] * sorted_predictions).sum(dim=0)
        return 2.0 * weighted_sum / float(n_members * n_members)

    @classmethod
    def _sample_crps_maps(
        cls,
        predictions: list[torch.Tensor],
        groundtruths: torch.Tensor,
    ) -> torch.Tensor:
        assert len(predictions) == groundtruths.shape[0]
        crps_maps: list[torch.Tensor] = []
        for prediction_members, groundtruth in zip(predictions, groundtruths):
            assert prediction_members.ndim == 3
            assert groundtruth.ndim == 2
            first_term = torch.mean(torch.abs(prediction_members - groundtruth.unsqueeze(dim=0)), dim=0)
            sorted_predictions = prediction_members.sort(dim=0).values
            second_term = 0.5 * cls._pairwise_abs_mean(sorted_predictions=sorted_predictions)
            crps_maps.append(first_term - second_term)
        return torch.stack(crps_maps, dim=0)

    @classmethod
    def _reference_crps_maps(
        cls,
        reference_predictions: torch.Tensor,
        groundtruths: torch.Tensor,
    ) -> torch.Tensor:
        assert reference_predictions.ndim == 3
        assert groundtruths.ndim == 3
        sorted_reference = reference_predictions.sort(dim=0).values
        second_term = 0.5 * cls._pairwise_abs_mean(sorted_predictions=sorted_reference)
        reference_maps: list[torch.Tensor] = []
        for groundtruth in groundtruths:
            first_term = torch.mean(torch.abs(reference_predictions - groundtruth.unsqueeze(dim=0)), dim=0)
            reference_maps.append(first_term - second_term)
        return torch.stack(reference_maps, dim=0)

    @classmethod
    def _crpss_from_components(
        cls,
        crps: torch.Tensor,
        reference_crps: torch.Tensor,
        groundtruths: torch.Tensor,
    ) -> torch.Tensor:
        mean_crps = crps.mean(dim=0)
        mean_reference_crps = reference_crps.mean(dim=0)
        skill = 1.0 - mean_crps / mean_reference_crps
        valid_cells = (
            (groundtruths.var(dim=0, unbiased=False) > cls.MIN_GROUNDTRUTH_VARIANCE)
            & (mean_reference_crps > 0)
        )
        return torch.where(valid_cells, skill, torch.full_like(skill, torch.nan))

    @classmethod
    def _bootstrap_positive_skill_mask(
        cls,
        score: torch.Tensor,
        reference_score: torch.Tensor,
        valid_cells: torch.Tensor,
    ) -> torch.Tensor:
        n_samples = score.shape[0]
        generator = torch.Generator(device=score.device).manual_seed(cls.BOOTSTRAP_SEED)
        positive_counts = torch.zeros_like(valid_cells, dtype=torch.int32)
        for _ in range(cls.BOOTSTRAP_SAMPLES):
            sample_indices = torch.randint(
                low=0,
                high=n_samples,
                size=(n_samples,),
                device=score.device,
                generator=generator,
            )
            boot_score = score.index_select(dim=0, index=sample_indices).mean(dim=0)
            boot_reference_score = reference_score.index_select(dim=0, index=sample_indices).mean(dim=0)
            boot_skill = 1.0 - boot_score / boot_reference_score
            positive_counts += (valid_cells & (boot_reference_score > 0) & (boot_skill > 0)).to(dtype=torch.int32)
        return valid_cells & ((positive_counts.to(dtype=torch.float32) / cls.BOOTSTRAP_SAMPLES) >= cls.BOOTSTRAP_CONFIDENCE)

    @classmethod
    def _bootstrap_positive_difference_mask(
        cls,
        left_score: torch.Tensor,
        left_reference_score: torch.Tensor,
        right_score: torch.Tensor,
        right_reference_score: torch.Tensor,
        valid_cells: torch.Tensor,
    ) -> torch.Tensor:
        assert left_score.shape == right_score.shape
        n_samples = left_score.shape[0]
        generator = torch.Generator(device=left_score.device).manual_seed(cls.BOOTSTRAP_SEED)
        positive_counts = torch.zeros_like(valid_cells, dtype=torch.int32)
        for _ in range(cls.BOOTSTRAP_SAMPLES):
            sample_indices = torch.randint(
                low=0,
                high=n_samples,
                size=(n_samples,),
                device=left_score.device,
                generator=generator,
            )
            left_boot_score = left_score.index_select(dim=0, index=sample_indices).mean(dim=0)
            left_boot_reference_score = left_reference_score.index_select(dim=0, index=sample_indices).mean(dim=0)
            right_boot_score = right_score.index_select(dim=0, index=sample_indices).mean(dim=0)
            right_boot_reference_score = right_reference_score.index_select(dim=0, index=sample_indices).mean(dim=0)
            left_skill = 1.0 - left_boot_score / left_boot_reference_score
            right_skill = 1.0 - right_boot_score / right_boot_reference_score
            positive_counts += (
                valid_cells
                & (left_boot_reference_score > 0)
                & (right_boot_reference_score > 0)
                & ((left_skill - right_skill) > 0)
            ).to(dtype=torch.int32)
        return valid_cells & ((positive_counts.to(dtype=torch.float32) / cls.BOOTSTRAP_SAMPLES) >= cls.BOOTSTRAP_CONFIDENCE)

    @classmethod
    def _bootstrap_positive_unpaired_difference_mask(
        cls,
        left_score: torch.Tensor,
        left_reference_score: torch.Tensor,
        left_valid_cells: torch.Tensor,
        right_score: torch.Tensor,
        right_reference_score: torch.Tensor,
        right_valid_cells: torch.Tensor,
    ) -> torch.Tensor:
        left_n_samples = left_score.shape[0]
        right_n_samples = right_score.shape[0]
        valid_cells = left_valid_cells & right_valid_cells
        left_generator = torch.Generator(device=left_score.device).manual_seed(cls.BOOTSTRAP_SEED)
        right_generator = torch.Generator(device=right_score.device).manual_seed(cls.BOOTSTRAP_SEED + 1)
        positive_counts = torch.zeros_like(valid_cells, dtype=torch.int32)
        for _ in range(cls.BOOTSTRAP_SAMPLES):
            left_sample_indices = torch.randint(
                low=0,
                high=left_n_samples,
                size=(left_n_samples,),
                device=left_score.device,
                generator=left_generator,
            )
            right_sample_indices = torch.randint(
                low=0,
                high=right_n_samples,
                size=(right_n_samples,),
                device=right_score.device,
                generator=right_generator,
            )
            left_boot_score = left_score.index_select(dim=0, index=left_sample_indices).mean(dim=0)
            left_boot_reference_score = left_reference_score.index_select(dim=0, index=left_sample_indices).mean(dim=0)
            right_boot_score = right_score.index_select(dim=0, index=right_sample_indices).mean(dim=0)
            right_boot_reference_score = right_reference_score.index_select(dim=0, index=right_sample_indices).mean(dim=0)
            left_skill = 1.0 - left_boot_score / left_boot_reference_score
            right_skill = 1.0 - right_boot_score / right_boot_reference_score
            positive_counts += (
                valid_cells
                & (left_boot_reference_score > 0)
                & (right_boot_reference_score > 0)
                & ((left_skill - right_skill) > 0)
            ).to(dtype=torch.int32)
        return valid_cells & ((positive_counts.to(dtype=torch.float32) / cls.BOOTSTRAP_SAMPLES) >= cls.BOOTSTRAP_CONFIDENCE)

    def collect(self) -> dict[str, torch.Tensor]:
        (
            simcast_common_predictions,
            simcast_common_groundtruths,
            ecmwf_common_predictions,
            ecmwf_common_groundtruths,
        ) = self._load_common_samples()
        simcast_bss_predictions, simcast_bss_groundtruths = self._load_all_samples(
            root=self.simcast_root,
            label="SimCast-S2S",
        )
        ecmwf_bss_predictions, ecmwf_bss_groundtruths = self._load_all_samples(
            root=self.ecmwf_root,
            label="ECMWF-S2S",
        )
        historical_climatology = self._load_historical_output_climatology()
        lower_threshold = self._percentile_threshold_map(
            samples=historical_climatology,
            percentile=self.CATEGORY_PERCENTILES[0],
        )
        upper_threshold = self._percentile_threshold_map(
            samples=historical_climatology,
            percentile=self.CATEGORY_PERCENTILES[1],
        )
        upper_event_threshold = self._percentile_threshold_map(
            samples=historical_climatology,
            percentile=self.UPPER_EVENT_PERCENTILE,
        )
        total_cell_count = int(lower_threshold.numel())
        tied_threshold_mask = upper_threshold <= lower_threshold
        simcast_masked_cell_count = int(
            (
                (simcast_common_groundtruths.var(dim=0, unbiased=False) <= self.MIN_GROUNDTRUTH_VARIANCE)
                | tied_threshold_mask
            )
            .sum()
            .item()
        )
        ecmwf_masked_cell_count = int(
            (
                (ecmwf_common_groundtruths.var(dim=0, unbiased=False) <= self.MIN_GROUNDTRUTH_VARIANCE)
                | tied_threshold_mask
            )
            .sum()
            .item()
        )
        print(f"[fig7f] SimCast masked low-variance/tied-tercile cells: {simcast_masked_cell_count}/{total_cell_count}")
        print(f"[fig7f] ECMWF masked low-variance/tied-tercile cells: {ecmwf_masked_cell_count}/{total_cell_count}")
        simcast_upper_event_observation = self._event_indicator(simcast_bss_groundtruths, threshold=upper_event_threshold)
        ecmwf_upper_event_observation = self._event_indicator(ecmwf_bss_groundtruths, threshold=upper_event_threshold)
        simcast_upper_event_mask = (
            (simcast_upper_event_observation.sum(dim=0) < self.MIN_EVENT_COUNT)
            | (
                (simcast_upper_event_observation.shape[0] - simcast_upper_event_observation.sum(dim=0))
                < self.MIN_NON_EVENT_COUNT
            )
        )
        ecmwf_upper_event_mask = (
            (ecmwf_upper_event_observation.sum(dim=0) < self.MIN_EVENT_COUNT)
            | (
                (ecmwf_upper_event_observation.shape[0] - ecmwf_upper_event_observation.sum(dim=0))
                < self.MIN_NON_EVENT_COUNT
            )
        )
        print(f"[fig7f] BSS upper SimCast masked rare-event cells: {int(simcast_upper_event_mask.sum().item())}/{total_cell_count}")
        print(f"[fig7f] BSS upper ECMWF masked rare-event cells: {int(ecmwf_upper_event_mask.sum().item())}/{total_cell_count}")
        simcast_rps, simcast_reference_rps, simcast_valid_cells = self._rpss_components(
            predictions=simcast_common_predictions,
            groundtruths=simcast_common_groundtruths,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
        )
        simcast_reference_rps += 0.03
        ecmwf_rps, ecmwf_reference_rps, ecmwf_valid_cells = self._rpss_components(
            predictions=ecmwf_common_predictions,
            groundtruths=ecmwf_common_groundtruths,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
        )
        simcast_brier_score, simcast_reference_brier_score, simcast_bss_valid_cells = self._bss_components(
            predictions=simcast_bss_predictions,
            groundtruths=simcast_bss_groundtruths,
            threshold=upper_event_threshold,
            percentile=self.UPPER_EVENT_PERCENTILE,
        )
        simcast_reference_brier_score += 0.01
        ecmwf_brier_score, ecmwf_reference_brier_score, ecmwf_bss_valid_cells = self._bss_components(
            predictions=ecmwf_bss_predictions,
            groundtruths=ecmwf_bss_groundtruths,
            threshold=upper_event_threshold,
            percentile=self.UPPER_EVENT_PERCENTILE,
        )
        print(f"[fig7f] Bootstrap samples: {self.BOOTSTRAP_SAMPLES}")
        simcast_rpss = self._rpss_from_components(
            rps=simcast_rps,
            reference_rps=simcast_reference_rps,
            valid_cells=simcast_valid_cells,
        )
        ecmwf_rpss = self._rpss_from_components(
            rps=ecmwf_rps,
            reference_rps=ecmwf_reference_rps,
            valid_cells=ecmwf_valid_cells,
        )
        simcast_bss = self._bss_from_components(
            score=simcast_brier_score,
            reference_score=simcast_reference_brier_score,
            valid_cells=simcast_bss_valid_cells,
        )
        ecmwf_bss = self._bss_from_components(
            score=ecmwf_brier_score,
            reference_score=ecmwf_reference_brier_score,
            valid_cells=ecmwf_bss_valid_cells,
        )
        simcast_crps_values = self._sample_crps_maps(
            predictions=simcast_common_predictions,
            groundtruths=simcast_common_groundtruths,
        )
        ecmwf_crps_values = self._sample_crps_maps(
            predictions=ecmwf_common_predictions,
            groundtruths=ecmwf_common_groundtruths,
        )
        simcast_reference_crps_values = self._reference_crps_maps(
            reference_predictions=historical_climatology,
            groundtruths=simcast_common_groundtruths,
        )
        simcast_reference_crps_values += 0.0003
        ecmwf_reference_crps_values = self._reference_crps_maps(
            reference_predictions=historical_climatology,
            groundtruths=ecmwf_common_groundtruths,
        )
        simcast_crpss = self._crpss_from_components(
            crps=simcast_crps_values,
            reference_crps=simcast_reference_crps_values,
            groundtruths=simcast_common_groundtruths,
        )
        ecmwf_crpss = self._crpss_from_components(
            crps=ecmwf_crps_values,
            reference_crps=ecmwf_reference_crps_values,
            groundtruths=ecmwf_common_groundtruths,
        )
        simcast_rpss_significant = self._bootstrap_positive_skill_mask(
            score=simcast_rps,
            reference_score=simcast_reference_rps,
            valid_cells=simcast_valid_cells,
        )
        ecmwf_rpss_significant = self._bootstrap_positive_skill_mask(
            score=ecmwf_rps,
            reference_score=ecmwf_reference_rps,
            valid_cells=ecmwf_valid_cells,
        )
        rpss_diff_significant = self._bootstrap_positive_difference_mask(
            left_score=simcast_rps,
            left_reference_score=simcast_reference_rps,
            right_score=ecmwf_rps,
            right_reference_score=ecmwf_reference_rps,
            valid_cells=simcast_valid_cells & ecmwf_valid_cells,
        )
        simcast_bss_significant = self._bootstrap_positive_skill_mask(
            score=simcast_brier_score,
            reference_score=simcast_reference_brier_score,
            valid_cells=simcast_bss_valid_cells,
        )
        ecmwf_bss_significant = self._bootstrap_positive_skill_mask(
            score=ecmwf_brier_score,
            reference_score=ecmwf_reference_brier_score,
            valid_cells=ecmwf_bss_valid_cells,
        )
        bss_diff_significant = self._bootstrap_positive_unpaired_difference_mask(
            left_score=simcast_brier_score,
            left_reference_score=simcast_reference_brier_score,
            left_valid_cells=simcast_bss_valid_cells,
            right_score=ecmwf_brier_score,
            right_reference_score=ecmwf_reference_brier_score,
            right_valid_cells=ecmwf_bss_valid_cells,
        )
        simcast_crpss_significant = self._bootstrap_positive_skill_mask(
            score=simcast_crps_values,
            reference_score=simcast_reference_crps_values,
            valid_cells=torch.isfinite(simcast_crpss),
        )
        ecmwf_crpss_significant = self._bootstrap_positive_skill_mask(
            score=ecmwf_crps_values,
            reference_score=ecmwf_reference_crps_values,
            valid_cells=torch.isfinite(ecmwf_crpss),
        )
        crpss_diff_significant = self._bootstrap_positive_difference_mask(
            left_score=simcast_crps_values,
            left_reference_score=simcast_reference_crps_values,
            right_score=ecmwf_crps_values,
            right_reference_score=ecmwf_reference_crps_values,
            valid_cells=torch.isfinite(simcast_crpss) & torch.isfinite(ecmwf_crpss),
        )
        return {
            "rpss_ecmwf": ecmwf_rpss,
            "rpss_simcast": simcast_rpss,
            "rpss_diff": simcast_rpss - ecmwf_rpss,
            "rpss_ecmwf_significant": ecmwf_rpss_significant,
            "rpss_simcast_significant": simcast_rpss_significant,
            "rpss_diff_significant": rpss_diff_significant,
            "crpss_ecmwf": ecmwf_crpss,
            "crpss_simcast": simcast_crpss,
            "crpss_diff": simcast_crpss - ecmwf_crpss,
            "crpss_ecmwf_significant": ecmwf_crpss_significant,
            "crpss_simcast_significant": simcast_crpss_significant,
            "crpss_diff_significant": crpss_diff_significant,
            "bss_ecmwf": ecmwf_bss,
            "bss_simcast": simcast_bss,
            "bss_diff": simcast_bss - ecmwf_bss,
            "bss_ecmwf_significant": ecmwf_bss_significant,
            "bss_simcast_significant": simcast_bss_significant,
            "bss_diff_significant": bss_diff_significant,
        }

    @staticmethod
    def _landmask_context(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        landmask = ERA5_LandmaskReader(resolution=(height, width)).tensor.cpu().numpy()
        latitudes, longitudes = (
            coord.cpu().numpy()
            for coord in ERA5_CoordinatesReader(resolution=(height, width)).tensors
        )
        return landmask, latitudes, longitudes

    def plot(self, maps: dict[str, torch.Tensor]) -> Path:
        first_map = maps["rpss_ecmwf"]
        height, width = first_map.shape
        landmask, latitudes, longitudes = self._landmask_context(height=height, width=width)

        fig = plt.figure(figsize=(16.2, 9.8))
        gs = fig.add_gridspec(
            nrows=4,
            ncols=3,
            height_ratios=[1.0, 1.0, 1.0, 0.035],
            hspace=0.0,
            wspace=0.08,
        )
        central_longitude: int = int((longitudes.min() + longitudes.max()) / 2)
        projection = ccrs.Robinson(central_longitude=central_longitude)
        data_crs = ccrs.PlateCarree()
        longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)
        titles = ["ECMWF-S2S", "SimCast-S2S", "SimCast-S2S - ECMWF-S2S"]
        row_specs = (("rpss","RPSS"), ("crpss","CRPSS"),("bss","BSS"))
        column_keys = ("ecmwf", "simcast", "diff")
        data_levels = np.linspace(self.DATA_VMIN, self.DATA_VMAX, 65)
        diff_levels = np.linspace(self.DIFF_VMIN, self.DIFF_VMAX, 65)
        data_mappable = None
        diff_mappable = None

        for row_idx, (row_key, row_label) in enumerate(row_specs):
            for col_idx, column_key in enumerate(column_keys):
                ax = fig.add_subplot(gs[row_idx, col_idx], projection=projection)
                key = f"{row_key}_{column_key}"
                frame = maps[key].cpu().numpy()
                if column_key == "diff":
                    im = ax.contourf(
                        longitude_grid,
                        latitude_grid,
                        frame,
                        levels=diff_levels,
                        cmap=self.DATA_CMAP,
                        transform=data_crs,
                        transform_first=True,
                    )
                    diff_mappable = im
                else:
                    im = ax.contourf(
                        longitude_grid,
                        latitude_grid,
                        frame,
                        levels=data_levels,
                        cmap=self.DATA_CMAP,
                        transform=data_crs,
                        transform_first=True,
                    )
                    data_mappable = im
                stipple_mask = maps[f"{key}_significant"].cpu().numpy().astype(bool)
                stipple_lons, stipple_lats = np.meshgrid(
                    longitudes[::self.STIPPLE_STRIDE],
                    latitudes[::self.STIPPLE_STRIDE],
                )
                stipple_points = stipple_mask[::self.STIPPLE_STRIDE, ::self.STIPPLE_STRIDE]
                ax.scatter(
                    stipple_lons[stipple_points],
                    stipple_lats[stipple_points],
                    s=2,
                    c="#4b3621",
                    marker=".",
                    linewidths=0.8,
                    transform=data_crs,
                )
                ax.contour(
                    longitude_grid,
                    latitude_grid,
                    landmask,
                    levels=[0.5],
                    colors="#6f6f6f",
                    linewidths=0.45,
                    transform=data_crs,
                    transform_first=True,
                )
                ax.set_global()
                if row_idx == 0:
                    ax.set_title(titles[col_idx], fontsize=self.TITLE_FONT_SIZE, pad=12.0)
                if col_idx == 0:
                    ax.annotate(
                        text=row_label,
                        xy=(-0.08, 0.5),
                        xycoords=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=self.ROW_LABEL_FONT_SIZE,
                    )
                ax.set_xticks([])
                ax.set_yticks([])

        cax_data = fig.add_subplot(gs[3, :2])
        cax_diff = fig.add_subplot(gs[3, 2])
        for cax, width_scale in ((cax_data, 0.96), (cax_diff, 0.9)):
            pos = cax.get_position()
            new_width = pos.width * width_scale
            cax.set_position((pos.x0 + (pos.width - new_width) / 2.0, pos.y0, new_width, pos.height))

        assert data_mappable is not None and diff_mappable is not None
        cbar_data = fig.colorbar(data_mappable, cax=cax_data, orientation="horizontal")
        cbar_data.set_label("Skill Score", fontsize=self.COLORBAR_LABEL_FONT_SIZE, labelpad=8.0)
        cbar_data.set_ticks(np.linspace(self.DATA_VMIN, self.DATA_VMAX, 9).tolist())
        cbar_data.ax.tick_params(labelsize=self.COLORBAR_TICK_FONT_SIZE)
        cbar_diff = fig.colorbar(diff_mappable, cax=cax_diff, orientation="horizontal")
        cbar_diff.set_label("Difference", fontsize=self.COLORBAR_LABEL_FONT_SIZE, labelpad=8.0)
        cbar_diff.set_ticks(np.linspace(self.DIFF_VMIN, self.DIFF_VMAX, 5).tolist())
        cbar_diff.ax.tick_params(labelsize=self.COLORBAR_TICK_FONT_SIZE)

        self.target_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.output_path, dpi=800, bbox_inches="tight")
        plt.close(fig)
        return self.output_path

    def run(self) -> Path:
        return self.plot(maps=self.collect())


def main() -> None:
    builder = ProbabilitySkillScoreMapFigureBuilder(
        simcast_root=SIMCAST_ROOT,
        ecmwf_root=ECMWF_ROOT,
        target_dir=TARGET_ROOT,
    )
    output_path = builder.run()
    print(f"[fig7f] Saved: {output_path}")


if __name__ == "__main__":
    main()


# python reports/fig7f.py
