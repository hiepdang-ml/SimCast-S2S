from pathlib import Path
from typing import Any

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


class BrierSkillScoreMapFigureBuilder:

    UPPER_EVENT_PERCENTILE: float = 90.0
    LOWER_EVENT_PERCENTILE: float = 10.0
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
    MIN_GROUNDTRUTH_VARIANCE: float = 0.
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
    COLORBAR_VERTICAL_OFFSET: float = 0.018

    def __init__(self, simcast_root: Path, ecmwf_root: Path, target_dir: Path) -> None:
        if not simcast_root.exists():
            raise FileNotFoundError(f"SimCast-S2S root directory does not exist: {simcast_root}")
        if not ecmwf_root.exists():
            raise FileNotFoundError(f"ECMWF-S2S root directory does not exist: {ecmwf_root}")
        self.dataset: str = "era5"
        self.simcast_root: Path = simcast_root
        self.ecmwf_root: Path = ecmwf_root
        self.target_dir: Path = target_dir
        self.output_path: Path = self.target_dir.joinpath("fig7g.png")

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
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor], torch.Tensor]:
        simcast_samples = self._load_grouped_samples(root=self.simcast_root)
        ecmwf_samples = self._load_grouped_samples(root=self.ecmwf_root)
        print(f"[fig7g] SimCast samples: {len(simcast_samples)}")
        print(f"[fig7g] ECMWF samples: {len(ecmwf_samples)}")

        simcast_predictions: list[torch.Tensor] = []
        simcast_groundtruths: list[torch.Tensor] = []
        for key in sorted(simcast_samples)[:70]:
            simcast_prediction, simcast_groundtruth = simcast_samples[key]
            simcast_predictions.append(simcast_prediction)
            simcast_groundtruths.append(simcast_groundtruth)

        ecmwf_predictions: list[torch.Tensor] = []
        ecmwf_groundtruths: list[torch.Tensor] = []
        for key in sorted(ecmwf_samples):
            ecmwf_prediction, ecmwf_groundtruth = ecmwf_samples[key]
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
        print(f"[fig7g] Historical climatology samples: {climatology.shape[0]}")
        return climatology

    @staticmethod
    def _event_indicator(values: torch.Tensor, threshold: torch.Tensor, tail: str) -> torch.Tensor:
        if tail == "upper":
            return (values > threshold).to(dtype=torch.float32)
        if tail == "lower":
            return (values < threshold).to(dtype=torch.float32)
        raise ValueError(f"Unsupported BSS event tail: {tail}")

    @staticmethod
    def _climatology_probability(percentile: float, tail: str) -> float:
        if tail == "upper":
            return 1.0 - percentile / 100.0
        if tail == "lower":
            return percentile / 100.0
        raise ValueError(f"Unsupported BSS event tail: {tail}")

    @classmethod
    def _bss_components(
        cls,
        predictions: list[torch.Tensor],
        groundtruths: torch.Tensor,
        threshold: torch.Tensor,
        percentile: float,
        tail: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(predictions) == groundtruths.shape[0]
        event_probability = torch.stack(
            [cls._event_indicator(prediction, threshold=threshold, tail=tail).mean(dim=0) for prediction in predictions],
            dim=0,
        )
        event_observation = cls._event_indicator(groundtruths, threshold=threshold, tail=tail)
        climatology_probability = cls._climatology_probability(percentile=percentile, tail=tail)
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
    def _bss_from_components(score: torch.Tensor, reference_score: torch.Tensor, valid_cells: torch.Tensor) -> torch.Tensor:
        mean_score = score.mean(dim=0)
        mean_reference_score = reference_score.mean(dim=0)
        skill = 1.0 - mean_score / mean_reference_score
        return torch.where(valid_cells, skill, torch.zeros_like(skill))

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
    ) -> torch.Tensor:
        left_n_samples = left_score.shape[0]
        right_n_samples = right_score.shape[0]
        left_generator = torch.Generator(device=left_score.device).manual_seed(cls.BOOTSTRAP_SEED)
        right_generator = torch.Generator(device=right_score.device).manual_seed(cls.BOOTSTRAP_SEED + 1)
        positive_counts = torch.zeros_like(left_score.sum(dim=0), dtype=torch.int32)
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
                (left_boot_reference_score > 0)
                & (right_boot_reference_score > 0)
                & ((left_skill - right_skill) > 0)
            ).to(dtype=torch.int32)
        return ((positive_counts.to(dtype=torch.float32) / cls.BOOTSTRAP_SAMPLES) >= cls.BOOTSTRAP_CONFIDENCE)

    def collect(self) -> dict[str, torch.Tensor]:
        simcast_predictions, simcast_groundtruths, ecmwf_predictions, ecmwf_groundtruths = self._load_all_samples()
        historical_climatology = self._load_historical_output_climatology()
        total_cell_count = int(simcast_groundtruths.shape[-2] * simcast_groundtruths.shape[-1])
        simcast_masked_cell_count = int(
            (simcast_groundtruths.var(dim=0, unbiased=False) <= self.MIN_GROUNDTRUTH_VARIANCE).sum().item()
        )
        ecmwf_masked_cell_count = int(
            (ecmwf_groundtruths.var(dim=0, unbiased=False) <= self.MIN_GROUNDTRUTH_VARIANCE).sum().item()
        )
        print(f"[fig7g] SimCast masked low-variance cells: {simcast_masked_cell_count}/{total_cell_count}")
        print(f"[fig7g] ECMWF masked low-variance cells: {ecmwf_masked_cell_count}/{total_cell_count}")
        print(f"[fig7g] Bootstrap samples: {self.BOOTSTRAP_SAMPLES}")

        maps: dict[str, torch.Tensor] = {}
        event_specs = (
            ("upper", self.UPPER_EVENT_PERCENTILE, "upper"),
            ("lower", self.LOWER_EVENT_PERCENTILE, "lower"),
        )
        for row_key, percentile, tail in event_specs:
            threshold = self._percentile_threshold_map(samples=historical_climatology, percentile=percentile)
            simcast_event_observation = self._event_indicator(simcast_groundtruths, threshold=threshold, tail=tail)
            ecmwf_event_observation = self._event_indicator(ecmwf_groundtruths, threshold=threshold, tail=tail)
            simcast_event_mask = (
                (simcast_event_observation.sum(dim=0) < self.MIN_EVENT_COUNT)
                | ((simcast_event_observation.shape[0] - simcast_event_observation.sum(dim=0)) < self.MIN_NON_EVENT_COUNT)
            )
            ecmwf_event_mask = (
                (ecmwf_event_observation.sum(dim=0) < self.MIN_EVENT_COUNT)
                | ((ecmwf_event_observation.shape[0] - ecmwf_event_observation.sum(dim=0)) < self.MIN_NON_EVENT_COUNT)
            )
            print(
                f"[fig7g] {row_key} SimCast masked rare-event cells: "
                f"{int(simcast_event_mask.sum().item())}/{total_cell_count}"
            )
            print(
                f"[fig7g] {row_key} ECMWF masked rare-event cells: "
                f"{int(ecmwf_event_mask.sum().item())}/{total_cell_count}"
            )
            simcast_score, simcast_reference_score, simcast_valid_cells = self._bss_components(
                predictions=simcast_predictions,
                groundtruths=simcast_groundtruths,
                threshold=threshold,
                percentile=percentile,
                tail=tail,
            )
            ecmwf_score, ecmwf_reference_score, ecmwf_valid_cells = self._bss_components(
                predictions=ecmwf_predictions,
                groundtruths=ecmwf_groundtruths,
                threshold=threshold,
                percentile=percentile,
                tail=tail,
            )
            simcast_bss = self._bss_from_components(
                score=simcast_score,
                reference_score=simcast_reference_score,
                valid_cells=simcast_valid_cells,
            )
            ecmwf_bss = self._bss_from_components(
                score=ecmwf_score,
                reference_score=ecmwf_reference_score,
                valid_cells=ecmwf_valid_cells,
            )
            simcast_significant = self._bootstrap_positive_skill_mask(
                score=simcast_score,
                reference_score=simcast_reference_score,
                valid_cells=simcast_valid_cells,
            )
            ecmwf_significant = self._bootstrap_positive_skill_mask(
                score=ecmwf_score,
                reference_score=ecmwf_reference_score,
                valid_cells=ecmwf_valid_cells,
            )
            diff_significant = self._bootstrap_positive_difference_mask(
                left_score=simcast_score,
                left_reference_score=simcast_reference_score,
                right_score=ecmwf_score,
                right_reference_score=ecmwf_reference_score,
            )
            maps[f"{row_key}_ecmwf"] = ecmwf_bss
            maps[f"{row_key}_simcast"] = simcast_bss
            maps[f"{row_key}_diff"] = simcast_bss - ecmwf_bss
            maps[f"{row_key}_ecmwf_significant"] = ecmwf_significant
            maps[f"{row_key}_simcast_significant"] = simcast_significant
            maps[f"{row_key}_diff_significant"] = diff_significant
        return maps

    @staticmethod
    def _landmask_context(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        landmask = ERA5_LandmaskReader(resolution=(height, width)).tensor.cpu().numpy()
        latitudes, longitudes = (
            coord.cpu().numpy()
            for coord in ERA5_CoordinatesReader(resolution=(height, width)).tensors
        )
        return landmask, latitudes, longitudes

    def plot(self, maps: dict[str, torch.Tensor]) -> Path:
        first_map = maps["upper_ecmwf"]
        height, width = first_map.shape
        landmask, latitudes, longitudes = self._landmask_context(height=height, width=width)

        fig = plt.figure(figsize=(16.2, 6.2))
        gs = fig.add_gridspec(
            nrows=3,
            ncols=3,
            height_ratios=[1.0, 1.0, 0.035],
            hspace=0.00,
            wspace=0.08,
        )
        central_longitude = float((longitudes.min() + longitudes.max()) / 2.0)
        projection = ccrs.Robinson(central_longitude=central_longitude)
        data_crs = ccrs.PlateCarree()
        axs = np.array(
            [
                fig.add_subplot(gs[0, 0], projection=projection),
                fig.add_subplot(gs[0, 1], projection=projection),
                fig.add_subplot(gs[0, 2], projection=projection),
                fig.add_subplot(gs[1, 0], projection=projection),
                fig.add_subplot(gs[1, 1], projection=projection),
                fig.add_subplot(gs[1, 2], projection=projection),
            ]
        ).reshape(2, 3)
        cax1 = fig.add_subplot(gs[2, :2])
        cax2 = fig.add_subplot(gs[2, 2])
        titles = ["ECMWF-S2S", "SimCast-S2S", "SimCast-S2S - ECMWF-S2S"]
        for cax, width_scale in ((cax1, 0.96), (cax2, 0.9)):
            pos = cax.get_position()
            new_width = pos.width * width_scale
            cax.set_position(
                (
                    pos.x0 + (pos.width - new_width) / 2.0,
                    pos.y0 - self.COLORBAR_VERTICAL_OFFSET,
                    new_width,
                    pos.height,
                )
            )

        row_specs = (
            ("upper", f"BSS > P{int(self.UPPER_EVENT_PERCENTILE)}"),
            ("lower", f"BSS < P{int(self.LOWER_EVENT_PERCENTILE)}"),
        )
        column_keys = ("ecmwf", "simcast", "diff")
        data_levels = np.linspace(self.DATA_VMIN, self.DATA_VMAX, 33)
        diff_levels = np.linspace(self.DIFF_VMIN, self.DIFF_VMAX, 33)

        data_mappable = None
        diff_mappable = None
        for row_idx, (row_key, row_label) in enumerate(row_specs):
            for col_idx, column_key in enumerate(column_keys):
                ax = axs[row_idx, col_idx]
                key = f"{row_key}_{column_key}"
                frame = maps[key].cpu().numpy()
                if column_key == "diff":
                    im = ax.contourf(
                        longitudes,
                        latitudes,
                        frame,
                        levels=diff_levels,
                        cmap=self.DATA_CMAP,
                        transform=data_crs,
                    )
                    diff_mappable = im
                else:
                    im = ax.contourf(
                        longitudes,
                        latitudes,
                        frame,
                        levels=data_levels,
                        cmap=self.DATA_CMAP,
                        transform=data_crs,
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
                    s=2.,
                    c="#4b3621",
                    marker=".",
                    linewidths=0.8,
                    transform=data_crs,
                )
                ax.contour(
                    longitudes,
                    latitudes,
                    landmask,
                    levels=[0.5],
                    colors="#6f6f6f",
                    linewidths=0.45,
                    transform=data_crs,
                )
                ax.set_global()
                if row_idx == 0:
                    ax.set_title(titles[col_idx], fontsize=self.TITLE_FONT_SIZE, pad=12.0)
                if col_idx == 0:
                    ax.text(
                        -0.08,
                        0.5,
                        row_label,
                        transform=ax.transAxes,
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=self.ROW_LABEL_FONT_SIZE,
                    )
                ax.set_xticks([])
                ax.set_yticks([])

        assert data_mappable is not None and diff_mappable is not None
        cbar1 = fig.colorbar(data_mappable, cax=cax1, orientation="horizontal")
        cbar1.set_label("Brier Skill Score", fontsize=self.COLORBAR_LABEL_FONT_SIZE, labelpad=12.0)
        cbar1.set_ticks(np.linspace(self.DATA_VMIN, self.DATA_VMAX, 9).tolist())
        cbar1.ax.tick_params(labelsize=self.COLORBAR_TICK_FONT_SIZE)
        cbar2 = fig.colorbar(diff_mappable, cax=cax2, orientation="horizontal")
        cbar2.set_label("BSS Difference", fontsize=self.COLORBAR_LABEL_FONT_SIZE, labelpad=12.0)
        cbar2.set_ticks(np.linspace(self.DIFF_VMIN, self.DIFF_VMAX, 5).tolist())
        cbar2.ax.tick_params(labelsize=self.COLORBAR_TICK_FONT_SIZE)

        self.target_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.output_path, dpi=800, bbox_inches="tight")
        plt.close(fig)
        return self.output_path

    def run(self) -> Path:
        return self.plot(maps=self.collect())


def main() -> None:
    builder = BrierSkillScoreMapFigureBuilder(
        simcast_root=SIMCAST_ROOT,
        ecmwf_root=ECMWF_ROOT,
        target_dir=TARGET_ROOT,
    )
    output_path = builder.run()
    print(f"[fig7g] Saved: {output_path}")


if __name__ == "__main__":
    main()


# python reports/fig7g.py
