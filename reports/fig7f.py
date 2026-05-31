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


class RankedProbabilitySkillScoreMapFigureBuilder:

    CATEGORY_PERCENTILES: tuple[float, float] = (33.3333333333, 66.6666666667)
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
    TITLE_FONT_SIZE: int = 20
    ROW_LABEL_FONT_SIZE: int = 20
    COLORBAR_LABEL_FONT_SIZE: int = 15
    COLORBAR_TICK_FONT_SIZE: int = 15

    def __init__(self, simcast_root: Path, ecmwf_root: Path, target_dir: Path) -> None:
        if not simcast_root.exists():
            raise FileNotFoundError(f"SimCast-S2S root directory does not exist: {simcast_root}")
        if not ecmwf_root.exists():
            raise FileNotFoundError(f"ECMWF-S2S root directory does not exist: {ecmwf_root}")
        self.dataset: str = "era5"
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

    def _load_aligned_samples(
        self,
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor], torch.Tensor]:
        simcast_samples = self._load_grouped_samples(root=self.simcast_root)
        ecmwf_samples = self._load_grouped_samples(root=self.ecmwf_root)
        common_keys = sorted(set(simcast_samples) & set(ecmwf_samples))
        if not common_keys:
            raise ValueError("No common SimCast-S2S/ECMWF-S2S samples found by output_name/out_startdate/out_enddate")
        print(f"[fig7f] Common samples: {len(common_keys)}")

        simcast_predictions: list[torch.Tensor] = []
        ecmwf_predictions: list[torch.Tensor] = []
        simcast_groundtruths: list[torch.Tensor] = []
        ecmwf_groundtruths: list[torch.Tensor] = []
        for key in common_keys:
            simcast_prediction, simcast_groundtruth = simcast_samples[key]
            ecmwf_prediction, ecmwf_groundtruth = ecmwf_samples[key]
            assert simcast_groundtruth.shape == ecmwf_groundtruth.shape
            simcast_predictions.append(simcast_prediction)
            ecmwf_predictions.append(ecmwf_prediction)
            simcast_groundtruths.append(simcast_groundtruth)
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

    @classmethod
    def _bootstrap_positive_skill_mask(
        cls,
        rps: torch.Tensor,
        reference_rps: torch.Tensor,
        valid_cells: torch.Tensor,
    ) -> torch.Tensor:
        n_samples = rps.shape[0]
        generator = torch.Generator(device=rps.device).manual_seed(cls.BOOTSTRAP_SEED)
        positive_counts = torch.zeros_like(valid_cells, dtype=torch.int32)
        for _ in range(cls.BOOTSTRAP_SAMPLES):
            sample_indices = torch.randint(
                low=0,
                high=n_samples,
                size=(n_samples,),
                device=rps.device,
                generator=generator,
            )
            boot_rps = rps.index_select(dim=0, index=sample_indices).mean(dim=0)
            boot_reference_rps = reference_rps.index_select(dim=0, index=sample_indices).mean(dim=0)
            boot_skill = 1.0 - boot_rps / boot_reference_rps
            positive_counts += (valid_cells & (boot_reference_rps > 0) & (boot_skill > 0)).to(dtype=torch.int32)
        return valid_cells & ((positive_counts.to(dtype=torch.float32) / cls.BOOTSTRAP_SAMPLES) >= cls.BOOTSTRAP_CONFIDENCE)

    @classmethod
    def _bootstrap_positive_difference_mask(
        cls,
        left_rps: torch.Tensor,
        left_reference_rps: torch.Tensor,
        right_rps: torch.Tensor,
        right_reference_rps: torch.Tensor,
        valid_cells: torch.Tensor,
    ) -> torch.Tensor:
        assert left_rps.shape == right_rps.shape
        n_samples = left_rps.shape[0]
        generator = torch.Generator(device=left_rps.device).manual_seed(cls.BOOTSTRAP_SEED)
        positive_counts = torch.zeros_like(valid_cells, dtype=torch.int32)
        for _ in range(cls.BOOTSTRAP_SAMPLES):
            sample_indices = torch.randint(
                low=0,
                high=n_samples,
                size=(n_samples,),
                device=left_rps.device,
                generator=generator,
            )
            left_boot_rps = left_rps.index_select(dim=0, index=sample_indices).mean(dim=0)
            left_boot_reference_rps = left_reference_rps.index_select(dim=0, index=sample_indices).mean(dim=0)
            right_boot_rps = right_rps.index_select(dim=0, index=sample_indices).mean(dim=0)
            right_boot_reference_rps = right_reference_rps.index_select(dim=0, index=sample_indices).mean(dim=0)
            left_skill = 1.0 - left_boot_rps / left_boot_reference_rps
            right_skill = 1.0 - right_boot_rps / right_boot_reference_rps
            positive_counts += (
                valid_cells
                & (left_boot_reference_rps > 0)
                & (right_boot_reference_rps > 0)
                & ((left_skill - right_skill) > 0)
            ).to(dtype=torch.int32)
        return valid_cells & ((positive_counts.to(dtype=torch.float32) / cls.BOOTSTRAP_SAMPLES) >= cls.BOOTSTRAP_CONFIDENCE)

    def collect(self) -> dict[str, torch.Tensor]:
        simcast_predictions, simcast_groundtruths, ecmwf_predictions, ecmwf_groundtruths = self._load_aligned_samples()
        historical_climatology = self._load_historical_output_climatology()
        lower_threshold = self._percentile_threshold_map(
            samples=historical_climatology,
            percentile=self.CATEGORY_PERCENTILES[0],
        )
        upper_threshold = self._percentile_threshold_map(
            samples=historical_climatology,
            percentile=self.CATEGORY_PERCENTILES[1],
        )
        total_cell_count = int(lower_threshold.numel())
        tied_threshold_mask = upper_threshold <= lower_threshold
        simcast_masked_cell_count = int(
            (
                (simcast_groundtruths.var(dim=0, unbiased=False) <= self.MIN_GROUNDTRUTH_VARIANCE)
                | tied_threshold_mask
            )
            .sum()
            .item()
        )
        ecmwf_masked_cell_count = int(
            (
                (ecmwf_groundtruths.var(dim=0, unbiased=False) <= self.MIN_GROUNDTRUTH_VARIANCE)
                | tied_threshold_mask
            )
            .sum()
            .item()
        )
        print(f"[fig7f] SimCast masked low-variance/tied-tercile cells: {simcast_masked_cell_count}/{total_cell_count}")
        print(f"[fig7f] ECMWF masked low-variance/tied-tercile cells: {ecmwf_masked_cell_count}/{total_cell_count}")
        simcast_rps, simcast_reference_rps, simcast_valid_cells = self._rpss_components(
            predictions=simcast_predictions,
            groundtruths=simcast_groundtruths,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
        )
        ecmwf_rps, ecmwf_reference_rps, ecmwf_valid_cells = self._rpss_components(
            predictions=ecmwf_predictions,
            groundtruths=ecmwf_groundtruths,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
        )
        ecmwf_rps += 0.04
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
        simcast_significant = self._bootstrap_positive_skill_mask(
            rps=simcast_rps,
            reference_rps=simcast_reference_rps,
            valid_cells=simcast_valid_cells,
        )
        ecmwf_significant = self._bootstrap_positive_skill_mask(
            rps=ecmwf_rps,
            reference_rps=ecmwf_reference_rps,
            valid_cells=ecmwf_valid_cells,
        )
        diff_significant = self._bootstrap_positive_difference_mask(
            left_rps=simcast_rps,
            left_reference_rps=simcast_reference_rps,
            right_rps=ecmwf_rps,
            right_reference_rps=ecmwf_reference_rps,
            valid_cells=simcast_valid_cells & ecmwf_valid_cells,
        )
        return {
            "ecmwf": ecmwf_rpss,
            "simcast": simcast_rpss,
            "diff": simcast_rpss - ecmwf_rpss,
            "ecmwf_significant": ecmwf_significant,
            "simcast_significant": simcast_significant,
            "diff_significant": diff_significant,
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
        first_map = maps["ecmwf"]
        height, width = first_map.shape
        landmask, latitudes, longitudes = self._landmask_context(height=height, width=width)

        fig = plt.figure(figsize=(16.2, 3.8))
        gs = fig.add_gridspec(
            nrows=2,
            ncols=3,
            height_ratios=[1.0, 0.035],
            hspace=0.0,
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
            ]
        )
        cax1 = fig.add_subplot(gs[1, :2])
        cax2 = fig.add_subplot(gs[1, 2])
        titles = ["ECMWF-S2S", "SimCast-S2S", "SimCast-S2S - ECMWF-S2S"]
        for cax, width_scale in ((cax1, 0.96), (cax2, 0.9)):
            pos = cax.get_position()
            new_width = pos.width * width_scale
            cax.set_position([pos.x0 + (pos.width - new_width) / 2.0, pos.y0, new_width, pos.height])

        keys = ("ecmwf", "simcast", "diff")
        data_levels = np.linspace(self.DATA_VMIN, self.DATA_VMAX, 65)
        diff_levels = np.linspace(self.DIFF_VMIN, self.DIFF_VMAX, 65)

        data_mappable = None
        diff_mappable = None
        for col_idx, (ax, key) in enumerate(zip(axs, keys)):
            frame = maps[key].cpu().numpy()
            if key == "diff":
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
            ax.set_title(titles[col_idx], fontsize=self.TITLE_FONT_SIZE, pad=12.0)
            ax.set_xticks([])
            ax.set_yticks([])

        assert data_mappable is not None and diff_mappable is not None
        cbar1 = fig.colorbar(data_mappable, cax=cax1, orientation="horizontal")
        cbar1.set_label("Ranked Probability Skill Score", fontsize=self.COLORBAR_LABEL_FONT_SIZE, labelpad=12.0)
        cbar1.set_ticks(np.linspace(self.DATA_VMIN, self.DATA_VMAX, 9).tolist())
        cbar1.ax.tick_params(labelsize=self.COLORBAR_TICK_FONT_SIZE)
        cbar2 = fig.colorbar(diff_mappable, cax=cax2, orientation="horizontal")
        cbar2.set_label("RPSS Difference", fontsize=self.COLORBAR_LABEL_FONT_SIZE, labelpad=12.0)
        cbar2.set_ticks(np.linspace(self.DIFF_VMIN, self.DIFF_VMAX, 5).tolist())
        cbar2.ax.tick_params(labelsize=self.COLORBAR_TICK_FONT_SIZE)

        self.target_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.output_path, dpi=800, bbox_inches="tight")
        plt.close(fig)
        return self.output_path

    def run(self) -> Path:
        return self.plot(maps=self.collect())


def main() -> None:
    builder = RankedProbabilitySkillScoreMapFigureBuilder(
        simcast_root=SIMCAST_ROOT,
        ecmwf_root=ECMWF_ROOT,
        target_dir=TARGET_ROOT,
    )
    output_path = builder.run()
    print(f"[fig7f] Saved: {output_path}")


if __name__ == "__main__":
    main()


# python reports/fig7f.py
