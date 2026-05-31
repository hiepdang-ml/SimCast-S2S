from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import torch
from datapipeline.readers import ERA5_CoordinatesReader, ERA5_LandmaskReader
from matplotlib.colors import LinearSegmentedColormap


SIMCAST_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/finetune/diffusion_v23_cosine_eta000_100steps_100members_guidancescale200")
ECMWF_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/ecmwfs2s_28/")
TARGET_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/")


class CRPSMapFigureBuilder:

    MAX_GROUPS: int = 70
    BOOTSTRAP_SAMPLES: int = 1000
    BOOTSTRAP_CONFIDENCE: float = 0.975
    BOOTSTRAP_SEED: int = 7341
    STIPPLE_STRIDE: int = 6
    TITLE_FONT_SIZE: int = 20
    ROW_LABEL_FONT_SIZE: int = 20
    COLORBAR_LABEL_FONT_SIZE: int = 15
    COLORBAR_TICK_FONT_SIZE: int = 15
    LOWER_IS_BETTER_CMAP = LinearSegmentedColormap.from_list(
        "crps_lower_is_better",
        [
            (0.00, "#ffffff"),
            (0.10, "#f7f7f7"),
            (0.26, "#fddbc7"),
            (0.48, "#ef8a62"),
            (0.74, "#b2182b"),
            (1.00, "#67001f"),
        ],
    )
    HIGHER_IS_BETTER_CMAP = LinearSegmentedColormap.from_list(
        "crps_higher_is_better",
        [
            (0.00, "#ffffff"),
            (0.10, "#f7f7f7"),
            (0.26, "#d1e5f0"),
            (0.48, "#92c5de"),
            (0.74, "#4393c3"),
            (1.00, "#2166ac"),
        ],
    )
    DIFF_CMAP = LinearSegmentedColormap.from_list(
        "crps_diff_better_worse",
        [
            (0.00, "#67001f"),
            (0.18, "#b2182b"),
            (0.36, "#ef8a62"),
            (0.50, "#f7f7f7"),
            (0.64, "#92c5de"),
            (0.82, "#4393c3"),
            (1.00, "#2166ac"),
        ],
    )
    LOWER_IS_BETTER_CMAP.set_bad(color="#ffffff", alpha=0.0)
    HIGHER_IS_BETTER_CMAP.set_bad(color="#ffffff", alpha=0.0)
    DIFF_CMAP.set_bad(color="#ffffff", alpha=0.0)

    def __init__(self, simcast_root: Path, ecmwf_root: Path, target_dir: Path) -> None:
        if not simcast_root.exists():
            raise FileNotFoundError(f"SimCast-S2S root directory does not exist: {simcast_root}")
        if not ecmwf_root.exists():
            raise FileNotFoundError(f"ECMWF-S2S root directory does not exist: {ecmwf_root}")
        self.dataset: str = "era5"
        self.simcast_root: Path = simcast_root
        self.ecmwf_root: Path = ecmwf_root
        self.target_dir: Path = target_dir
        self.output_path: Path = self.target_dir.joinpath("fig4b.png")

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

    def _load_all_samples(self, root: Path, label: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        samples = self._load_grouped_samples(root=root)
        selected_keys = sorted(samples)[: self.MAX_GROUPS]
        if not selected_keys:
            raise ValueError(f"No {label} samples found")
        print(f"[fig4b] {label} groups: {len(selected_keys)}")
        predictions: list[torch.Tensor] = []
        groundtruths: list[torch.Tensor] = []
        for key in selected_keys:
            prediction, groundtruth = samples[key]
            predictions.append(prediction)
            groundtruths.append(groundtruth)
        return predictions, groundtruths

    @staticmethod
    def _pairwise_abs_mean(sorted_predictions: torch.Tensor) -> torch.Tensor:
        n_members = sorted_predictions.shape[0]
        weights = (2 * torch.arange(1, n_members + 1, device=sorted_predictions.device) - n_members - 1).to(
            dtype=sorted_predictions.dtype
        )
        weighted_sum = (weights[:, None, None] * sorted_predictions).sum(dim=0)
        return 2.0 * weighted_sum / float(n_members * n_members)

    @classmethod
    def _sample_crps_component_maps(
        cls,
        predictions: list[torch.Tensor],
        groundtruths: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if len(predictions) != len(groundtruths):
            raise ValueError("predictions and groundtruths must have the same number of samples")

        crps_maps: list[torch.Tensor] = []
        accuracy_maps: list[torch.Tensor] = []
        dispersion_maps: list[torch.Tensor] = []
        for prediction_members, groundtruth in zip(predictions, groundtruths):
            assert prediction_members.ndim == 3
            assert groundtruth.ndim == 2
            first_term = torch.mean(torch.abs(prediction_members - groundtruth.unsqueeze(dim=0)), dim=0)
            sorted_predictions = prediction_members.sort(dim=0).values
            second_term = 0.5 * cls._pairwise_abs_mean(sorted_predictions=sorted_predictions)
            accuracy_maps.append(first_term)
            dispersion_maps.append(second_term)
            crps_maps.append(first_term - second_term)
        return {
            "crps": torch.stack(crps_maps, dim=0),
            "accuracy": torch.stack(accuracy_maps, dim=0),
            "dispersion": torch.stack(dispersion_maps, dim=0),
        }

    @classmethod
    def _bootstrap_difference_mask(
        cls,
        simcast_values: torch.Tensor,
        ecmwf_values: torch.Tensor,
        simcast_better_when: str,
    ) -> torch.Tensor:
        assert simcast_values.shape[1:] == ecmwf_values.shape[1:]
        if simcast_better_when not in {"lower", "higher"}:
            raise ValueError("simcast_better_when must be 'lower' or 'higher'")
        n_simcast_samples = simcast_values.shape[0]
        n_ecmwf_samples = ecmwf_values.shape[0]
        generator = torch.Generator(device=simcast_values.device).manual_seed(cls.BOOTSTRAP_SEED)
        better_counts = torch.zeros_like(simcast_values[0], dtype=torch.int32)
        for _ in range(cls.BOOTSTRAP_SAMPLES):
            simcast_indices = torch.randint(
                low=0,
                high=n_simcast_samples,
                size=(n_simcast_samples,),
                device=simcast_values.device,
                generator=generator,
            )
            ecmwf_indices = torch.randint(
                low=0,
                high=n_ecmwf_samples,
                size=(n_ecmwf_samples,),
                device=simcast_values.device,
                generator=generator,
            )
            simcast_boot = simcast_values.index_select(dim=0, index=simcast_indices).mean(dim=0)
            ecmwf_boot = ecmwf_values.index_select(dim=0, index=ecmwf_indices).mean(dim=0)
            if simcast_better_when == "lower":
                better_counts += (simcast_boot < ecmwf_boot).to(dtype=torch.int32)
            else:
                better_counts += (simcast_boot > ecmwf_boot).to(dtype=torch.int32)
        return (better_counts.to(dtype=torch.float32) / cls.BOOTSTRAP_SAMPLES) >= cls.BOOTSTRAP_CONFIDENCE

    def collect(self) -> dict[str, torch.Tensor]:
        simcast_predictions, simcast_groundtruths = self._load_all_samples(
            root=self.simcast_root,
            label="SimCast-S2S",
        )
        ecmwf_predictions, ecmwf_groundtruths = self._load_all_samples(
            root=self.ecmwf_root,
            label="ECMWF-S2S",
        )
        simcast_components = self._sample_crps_component_maps(
            predictions=simcast_predictions,
            groundtruths=simcast_groundtruths,
        )
        ecmwf_components = self._sample_crps_component_maps(
            predictions=ecmwf_predictions,
            groundtruths=ecmwf_groundtruths,
        )
        print(f"[fig4b] Bootstrap samples: {self.BOOTSTRAP_SAMPLES}")
        maps: dict[str, torch.Tensor] = {}
        for component_key, simcast_values in simcast_components.items():
            ecmwf_values = ecmwf_components[component_key]
            simcast_mean = simcast_values.mean(dim=0)
            ecmwf_mean = ecmwf_values.mean(dim=0)
            maps[f"{component_key}_ecmwf"] = ecmwf_mean
            maps[f"{component_key}_simcast"] = simcast_mean
            maps[f"{component_key}_diff"] = simcast_mean - ecmwf_mean
            maps[f"{component_key}_diff_significant"] = self._bootstrap_difference_mask(
                simcast_values=simcast_values,
                ecmwf_values=ecmwf_values,
                simcast_better_when="higher" if component_key == "dispersion" else "lower",
            )
        return maps

    @staticmethod
    def _landmask_context(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        landmask = ERA5_LandmaskReader(resolution=(height, width)).tensor.cpu().numpy()
        latitudes, longitudes = (
            coord.cpu().numpy()
            for coord in ERA5_CoordinatesReader(resolution=(height, width)).tensors
        )
        return landmask, latitudes, longitudes

    @staticmethod
    def _positive_level_max(values: torch.Tensor) -> float:
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return 1.0
        return max(float(torch.quantile(finite.clamp_min(0.0), q=0.98).item()), 1e-6)

    @staticmethod
    def _symmetric_level_max(values: torch.Tensor) -> float:
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return 1.0
        return max(float(torch.quantile(finite.abs(), q=0.98).item()), 1e-6)

    def plot(self, maps: dict[str, torch.Tensor]) -> Path:
        first_map = maps["crps_ecmwf"]
        height, width = first_map.shape
        landmask, latitudes, longitudes = self._landmask_context(height=height, width=width)

        fig = plt.figure(figsize=(16.2, 12.4))
        outer_gs = fig.add_gridspec(
            nrows=3,
            ncols=1,
            height_ratios=[1.0, 1.0, 1.0],
            hspace=0.18,
        )
        central_longitude = float((longitudes.min() + longitudes.max()) / 2.0)
        projection = ccrs.Robinson(central_longitude=central_longitude)  # pyright: ignore
        data_crs = ccrs.PlateCarree()
        row_configs = [
            ("crps", "CRPS", "lower", 0.08, -0.01, 0.01),
            ("accuracy", "Error term", "lower", 0.08, -0.01, 0.01),
            ("dispersion", "Dispersion term", "higher", 0.08, -0.01, 0.01),
        ]
        column_titles = ["ECMWF-S2S", "SimCast-S2S", "SimCast-S2S - ECMWF-S2S"]
        column_suffixes = ("ecmwf", "simcast", "diff")

        for row_idx, (
            component_key,
            row_label,
            better_direction,
            fixed_data_vmax,
            fixed_diff_vmin,
            fixed_diff_vmax,
        ) in enumerate(row_configs):
            row_gs = outer_gs[row_idx].subgridspec(
                nrows=2,
                ncols=3,
                height_ratios=[1.0, 0.035],
                hspace=0.00,
                wspace=0.10,
            )
            axes = [
                fig.add_subplot(row_gs[0, 0], projection=projection),
                fig.add_subplot(row_gs[0, 1], projection=projection),
                fig.add_subplot(row_gs[0, 2], projection=projection),
            ]
            cax_data = fig.add_subplot(row_gs[1, :2])
            cax_diff = fig.add_subplot(row_gs[1, 2])
            for cax, width_scale in ((cax_data, 0.96), (cax_diff, 0.9)):
                pos = cax.get_position()
                new_width = pos.width * width_scale
                cax.set_position([pos.x0 + (pos.width - new_width) / 2.0, pos.y0, new_width, pos.height]) # pyright: ignore

            if fixed_data_vmax is None:
                data_vmax = self._positive_level_max(
                    torch.stack(
                        [maps[f"{component_key}_ecmwf"], maps[f"{component_key}_simcast"]],
                        dim=0,
                    )
                )
            else:
                data_vmax = fixed_data_vmax
            if fixed_diff_vmin is None or fixed_diff_vmax is None:
                diff_vmax_abs = self._symmetric_level_max(maps[f"{component_key}_diff"])
                diff_vmin = -diff_vmax_abs
                diff_vmax = diff_vmax_abs
            else:
                diff_vmin = fixed_diff_vmin
                diff_vmax = fixed_diff_vmax

            data_levels = np.linspace(0.0, data_vmax, 33)
            diff_levels = np.linspace(diff_vmin, diff_vmax, 33)
            data_cmap = self.HIGHER_IS_BETTER_CMAP if better_direction == "higher" else self.LOWER_IS_BETTER_CMAP
            data_mappable = None
            diff_mappable = None

            for col_idx, (ax, suffix) in enumerate(zip(axes, column_suffixes)):
                key = f"{component_key}_{suffix}"
                frame = maps[key].cpu().numpy()
                if suffix == "diff":
                    im = ax.contourf(
                        longitudes,
                        latitudes,
                        frame,
                        levels=diff_levels,
                        cmap=self.DIFF_CMAP,
                        extend="both",
                        transform=data_crs,
                    )
                    diff_mappable = im
                    stipple_mask = maps[f"{component_key}_diff_significant"].cpu().numpy().astype(bool)
                    stipple_lons, stipple_lats = np.meshgrid(
                        longitudes[:: self.STIPPLE_STRIDE],
                        latitudes[:: self.STIPPLE_STRIDE],
                    )
                    stipple_points = stipple_mask[:: self.STIPPLE_STRIDE, :: self.STIPPLE_STRIDE]
                    ax.scatter(
                        stipple_lons[stipple_points],
                        stipple_lats[stipple_points],
                        s=2,
                        c="#4b3621",
                        marker=".",
                        linewidths=0.8,
                        transform=data_crs,
                    )
                else:
                    im = ax.contourf(
                        longitudes,
                        latitudes,
                        frame,
                        levels=data_levels,
                        cmap=data_cmap,
                        extend="max",
                        transform=data_crs,
                    )
                    data_mappable = im

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
                    ax.set_title(column_titles[col_idx], fontsize=self.TITLE_FONT_SIZE, pad=12.0)
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

            assert data_mappable is not None and diff_mappable is not None
            cbar_data = fig.colorbar(data_mappable, cax=cax_data, orientation="horizontal", extendfrac=0.012)
            cbar_data.set_label(row_label, fontsize=self.COLORBAR_LABEL_FONT_SIZE)
            cbar_data.set_ticks(np.linspace(0.0, data_vmax, 5).tolist())
            cbar_data.ax.tick_params(labelsize=self.COLORBAR_TICK_FONT_SIZE)
            cbar_diff = fig.colorbar(diff_mappable, cax=cax_diff, orientation="horizontal", extendfrac=0.012)
            cbar_diff.set_label(f"{row_label} Difference", fontsize=self.COLORBAR_LABEL_FONT_SIZE)
            cbar_diff.set_ticks(np.linspace(diff_vmin, diff_vmax, 5).tolist())
            cbar_diff.ax.tick_params(labelsize=self.COLORBAR_TICK_FONT_SIZE)

        self.target_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.output_path, dpi=800, bbox_inches="tight")
        plt.close(fig)
        return self.output_path

    def run(self) -> Path:
        return self.plot(maps=self.collect())


def main() -> None:
    builder = CRPSMapFigureBuilder(
        simcast_root=SIMCAST_ROOT,
        ecmwf_root=ECMWF_ROOT,
        target_dir=TARGET_ROOT,
    )
    output_path = builder.run()
    print(f"[fig4b] Saved: {output_path}")


if __name__ == "__main__":
    main()


# python reports/fig4b.py
