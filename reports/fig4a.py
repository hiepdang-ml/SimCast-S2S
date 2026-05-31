import re
import shutil
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.configs import MetaData


TARGET_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/")
TEMP_ROOT: Path = TARGET_ROOT.joinpath("tmp_fig4a")
DATASET: str = "era5"


@dataclass(frozen=True)
class ModelSpec:
    label: str
    slug: str
    root: Path


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        label="SimCast-S2S",
        slug="simcast_s2s",
        root=TARGET_ROOT.joinpath("finetune/diffusion_v23_cosine_eta000_100steps_100members_guidancescale100"),
    ),
    ModelSpec(
        label="ECMWF-S2S",
        slug="ecmwf_s2s",
        root=TARGET_ROOT.joinpath("ecmwfs2s_28"),
    ),
)


class EnsembleQuantileBuilder:

    _MEMBER_PATTERN = re.compile(r"^(?P<prefix>.+)_ens_(?P<member>\d{4})\.pt$")
    MAX_GROUPS: int = 70
    TROPICAL_LATITUDE_LIMIT: float = 25.0

    def __init__(
        self,
        model: ModelSpec,
        output_name: str,
        target_dir: Path,
    ) -> None:
        self.model: ModelSpec = model
        self.output_name: str = output_name
        self.source_dir: Path = model.root.joinpath(DATASET, "tensors")
        self.target_dir: Path = target_dir
        self.target_dir.mkdir(parents=True, exist_ok=True)

    @cached_property
    def _ensemble_groups(self) -> dict[str, list[Path]]:
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Missing tensor directory: {self.source_dir}")

        groups: dict[str, list[Path]] = {}
        for path in sorted(self.source_dir.glob("*_ens_*.pt")):
            if path.name.endswith("_ens_aggregate.pt"):
                continue
            match: re.Match | None = self._MEMBER_PATTERN.match(path.name)
            if match is None:
                continue
            prefix: str = match.group("prefix")
            if f"_{self.output_name}_" not in prefix:
                continue
            groups.setdefault(prefix, []).append(path)

        for prefix in groups:
            groups[prefix].sort(key=lambda x: x.name)
        if not groups:
            raise FileNotFoundError(f"No ensemble member files found for {self.model.label} {self.output_name}")
        return groups

    @staticmethod
    def compute_quantile_map(
        groundtruth: torch.Tensor,
        member_predictions: torch.Tensor,
    ) -> torch.Tensor:
        assert groundtruth.ndim == 2
        assert member_predictions.ndim == 3
        assert groundtruth.shape == member_predictions.shape[1:]
        n_members, H, W = member_predictions.shape
        if n_members == 1:
            return torch.full_like(groundtruth, fill_value=50.0, dtype=torch.float32)

        sorted_members: torch.Tensor = member_predictions.sort(dim=0).values
        sorted_members_flat: torch.Tensor = sorted_members.permute(1, 2, 0).reshape(H * W, n_members).contiguous()
        groundtruth_flat: torch.Tensor = groundtruth.reshape(H * W, 1).contiguous()
        lower_idx: torch.Tensor = torch.searchsorted(
            sorted_sequence=sorted_members_flat,
            input=groundtruth_flat,
            right=True,
        ).squeeze(dim=-1).to(dtype=torch.long)

        below_mask: torch.Tensor = lower_idx == 0
        above_mask: torch.Tensor = lower_idx == n_members

        lower_idx = lower_idx.clamp(min=1, max=n_members - 1)
        left_index: torch.Tensor = lower_idx - 1
        right_index: torch.Tensor = lower_idx

        left_value: torch.Tensor = torch.gather(
            sorted_members_flat, dim=1, index=left_index.unsqueeze(dim=1)
        ).squeeze(dim=1)
        right_value: torch.Tensor = torch.gather(
            sorted_members_flat, dim=1, index=right_index.unsqueeze(dim=1)
        ).squeeze(dim=1)

        denom: torch.Tensor = right_value - left_value
        same_value_mask: torch.Tensor = denom == 0
        truth_values: torch.Tensor = groundtruth_flat.squeeze(dim=1)
        frac: torch.Tensor = torch.zeros_like(truth_values, dtype=torch.float32)
        frac[~same_value_mask] = (
            (truth_values[~same_value_mask] - left_value[~same_value_mask]) / denom[~same_value_mask]
        ).to(dtype=torch.float32)
        frac = frac.clamp(min=0.0, max=1.0)

        rank: torch.Tensor = left_index.to(dtype=torch.float32) + frac
        if same_value_mask.any():
            equal_to_truth: torch.Tensor = sorted_members_flat == groundtruth_flat
            first_equal: torch.Tensor = equal_to_truth.to(dtype=torch.int64).argmax(dim=1)
            last_equal: torch.Tensor = (
                n_members - 1
            ) - equal_to_truth.flip(dims=(1,)).to(dtype=torch.int64).argmax(dim=1)
            rank[same_value_mask] = (
                (first_equal[same_value_mask] + last_equal[same_value_mask]).to(dtype=torch.float32)
            ) / 2.0

        rank[below_mask] = 0.0
        rank[above_mask] = float(n_members - 1)
        quantile_map: torch.Tensor = (rank / float(n_members - 1) * 100.0).reshape(H, W)
        return quantile_map

    def _quantile_path(self, prefix: str) -> Path:
        return self.target_dir.joinpath(f"fig4a_{self.model.slug}_{prefix}_quantile.pt")

    def build_one(self, prefix: str, member_paths: list[Path]) -> None:
        assert len(member_paths) > 0
        predictions: list[torch.Tensor] = []
        groundtruth: torch.Tensor | None = None
        last_object: dict[str, Any] | None = None
        for path in member_paths:
            obj: dict[str, Any] = torch.load(path, map_location="cpu")
            last_object = obj
            if groundtruth is None:
                groundtruth = torch.as_tensor(obj["groundtruth"], dtype=torch.float32)
            predictions.append(torch.as_tensor(obj["prediction"], dtype=torch.float32))

        assert groundtruth is not None
        assert last_object is not None
        prediction_stack: torch.Tensor = torch.stack(predictions, dim=0)
        quantile_map: torch.Tensor = self.compute_quantile_map(
            groundtruth=groundtruth,
            member_predictions=prediction_stack,
        )
        result_object: dict[str, Any] = {
            "prefix": prefix,
            "model_label": self.model.label,
            "model_slug": self.model.slug,
            "model_root": self.model.root.as_posix(),
            "dataset": DATASET,
            "quantile_map": quantile_map,
            "output_name": last_object["output_name"],
            "sim_id": last_object["sim_id"],
            "in_startdate": last_object["in_startdate"],
            "in_enddate": last_object["in_enddate"],
            "out_startdate": last_object["out_startdate"],
            "out_enddate": last_object["out_enddate"],
        }
        torch.save(obj=result_object, f=self._quantile_path(prefix=prefix))

    def build_quantile_maps(self, selected_prefixes: list[str] | None = None) -> None:
        groups: dict[str, list[Path]] = self._ensemble_groups
        prefixes = selected_prefixes if selected_prefixes is not None else sorted(groups.keys())[:self.MAX_GROUPS]
        for prefix in prefixes:
            self.build_one(prefix=prefix, member_paths=groups[prefix])

    def load_all_quantile_maps(self, selected_prefixes: list[str] | None = None) -> torch.Tensor:
        pattern: str = f"fig4a_{self.model.slug}_*_{self.output_name}_*_quantile.pt"
        if selected_prefixes is None:
            filepaths: list[Path] = sorted(self.target_dir.glob(pattern))[:self.MAX_GROUPS]
        else:
            filepaths = [self._quantile_path(prefix=prefix) for prefix in selected_prefixes]
        if not filepaths:
            raise FileNotFoundError(f"No quantile files found for {self.model.label} in: {self.target_dir}")

        quantile_maps: list[torch.Tensor] = []
        for filepath in filepaths:
            result_object: dict[str, Any] = torch.load(filepath, map_location="cpu")
            quantile_maps.append(torch.as_tensor(result_object["quantile_map"], dtype=torch.float32))
        return torch.stack(quantile_maps, dim=0)

    def collect(self, selected_prefixes: list[str] | None = None) -> dict[str, Any]:
        quantile_maps: torch.Tensor = self.load_all_quantile_maps(selected_prefixes=selected_prefixes)
        pit_values: torch.Tensor = (quantile_maps.reshape(-1) / 100.0).clamp(min=0.0, max=1.0)
        region_pit_values = self._region_pit_values(quantile_maps=quantile_maps)
        return {
            "model_label": self.model.label,
            "model_slug": self.model.slug,
            "output_name": self.output_name,
            "n_samples": int(quantile_maps.shape[0]),
            "pit_values": pit_values,
            "region_pit_values": region_pit_values,
        }

    @classmethod
    def _region_pit_values(cls, quantile_maps: torch.Tensor) -> dict[str, torch.Tensor]:
        assert quantile_maps.ndim == 3
        height = quantile_maps.shape[-2]
        latitudes = torch.linspace(-90.0, 90.0, steps=height, dtype=torch.float32)
        tropical_mask = latitudes.abs() <= cls.TROPICAL_LATITUDE_LIMIT
        extratropical_mask = ~tropical_mask
        normalized = (quantile_maps / 100.0).clamp(min=0.0, max=1.0)
        return {
            "tropical": normalized[:, tropical_mask, :].reshape(-1),
            "extratropical": normalized[:, extratropical_mask, :].reshape(-1),
        }


class Fig4aBuilder:

    Y_AXIS_MAX: float = 4.0

    def __init__(
        self,
        models: tuple[ModelSpec, ...],
        target_dir: Path,
        temp_dir: Path,
        dataset: str,
    ) -> None:
        self.models: tuple[ModelSpec, ...] = models
        self.target_dir: Path = target_dir
        self.temp_dir: Path = temp_dir
        self.dataset: str = dataset
        self.metadata: MetaData = MetaData(dataset_name="era5", tp="test")
        if dataset != "era5":
            raise ValueError(f"fig4a is ERA5-only, got {dataset}")
        self.target_dir.mkdir(parents=True, exist_ok=True)
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def collect(self, output_name: str) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        builders = [
            EnsembleQuantileBuilder(
                model=model,
                output_name=output_name,
                target_dir=self.temp_dir,
            )
            for model in self.models
        ]
        for builder in builders:
            model = builder.model
            builder.build_quantile_maps()
            summary = builder.collect()
            summaries.append(summary)
            stats_path = self.temp_dir.joinpath(f"fig4a_{model.slug}_{output_name}_pit.pt")
            torch.save(summary, stats_path)
            print(f"[fig4a] Saved {model.label} PIT stats to: {stats_path}")
        return summaries

    @staticmethod
    def _pit_histogram_ratio(pit_percentiles: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray, float]:
        counts, edges = np.histogram(pit_percentiles, bins=n_bins, range=(0.0, 100.0))
        expected_count = counts.sum() / float(n_bins)
        if expected_count == 0.0:
            return np.zeros_like(counts, dtype=np.float64), edges, float("nan")
        ratio = counts.astype(np.float64) / expected_count
        probabilities = counts.astype(np.float64) / float(counts.sum())
        expected_probability = 1.0 / float(n_bins)
        chi2_distance = float(np.sum((probabilities - expected_probability) ** 2 / expected_probability))
        return ratio, edges, chi2_distance

    def plot_pit_grid(self, output_name: str, summaries: list[dict[str, Any]], n_bins: int = 20) -> Path:
        assert n_bins > 1
        region_specs = (
            ("global", "Global"),
            ("tropical", "Tropical"),
            ("extratropical", "Extratropical"),
        )
        fig, axs = plt.subplots(
            len(region_specs),
            len(summaries),
            figsize=(8.6, 6.2),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axs = np.asarray(axs).reshape(len(region_specs), len(summaries))
        for col_idx, summary in enumerate(summaries):
            region_pit_values: dict[str, torch.Tensor] = summary["region_pit_values"]
            for row_idx, (region_key, region_label) in enumerate(region_specs):
                ax = axs[row_idx, col_idx]
                pit_values = summary["pit_values"] if region_key == "global" else region_pit_values[region_key]
                pit_percentiles = (pit_values.cpu().numpy() * 100.0).clip(min=0.0, max=100.0)
                ratio, edges, chi2_distance = self._pit_histogram_ratio(
                    pit_percentiles=pit_percentiles,
                    n_bins=n_bins,
                )
                ax.bar(
                    edges[:-1],
                    ratio,
                    width=np.diff(edges),
                    align="edge",
                    color="#4C78A8",
                    edgecolor="black",
                    linewidth=0.7,
                    alpha=0.85,
                )
                ax.axhline(y=1.0, color="#C43C39", linestyle="--", linewidth=1.2)
                ax.text(
                    0.50,
                    0.80,
                    rf"$\chi^2$ distance = {chi2_distance:.2f}",
                    ha="center",
                    va="top",
                    fontsize=9,
                    transform=ax.transAxes,
                )
                ax.set_xlim(0.0, 100.0)
                ax.set_ylim(0.0, self.Y_AXIS_MAX)
                ax.set_xticks([0, 25, 50, 75, 100])
                ax.grid(axis="y", linestyle="--", alpha=0.3)
                if row_idx == 0:
                    ax.set_title(str(summary["model_label"]), fontsize=12)
                if col_idx == 0:
                    ax.set_ylabel(region_label, fontsize=12)
        fig.supxlabel("Percentile", fontsize=12)
        output_path = self.target_dir.joinpath(f"fig4a_{output_name}_pit_histogram.png")
        fig.savefig(output_path, dpi=500, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def run(self) -> None:
        for output_name in self.metadata.output_vars:
            summaries = self.collect(output_name=output_name)
            pit_path = self.plot_pit_grid(output_name=output_name, summaries=summaries)
            print(f"[fig4a] Saved PIT figure to: {pit_path}")


def main() -> None:
    builder = Fig4aBuilder(
        models=MODEL_SPECS,
        target_dir=TARGET_ROOT,
        temp_dir=TEMP_ROOT,
        dataset=DATASET,
    )
    builder.run()


if __name__ == "__main__":
    main()
