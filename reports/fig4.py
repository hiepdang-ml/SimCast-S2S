import re
from functools import cached_property
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Literal

import cmocean
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kstest
from common.configs import DiffusionConfig, ECMWFS2SConfig, MetaData


class EnsembleQuantileBuilder:

    _MEMBER_PATTERN = re.compile(r"^(?P<prefix>.+)_ens_(?P<member>\d{4})\.pt$")
    KS_STAT_CMAP = cmocean.cm.turbid
    P_VALUE_CMAP = cmocean.cm.deep_r
    REJECT_CMAP_COLORS: list[str] = ["#1B9E77", "#D95F02"]

    def __init__(
        self,
        source_dir: Path,
        target_dir: Path,
        output_name: str,
    ) -> None:
        self.source_dir: Path = source_dir
        self.target_dir: Path = target_dir
        self.output_name: str = output_name
        self.target_dir.mkdir(parents=True, exist_ok=True)

    @cached_property
    def _ensemble_groups(self) -> dict[str, list[Path]]:
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

        # Outside the ensemble range: clamp to the edge quantiles.
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
        # When the truth falls inside a plateau, use the middle of the plateau rank.
        if same_value_mask.any():
            equal_to_truth: torch.Tensor = sorted_members_flat == groundtruth_flat
            first_equal: torch.Tensor = equal_to_truth.to(dtype=torch.int64).argmax(dim=1)
            last_equal: torch.Tensor = (
                n_members - 1
            ) - equal_to_truth.flip(dims=(1,)).to(dtype=torch.int64).argmax(dim=1)
            rank[same_value_mask] = ((first_equal[same_value_mask] + last_equal[same_value_mask]).to(dtype=torch.float32)) / 2.0

        rank[below_mask] = 0.0
        rank[above_mask] = float(n_members - 1)
        quantile_map: torch.Tensor = (rank / float(n_members - 1) * 100.0).reshape(H, W)
        return quantile_map

    @staticmethod
    def resize_groundtruth(
        groundtruth: torch.Tensor,
        target_shape: tuple[int, int],
    ) -> torch.Tensor:
        assert groundtruth.ndim == 2, f"groundtruth must have shape (H, W), got {tuple(groundtruth.shape)}"
        if groundtruth.shape == target_shape:
            return groundtruth

        resized: torch.Tensor = F.interpolate(
            input=groundtruth.unsqueeze(dim=0).unsqueeze(dim=0),
            size=target_shape,
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(dim=0).squeeze(dim=0)

    def build_one(self, prefix: str, member_paths: list[Path]) -> None:
        assert len(member_paths) > 0
        predictions: list[torch.Tensor] = []
        for path in member_paths:
            obj: dict[str, Any] = torch.load(path, map_location="cpu")
            groundtruth = obj["groundtruth"]
            predictions.append(obj["prediction"])

        assert groundtruth is not None
        prediction_stack: torch.Tensor = torch.stack(predictions, dim=0)
        resized_groundtruth: torch.Tensor = self.resize_groundtruth(
            groundtruth=groundtruth,
            target_shape=tuple(prediction_stack.shape[1:]), # pyright: ignore
        )
        quantile_map: torch.Tensor = self.compute_quantile_map(
            groundtruth=resized_groundtruth,
            member_predictions=prediction_stack,
        )
        result_object: dict[str, Any] = {
            "prefix": prefix,
            "quantile_map": quantile_map,
            "model_name": obj["model_name"],
            "output_name": obj["output_name"],
            "sim_id": obj["sim_id"],
            "in_startdate": obj["in_startdate"],
            "in_enddate": obj["in_enddate"],
            "out_startdate": obj["out_startdate"],
            "out_enddate": obj["out_enddate"],
        }
        output_path: Path = self.target_dir.joinpath(f"{prefix}_quantile.pt")
        torch.save(obj=result_object, f=output_path)

    def load_all_quantile_maps(self) -> torch.Tensor:
        pattern: str = (
            f"*_{self.output_name}_*_quantile.pt"
            if self.output_name is not None
            else "*_quantile.pt"
        )
        filepaths: list[Path] = sorted(self.target_dir.glob(pattern))
        if not filepaths:
            raise FileNotFoundError(f"No quantile files found in: {self.target_dir}")

        quantile_maps: list[torch.Tensor] = []
        for filepath in filepaths:
            result_object: dict[str, Any] = torch.load(filepath, map_location="cpu")
            quantile_maps.append(torch.as_tensor(result_object["quantile_map"], dtype=torch.float32))
        return torch.stack(quantile_maps, dim=0)

    @staticmethod
    def ks_test_uniform(quantile_maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert quantile_maps.ndim == 3
        n_samples, H, W = quantile_maps.shape
        values_np = (quantile_maps / 100.0).clamp(min=0.0, max=1.0).reshape(n_samples, H * W).cpu().numpy()
        d_stats: list[float] = []
        p_values: list[float] = []
        for pixel_idx in range(H * W):
            result = kstest(rvs=values_np[:, pixel_idx], cdf="uniform")
            d_stats.append(float(result.statistic))
            p_values.append(float(result.pvalue))
        d_stat = torch.tensor(d_stats, dtype=torch.float32).reshape(H, W)
        p_value = torch.tensor(p_values, dtype=torch.float32).reshape(H, W)
        return d_stat, p_value

    def plot_uniformity_test(self, alpha: float = 0.05) -> tuple[Path, Path]:
        assert 0. < alpha < 1.
        quantile_maps: torch.Tensor = self.load_all_quantile_maps()
        d_stat, p_value = self.ks_test_uniform(quantile_maps=quantile_maps)
        reject_map: torch.Tensor = (p_value < alpha).to(dtype=torch.float32)
        stats_path: Path = self.target_dir.joinpath(f"{self.output_name}_uniformity_ks.pt")
        torch.save(
            {
                "output_name": self.output_name,
                "n_samples": int(quantile_maps.shape[0]),
                "alpha": alpha,
                "ks_d_stat": d_stat,
                "ks_p_value": p_value,
                "reject_map": reject_map,
            },
            stats_path,
        )
        fig, axs = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
        y = np.arange(d_stat.shape[0], dtype=np.float64)
        x = np.arange(d_stat.shape[1], dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        levels_d = np.linspace(0.0, 0.8, 17)
        levels_p = np.linspace(0.0, 0.2, 17)
        im0 = axs[0].contourf(xx, yy, d_stat, levels=levels_d, cmap=self.KS_STAT_CMAP, extend="max")
        axs[0].set_title("KS D Statistic", fontsize=10)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        cax0 = axs[0].inset_axes([0.0, -0.18, 1.0, 0.06])
        fig.colorbar(im0, cax=cax0, orientation="horizontal")
        im1 = axs[1].contourf(xx, yy, p_value, levels=levels_p, cmap=self.P_VALUE_CMAP, extend="max")
        axs[1].set_title("p-value", fontsize=10)
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        cax1 = axs[1].inset_axes([0.0, -0.18, 1.0, 0.06])
        fig.colorbar(im1, cax=cax1, orientation="horizontal")
        reject_cmap = ListedColormap(self.REJECT_CMAP_COLORS)
        reject_norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=reject_cmap.N)
        im2 = axs[2].imshow(reject_map, origin="lower", cmap=reject_cmap, norm=reject_norm)
        axs[2].set_title(f"Reject Uniformity (alpha={alpha})", fontsize=10)
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        cax2 = axs[2].inset_axes([0.0, -0.18, 1.0, 0.06])
        cbar = fig.colorbar(im2, cax=cax2, orientation="horizontal", ticks=[0.25, 0.75])
        cbar.ax.set_xticklabels(["Cannot Reject", "Reject"])
        fig.suptitle("KS Uniformity Test of Quantile Maps", fontsize=16)
        figure_path: Path = self.target_dir.joinpath(f"{self.output_name}_uniformity_ks.png")
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return stats_path, figure_path

    def plot_pit_histogram(self, n_bins: int = 20) -> Path:
        assert n_bins > 1
        quantile_maps: torch.Tensor = self.load_all_quantile_maps()
        pit_values: torch.Tensor = (quantile_maps.reshape(-1) / 100.0).clamp(min=0.0, max=1.0)
        fig, ax = plt.subplots(1, 1, figsize=(6, 3.2))
        ax.hist(
            pit_values.cpu().numpy(),
            bins=n_bins,
            range=(0.0, 1.0),
            density=True,
            color="#4C78A8",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85,
        )
        ax.axhline(y=1.0, color="#D62728", linestyle="--", linewidth=1.5, label="Uniform density")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 5.0)
        bin_width: float = 1.0 / n_bins
        tick_positions: list[float] = [i * bin_width for i in range(n_bins + 1)]
        tick_labels: list[str] = [f"{i * bin_width:.2f}" for i in range(n_bins + 1)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="center")
        ax.set_xlabel("PIT")
        ax.set_ylabel("Density")
        ax.set_title("PIT Histogram")
        ax.legend(frameon=False)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()

        output_path: Path = self.target_dir.joinpath(f"{self.output_name}_pit_histogram.png")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def run(self) -> None:
        groups: dict[str, list[Path]] = self._ensemble_groups
        for prefix in sorted(self._ensemble_groups.keys()):
            self.build_one(prefix=prefix, member_paths=groups[prefix])

        stats_path, figure_path = self.plot_uniformity_test()
        pit_histogram_path: Path = self.plot_pit_histogram()
        print(f"[fig4] Saved uniformity stats to: {stats_path}")
        print(f"[fig4] Saved uniformity figure to: {figure_path}")
        print(f"[fig4] Saved PIT histogram to: {pit_histogram_path}")


class EnsembleCRPSDecompositionBuilder:

    _MEMBER_PATTERN = re.compile(r"^(?P<prefix>.+)_ens_(?P<member>\d{4})\.pt$")
    CRPS_CMAP = cmocean.cm.rain

    def __init__(
        self,
        source_dir: Path,
        target_dir: Path,
        output_name: str,
    ) -> None:
        self.source_dir: Path = source_dir
        self.target_dir: Path = target_dir
        self.output_name: str = output_name
        self.target_dir.mkdir(parents=True, exist_ok=True)

    @cached_property
    def _ensemble_groups(self) -> dict[str, list[Path]]:
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
        return groups

    @staticmethod
    def compute_crps_terms(
        groundtruth: torch.Tensor,
        member_predictions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert groundtruth.ndim == 2
        assert member_predictions.ndim == 3
        assert groundtruth.shape == member_predictions.shape[1:]

        error_term: torch.Tensor = (member_predictions - groundtruth.unsqueeze(dim=0)).abs().mean(dim=0)
        pairwise_distance: torch.Tensor = (
            member_predictions[:, None, :, :] - member_predictions[None, :, :, :]
        ).abs()
        spread_term: torch.Tensor = 0.5 * pairwise_distance.mean(dim=(0, 1))
        crps_map: torch.Tensor = error_term - spread_term
        return error_term, spread_term, crps_map

    def build_one(self, prefix: str, member_paths: list[Path]) -> Path:
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

        assert last_object is not None
        assert groundtruth is not None
        prediction_stack: torch.Tensor = torch.stack(predictions, dim=0)
        resized_groundtruth: torch.Tensor = EnsembleQuantileBuilder.resize_groundtruth(
            groundtruth=groundtruth,
            target_shape=tuple(prediction_stack.shape[1:]),   # pyright: ignore
        )
        error_term, spread_term, crps_map = self.compute_crps_terms(
            groundtruth=resized_groundtruth,
            member_predictions=prediction_stack,
        )
        result_object: dict[str, Any] = {
            "prefix": prefix,
            "groundtruth": resized_groundtruth,
            "ensemble_mean": prediction_stack.mean(dim=0),
            "error_term": error_term,
            "spread_term": spread_term,
            "crps_map": crps_map,
            "ensemble_size": int(prediction_stack.shape[0]),
            "model_name": last_object["model_name"],
            "output_name": last_object["output_name"],
            "sim_id": last_object["sim_id"],
            "in_startdate": last_object["in_startdate"],
            "in_enddate": last_object["in_enddate"],
            "out_startdate": last_object["out_startdate"],
            "out_enddate": last_object["out_enddate"],
        }
        output_path: Path = self.target_dir.joinpath(f"{prefix}_crps.pt")
        torch.save(result_object, output_path)
        return output_path

    def load_all_crps_maps(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pattern: str = f"*_{self.output_name}_*_crps.pt"
        filepaths: list[Path] = sorted(self.target_dir.glob(pattern))
        if not filepaths:
            raise FileNotFoundError(f"No CRPS files found in: {self.target_dir}")

        error_terms: list[torch.Tensor] = []
        spread_terms: list[torch.Tensor] = []
        crps_maps: list[torch.Tensor] = []
        for filepath in filepaths:
            result_object: dict[str, Any] = torch.load(filepath, map_location="cpu")
            error_terms.append(torch.as_tensor(result_object["error_term"], dtype=torch.float32))
            spread_terms.append(torch.as_tensor(result_object["spread_term"], dtype=torch.float32))
            crps_maps.append(torch.as_tensor(result_object["crps_map"], dtype=torch.float32))
        return (
            torch.stack(error_terms, dim=0),
            torch.stack(spread_terms, dim=0),
            torch.stack(crps_maps, dim=0),
        )

    def plot_summary(self) -> tuple[Path, Path]:
        error_terms, spread_terms, crps_maps = self.load_all_crps_maps()
        mean_error_term: torch.Tensor = error_terms.mean(dim=0)
        mean_spread_term: torch.Tensor = spread_terms.mean(dim=0)
        mean_crps: torch.Tensor = crps_maps.mean(dim=0)
        avg_error_term: float = float(mean_error_term.mean().item())
        avg_spread_term: float = float(mean_spread_term.mean().item())
        avg_crps: float = float(mean_crps.mean().item())

        stats_path: Path = self.target_dir.joinpath(f"{self.output_name}_crps_decomposition.pt")
        torch.save(
            {
                "output_name": self.output_name,
                "n_samples": int(crps_maps.shape[0]),
                "mean_error_term": mean_error_term,
                "mean_spread_term": mean_spread_term,
                "mean_crps": mean_crps,
                "avg_error_term": avg_error_term,
                "avg_spread_term": avg_spread_term,
                "avg_crps": avg_crps,
            },
            stats_path,
        )

        fig, axs = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
        titles: list[str] = ["Error Term", "Spread Term", "CRPS"]
        frames: list[torch.Tensor] = [mean_error_term, mean_spread_term, mean_crps]
        y = np.arange(mean_crps.shape[0], dtype=np.float64)
        x = np.arange(mean_crps.shape[1], dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        levels = np.linspace(0.0, 0.04, 17)
        im = None
        for ax, title, frame in zip(axs, titles, frames):
            im = ax.contourf(xx, yy, frame, levels=levels, cmap=self.CRPS_CMAP, extend="max")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        assert im is not None
        fig.canvas.draw()
        middle_bbox = axs[1].get_position()
        subplot_width: float = middle_bbox.width
        colorbar_width: float = 2.0 * subplot_width
        colorbar_left: float = middle_bbox.x0 + 0.5 * subplot_width - 0.5 * colorbar_width
        colorbar_bottom: float = middle_bbox.y0 - 0.10
        cax = fig.add_axes((colorbar_left, colorbar_bottom, colorbar_width, 0.025))
        fig.colorbar(im, cax=cax, orientation="horizontal")

        fig.suptitle("CRPS Decomposition", fontsize=16, y=1.01)
        figure_path: Path = self.target_dir.joinpath(f"{self.output_name}_crps_decomposition.png")
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return stats_path, figure_path

    def run(self) -> None:
        groups: dict[str, list[Path]] = self._ensemble_groups
        for prefix in sorted(groups.keys()):
            self.build_one(prefix=prefix, member_paths=groups[prefix])
        stats_path, figure_path = self.plot_summary()
        stats_object: dict[str, Any] = torch.load(stats_path, map_location="cpu")
        print(
            "[fig4] CRPS decomposition averages:\n"
            f"error_term: {stats_object['avg_error_term']:.3f}\n"
            f"spread_term: {stats_object['avg_spread_term']:.3f}\n"
            f"crps: {stats_object['avg_crps']:.3f}\n"
        )
        print(f"[fig4] Saved CRPS decomposition stats to: {stats_path}")
        print(f"[fig4] Saved CRPS decomposition figure to: {figure_path}")


def main(
    model: Literal["diffusion", "ecmwf-s2s"],
    dataset: Literal["cesm2", "era5"],
) -> None:
    if model == "diffusion":
        config: DiffusionConfig = DiffusionConfig()
    else:
        config: ECMWFS2SConfig = ECMWFS2SConfig()
    metadata: MetaData = MetaData(dataset_name=dataset, tp="test")
    source_dir: Path = config.target_path.joinpath(f"{dataset}/tensors")
    target_dir: Path = config.target_path.joinpath(f"{dataset}/quantiles")
    for output_name in metadata.output_vars:
        quantile_builder = EnsembleQuantileBuilder(
            source_dir=source_dir,
            target_dir=target_dir,
            output_name=output_name,
        )
        quantile_builder.run()
        crps_builder = EnsembleCRPSDecompositionBuilder(
            source_dir=source_dir,
            target_dir=target_dir,
            output_name=output_name,
        )
        crps_builder.run()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, choices=["diffusion", "ecmwf-s2s"], required=True)
    parser.add_argument("--dataset", type=str, choices=["cesm2", "era5"], required=True)
    args: Namespace = parser.parse_args()
    main(model=args.model, dataset=args.dataset)
