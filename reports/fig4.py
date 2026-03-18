import re
from functools import cached_property
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn.functional as F
from scipy.stats import kstest
from common.configs import DiffusionConfig, MetaData


class DiffusionQuantileBuilder:

    _MEMBER_PATTERN = re.compile(r"^(?P<prefix>.+)_ens_(?P<member>\d{4})\.pt$")

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

    def plot_all(self) -> None:
        pattern: str = (
            f"*_{self.output_name}_*_quantile.pt"
            if self.output_name is not None
            else "*_quantile.pt"
        )
        filepaths: list[Path] = sorted(self.target_dir.glob(pattern))
        for filepath in filepaths:
            result_object: dict[str, Any] = torch.load(filepath, map_location="cpu")
            quantile_map: torch.Tensor = result_object["quantile_map"]
            output_name: str = result_object["output_name"]
            prefix: str = result_object["prefix"]
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))
            im = ax.imshow(quantile_map, origin="lower", cmap="viridis", vmin=0.0, vmax=100.0)
            ax.set_title("Truth Quantile")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(f"{output_name}: {prefix}", fontsize=12)
            fig.tight_layout()
            fig.savefig(fname=filepath.with_suffix(".png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

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
        fig, axs = plt.subplots(1, 3, figsize=(18, 4.2))
        im0 = axs[0].imshow(d_stat, origin="lower", cmap="magma")
        axs[0].set_title("KS D Statistic")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        cax0 = make_axes_locatable(axs[0]).append_axes("right", size="4%", pad=0.08)
        fig.colorbar(im0, cax=cax0)
        im1 = axs[1].imshow(p_value, origin="lower", cmap="viridis", vmin=0.0, vmax=0.2)
        axs[1].set_title("p-value")
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        cax1 = make_axes_locatable(axs[1]).append_axes("right", size="4%", pad=0.08)
        fig.colorbar(im1, cax=cax1)
        reject_cmap = ListedColormap(["#9ecae1", "#ef8a62"])
        reject_norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=reject_cmap.N)
        im2 = axs[2].imshow(reject_map, origin="lower", cmap=reject_cmap, norm=reject_norm)
        axs[2].set_title(f"Reject Uniformity (alpha={alpha})")
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        cax2 = make_axes_locatable(axs[2]).append_axes("right", size="4%", pad=0.08)
        cbar = fig.colorbar(im2, cax=cax2, ticks=[0.1, 0.9])
        cbar.ax.set_yticklabels(["Cannot\nReject\nUniform", "Reject\nUniform"])
        fig.suptitle(f"KS Uniformity Test of Quantile Maps: {self.output_name}", fontsize=16)
        fig.tight_layout()
        figure_path: Path = self.target_dir.joinpath(f"{self.output_name}_uniformity_ks.png")
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return stats_path, figure_path

    def plot_pit_histogram(self, n_bins: int = 20) -> Path:
        assert n_bins > 1
        quantile_maps: torch.Tensor = self.load_all_quantile_maps()
        pit_values: torch.Tensor = (quantile_maps.reshape(-1) / 100.0).clamp(min=0.0, max=1.0)
        mean_pit: float = float(pit_values.mean())

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
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
        ax.axvline(x=mean_pit, color="#111827", linestyle="-", linewidth=1.2, label=f"Mean PIT = {mean_pit:.3f}")
        ax.set_xlim(0.0, 1.0)
        bin_width: float = 1.0 / n_bins
        tick_positions: list[float] = [i * bin_width for i in range(n_bins + 1)]
        tick_labels: list[str] = [f"{i * bin_width:.2f}" for i in range(n_bins + 1)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="center")
        ax.set_xlabel("PIT")
        ax.set_ylabel("Density")
        ax.set_title(f"PIT Histogram: {self.output_name}")
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
        self.plot_all()
        stats_path, figure_path = self.plot_uniformity_test()
        pit_histogram_path: Path = self.plot_pit_histogram()
        print(f"[fig4] Saved uniformity stats to: {stats_path}")
        print(f"[fig4] Saved uniformity figure to: {figure_path}")
        print(f"[fig4] Saved PIT histogram to: {pit_histogram_path}")



def main(
    dataset: Literal["cesm2", "era5"],
) -> None:
    config: DiffusionConfig = DiffusionConfig()
    metadata: MetaData = MetaData(dataset_name=dataset, tp="test")
    source_dir: Path = config.target_path.joinpath(f"{dataset}/tensors")
    target_dir: Path = config.target_path.joinpath(f"{dataset}/quantiles")
    for output_name in metadata.output_vars:
        builder = DiffusionQuantileBuilder(
            source_dir=source_dir,
            target_dir=target_dir,
            output_name=output_name,
        )
        builder.run()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["cesm2", "era5"], required=True)
    args: Namespace = parser.parse_args()
    main(dataset=args.dataset)
