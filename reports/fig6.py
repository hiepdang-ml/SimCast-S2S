from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch


DDIM_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/finetune/diffusion_v23_cosine_eta000_rank64/")
DDPM_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/finetune/diffusion_v23_cosine_eta100_rank64/")
ECMWF_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/ecmwfs2s/")
TARGET_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/")

class SpatialAutocorrelationFigureBuilder:

    MODEL_SPECS: list[tuple[str, Any]] = [
        ("Observed", "#ffffff", "#000000"),
        ("DDIM", "#8ec1da", "#000000"),
        ("DDPM", "#3b6e8f", "#000000"),
        ("ECMWF-S2S", "#c44e52", "#000000"),
    ]
    DIRECTIONS: list[tuple[str, str]] = [
        ("horizontal", "(a) Horizontal"),
        ("vertical", "(b) Vertical"),
        ("diag_main", "(c) Main diagonal"),
        ("diag_anti", "(d) Anti-diagonal"),
    ]

    def __init__(
        self,
        ddim_root: Path,
        ddpm_root: Path,
        ecmwf_root: Path,
        target_dir: Path,
        max_shift: int,
    ) -> None:
        if not ddim_root.exists():
            raise FileNotFoundError(f"DDIM root directory does not exist: {ddim_root}")
        if not ddpm_root.exists():
            raise FileNotFoundError(f"DDPM root directory does not exist: {ddpm_root}")
        if not ecmwf_root.exists():
            raise FileNotFoundError(f"ECMWF root directory does not exist: {ecmwf_root}")
        if max_shift <= 0:
            raise ValueError(f"max_shift must be > 0, got {max_shift}")

        # fig6.py only supports because ECMWF-S2S is ERA5-only
        self.dataset: str = "era5"
        self.ddim_root: Path = ddim_root
        self.ddpm_root: Path = ddpm_root
        self.ecmwf_root: Path = ecmwf_root
        self.target_dir: Path = target_dir
        self.max_shift: int = max_shift
        self.output_path: Path = self.target_dir.joinpath("fig6.png")

    @staticmethod
    def _sample_key(payload: dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(payload["in_startdate"]),
            str(payload["in_enddate"]),
            str(payload["out_startdate"]),
            str(payload["out_enddate"]),
        )

    @staticmethod
    def _autocorrelation(x: torch.Tensor, y: torch.Tensor) -> float:
        x_flat = x.reshape(-1).to(dtype=torch.float32)
        y_flat = y.reshape(-1).to(dtype=torch.float32)
        x_centered = x_flat - x_flat.mean()
        y_centered = y_flat - y_flat.mean()
        numerator = torch.sum(x_centered * y_centered)
        denominator = torch.sqrt(torch.sum(x_centered ** 2) * torch.sum(y_centered ** 2))
        return float((numerator / denominator).item())

    @staticmethod
    def _shifted_pair(field: torch.Tensor, direction: str, shift: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert field.ndim == 2
        if direction == "horizontal":
            return field[:, :-shift], field[:, shift:]
        if direction == "vertical":
            return field[:-shift, :], field[shift:, :]
        if direction == "diag_main":
            return field[:-shift, :-shift], field[shift:, shift:]
        if direction == "diag_anti":
            return field[:-shift, shift:], field[shift:, :-shift]
        raise ValueError(f"Unsupported direction: {direction}")

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

    def collect(self) -> dict[str, dict[str, list[list[float]]]]:
        ddim_payloads = self._load_member_payloads(root=self.ddim_root)
        ddpm_payloads = self._load_member_payloads(root=self.ddpm_root)
        ecmwf_payloads = self._load_member_payloads(root=self.ecmwf_root)

        correlations: dict[str, dict[str, list[list[float]]]] = {
            model_name: {direction: [] for direction, _ in self.DIRECTIONS}
            for model_name, _, _ in self.MODEL_SPECS
        }

        observed_cache: set[tuple[tuple[str, str, str, str], str, int]] = set()
        for shift in range(1, self.max_shift + 1):
            for direction, _ in self.DIRECTIONS:
                observed_values: list[float] = []
                for payload in ddim_payloads + ddpm_payloads + ecmwf_payloads:
                    key = self._sample_key(payload)
                    observed_key = (key, direction, shift)
                    if observed_key in observed_cache:
                        continue
                    observed_cache.add(observed_key)
                    groundtruth = torch.as_tensor(payload["groundtruth"], dtype=torch.float32)
                    left, right = self._shifted_pair(groundtruth, direction=direction, shift=shift)
                    observed_values.append(self._autocorrelation(left, right))
                correlations["Observed"][direction].append(observed_values)

                ddim_values: list[float] = []
                for payload in ddim_payloads:
                    prediction = torch.as_tensor(payload["prediction"], dtype=torch.float32)
                    left, right = self._shifted_pair(prediction, direction=direction, shift=shift)
                    ddim_values.append(self._autocorrelation(left, right))
                correlations["DDIM"][direction].append(ddim_values)

                ddpm_values: list[float] = []
                for payload in ddpm_payloads:
                    prediction = torch.as_tensor(payload["prediction"], dtype=torch.float32)
                    left, right = self._shifted_pair(prediction, direction=direction, shift=shift)
                    ddpm_values.append(self._autocorrelation(left, right))
                correlations["DDPM"][direction].append(ddpm_values)

                ecmwf_values: list[float] = []
                for payload in ecmwf_payloads:
                    prediction = torch.as_tensor(payload["prediction"], dtype=torch.float32)
                    left, right = self._shifted_pair(prediction, direction=direction, shift=shift)
                    ecmwf_values.append(self._autocorrelation(left, right))
                correlations["ECMWF-S2S"][direction].append(ecmwf_values)

        return correlations

    def plot(self, correlations: dict[str, dict[str, list[list[float]]]]) -> Path:
        fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
        axes = axs.ravel().tolist()
        shifts = np.arange(1, self.max_shift + 1)
        box_width = 0.18
        offsets = [-1.5 * box_width, -0.5 * box_width, 0.5 * box_width, 1.5 * box_width]

        for ax, (direction, title) in zip(axes, self.DIRECTIONS):
            for offset, (model_name, fcolor, ecolor) in zip(offsets, self.MODEL_SPECS):
                values = correlations[model_name][direction]
                ax.boxplot(
                    values,
                    positions=shifts + offset,
                    widths=box_width,
                    patch_artist=True,
                    boxprops=dict(edgecolor=ecolor, facecolor=fcolor, linewidth=0.5),
                    medianprops=dict(color=ecolor, linewidth=0.5),
                    capprops=dict(color=ecolor, linewidth=0.5),
                    whiskerprops=dict(color=ecolor, linewidth=0.5),
                    showfliers=False,
                )

            ax.set_title(title, fontsize=13, loc="left", pad=2.0, fontweight="bold")
            ax.grid(axis="y", linestyle="-", alpha=0.4, linewidth=0.3)
            ax.set_ylim(-0.2, 1.0)
            ax.tick_params(axis="both", labelsize=10)

        axes[0].set_ylabel("Autocorrelation", fontsize=14)
        axes[2].set_ylabel("Autocorrelation", fontsize=14)
        axes[2].set_xlabel("Spatial lag (pixels)", fontsize=14)
        axes[3].set_xlabel("Spatial lag (pixels)", fontsize=14)
        for ax in axes:
            ax.set_xticks(shifts)
            ax.set_xticklabels([str(int(shift)) for shift in shifts])

        legend_handles = [
            Patch(facecolor=fcolor, edgecolor=ecolor, label=model_name)
            for model_name, fcolor, ecolor in self.MODEL_SPECS
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=4,
            frameon=False,
            fontsize=12,
            title="Model",
            title_fontsize=12,
        )
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.output_path, dpi=800, bbox_inches="tight")
        plt.close(fig)
        return self.output_path

    def run(self) -> Path:
        correlations = self.collect()
        return self.plot(correlations=correlations)


def main(max_shift: int) -> None:

    builder = SpatialAutocorrelationFigureBuilder(
        ddim_root=DDIM_ROOT,
        ddpm_root=DDPM_ROOT,
        ecmwf_root=ECMWF_ROOT,
        target_dir=TARGET_ROOT,
        max_shift=max_shift,
    )
    output_path = builder.run()
    print(f"[fig6] Saved: {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--max-shift", type=int, default=8, required=False)
    args: Namespace = parser.parse_args()
    main(max_shift=args.max_shift)


# python reports/fig6.py
