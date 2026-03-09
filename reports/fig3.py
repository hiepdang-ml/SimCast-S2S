from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch


class MAEBoxplotFigureBuilder:

    MODEL_SPECS: list[tuple[str, list[str], bool]] = [
        ("cnn-small", ["cnn_small", "cnn-small"], False),
        ("cnn-mid", ["cnn_mid", "cnn-mid"], False),
        ("cnn-large", ["cnn_large", "cnn-large"], False),
        ("unet-small", ["unet_small", "unet-small"], False),
        ("unet-mid", ["unet_mid", "unet-mid"], False),
        ("unet-large", ["unet_large", "unet-large"], False),
        ("diffusion-000", ["diffusion_eta000", "diffusion-000"], True),
        ("diffusion-020", ["diffusion_eta020", "diffusion-020"], True),
        ("diffusion-040", ["diffusion_eta040", "diffusion-040"], True),
        ("diffusion-060", ["diffusion_eta060", "diffusion-060"], True),
        ("diffusion-080", ["diffusion_eta080", "diffusion-080"], True),
        ("diffusion-100", ["diffusion_eta100", "diffusion-100"], True),
    ]
    METRICS: list[tuple[str, str]] = [
        ("global_mae", "Global MAE"),
        ("tropical_mae", "Tropical MAE"),
        ("extratropical_mae", "Extratropical MAE"),
    ]

    def __init__(self, root: Path, dpi: int) -> None:
        if dpi <= 0:
            raise ValueError(f"dpi must be > 0, got {dpi}")
        self.root: Path = root
        self.dpi: int = dpi
        self.dataset: str = "era5"
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {self.root}")

    def _resolve_model_dir(self, candidates: list[str]) -> Path:
        for name in candidates:
            candidate: Path = self.root.joinpath(name)
            if candidate.exists():
                return candidate
        return self.root.joinpath(candidates[0])

    def _collect_mae_values(self, tensor_dir: Path, use_diffusion_aggregate_only: bool) -> dict[str, list[float]]:
        if use_diffusion_aggregate_only:
            filepaths: list[Path] = sorted(tensor_dir.glob("*_ens_aggregate.pt"))
        else:
            filepaths = sorted(tensor_dir.glob("*.pt"))

        values: dict[str, list[float]] = {metric: [] for metric, _ in self.METRICS}
        for path in filepaths:
            try:
                payload: dict[str, Any] = torch.load(path, map_location="cpu")
            except Exception as err:
                print(f"[fig3] skip unreadable file: {path} ({err})")
                continue

            for metric, _ in self.METRICS:
                if metric not in payload:
                    raise KeyError(f"Missing key '{metric}' in {path}")
                values[metric].append(float(payload[metric]))

        return values

    def collect_all(self) -> tuple[list[str], dict[str, dict[str, list[float]]]]:
        model_labels: list[str] = [label for label, _, _ in self.MODEL_SPECS]
        model_values: dict[str, dict[str, list[float]]] = {}
        missing: list[str] = []

        for label, dir_candidates, is_diffusion in self.MODEL_SPECS:
            model_dir: Path = self._resolve_model_dir(candidates=dir_candidates)
            tensor_dir: Path = model_dir.joinpath(self.dataset, "tensors")
            values: dict[str, list[float]] = self._collect_mae_values(
                tensor_dir=tensor_dir,
                use_diffusion_aggregate_only=is_diffusion,
            )
            if any(len(values[k]) == 0 for k, _ in self.METRICS):
                missing.append(f"{label} -> {tensor_dir}")
            model_values[label] = values

        if missing:
            raise FileNotFoundError("Missing MAE records for:\n" + "\n".join(missing))

        return model_labels, model_values

    def plot(self, model_labels: list[str], model_values: dict[str, dict[str, list[float]]]) -> Path:
        fig, axs = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        axes = list(axs.ravel()) if hasattr(axs, "ravel") else [axs]

        for ax, (metric, metric_title) in zip(axes, self.METRICS):
            data: list[list[float]] = [model_values[label][metric] for label in model_labels]
            bp = ax.boxplot(
                data,
                labels=model_labels,
                showfliers=False,
                widths=0.65,
                patch_artist=True,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor("#8fbcd4")
                patch.set_edgecolor("#1f2937")
                patch.set_linewidth(1.0)
            for artist_key in ("medians", "whiskers", "caps"):
                for artist in bp[artist_key]:
                    artist.set_color("#1f2937")
                    artist.set_linewidth(1.0)

            ax.set_ylabel(metric_title, fontsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.tick_params(axis="y", labelsize=10)

        axes[0].set_title("ERA5 Fine-Tuning MAE Distribution by Model", fontsize=13)
        axes[-1].tick_params(axis="x", labelrotation=30, labelsize=9)
        fig.tight_layout()

        output_path: Path = self.root.joinpath("finetune_boxplot/finetune.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def run(self) -> Path:
        labels, values = self.collect_all()
        return self.plot(model_labels=labels, model_values=values)


def main(root: str, output: str | None, dpi: int) -> None:
    builder = MAEBoxplotFigureBuilder(
        root=Path(root),
        dpi=dpi,
    )
    output_path: Path = builder.run()
    print(f"[fig3] Saved: {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="/scratch/zgp2ps/s2s_results/finetune",
        required=False,
        help="Root directory that contains finetune result folders (cnn_small, unet_small, diffusion_eta000, etc.).",
    )
    parser.add_argument("--dpi", type=int, default=800, required=False)
    args: Namespace = parser.parse_args()
    main(root=args.root, output=args.output, dpi=args.dpi)
