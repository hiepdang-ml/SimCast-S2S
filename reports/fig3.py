from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Patch
import torch


class MAEBoxplotFigureBuilder:

    MODEL_SPECS: list[tuple[str, str, bool]] = [
        # ("cnn-small", "finetune/cnn_small", False),
        ("cnn-mid", "train_era5/cnn_mid", False),
        ("cnn-mid\n+ lora", "finetune/cnn_mid", False),
        # ("cnn-large", "finetune/cnn_large", False),
        # ("unet-small", "finetune/unet_small", False),
        ("unet-mid", "train_era5/unet_mid", False),
        ("unet-mid\n+ lora", "finetune/unet_mid", False),
        # ("unet-large", "finetune/unet_large", False),

        # ("cnn-small", "train_era5/cnn_small", False),
        # ("cnn-mid", "train_era5/cnn_mid", False),
        # ("cnn-large", "train_era5/cnn_large", False),
        # ("unet-small", "train_era5/unet_small", False),
        # ("unet-mid", "train_era5/unet_mid", False),
        # ("unet-large", "train_era5/unet_large", False),

        # ("diffusion-000", "train_era5/diffusion_eta000", True),
        # ("diffusion-000\nfinetuned", "finetune/diffusion_v23_cosine_eta000_rank64", True),
        # ("diffusion-020", "finetune/diffusion_v23_cosine_eta020_rank64", True),
        # ("diffusion-040", "finetune/diffusion_v23_cosine_eta040_rank64", True),
        # ("diffusion-060", "finetune/diffusion_v23_cosine_eta060_rank64", True),
        # ("diffusion-080", "finetune/diffusion_v23_cosine_eta080_rank64", True),
        ("ddpm", "train_era5/diffusion_eta100", True),
        ("ddpm\n+ lora", "finetune/diffusion_v23_cosine_eta100_rank64", True),
        ("ecmwf-s2s", "ecmwfs2s", True),
    ]
    METRICS: list[tuple[str, str]] = [
        ("global_mae", "Global MAE"),
        ("tropical_mae", "Tropical MAE"),
        ("extratropical_mae", "Extratropical MAE"),
    ]
    GROUP_COLORS: dict[str, str] = {
        "cnn": "#4C78A8",
        "unet": "#7A6A9A",
        "ddpm": "#54A24B",
        "ecmwf-s2s": "#B43A3A",
    }

    def __init__(self, root: Path) -> None:
        if not root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {root}")

        self.root: Path = root
        self.dataset: str = "era5"
        self.output_path: Path = self.root.joinpath("fig3.png")

    def _resolve_model_dir(self, relative_dir: str) -> Path:
        candidate: Path = self.root.joinpath(relative_dir)
        if not candidate.exists():
            raise FileNotFoundError(f"Missing model directory: {candidate}")
        return candidate

    @staticmethod
    def _group_from_label(label: str) -> str:
        normalized: str = label.lower().replace("\nfinetuned", "").replace("\n", " ").strip()
        if normalized == "ecmwf-s2s":
            return "ecmwf-s2s"
        if normalized.startswith("cnn"):
            return "cnn"
        if normalized.startswith("unet"):
            return "unet"
        if normalized.startswith("ddpm"):
            return "ddpm"
        return normalized.split("-")[0]

    @staticmethod
    def _is_finetuned_entry(relative_dir: str) -> bool:
        return relative_dir.startswith("finetune/")

    def _collect_mae_values(self, tensor_dir: Path, aggregate_only: bool) -> dict[str, list[float]]:
        if aggregate_only:
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

        for label, relative_dir, aggregate_only in self.MODEL_SPECS:
            model_dir: Path = self._resolve_model_dir(relative_dir=relative_dir)
            tensor_dir: Path = model_dir.joinpath(self.dataset, "tensors")
            values: dict[str, list[float]] = self._collect_mae_values(
                tensor_dir=tensor_dir,
                aggregate_only=aggregate_only,
            )
            if any(len(values[k]) == 0 for k, _ in self.METRICS):
                missing.append(f"{label} -> {tensor_dir}")
            model_values[label] = values

        if missing:
            raise FileNotFoundError("Missing MAE records for:\n" + "\n".join(missing))

        return model_labels, model_values

    @staticmethod
    def _stronger_family_color(color: str, amount: float = 0.3) -> str:
        rgb: tuple[float, float, float] = to_rgb(color)
        darker: tuple[float, float, float] = tuple(channel * (1.0 - amount) for channel in rgb)
        return to_hex(darker)

    def plot(self, model_labels: list[str], model_values: dict[str, dict[str, list[float]]]) -> Path:
        fig, axs = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
        axes = list(axs.ravel()) if hasattr(axs, "ravel") else [axs]
        use_base_color: dict[str, bool] = {
            label: (label == "ecmwf-s2s" or self._is_finetuned_entry(relative_dir))
            for label, relative_dir, _ in self.MODEL_SPECS
        }

        for idx, (ax, (metric, metric_title)) in enumerate(zip(axes, self.METRICS)):
            data: list[list[float]] = [model_values[label][metric] for label in model_labels]
            bp = ax.boxplot(
                data,
                tick_labels=model_labels,
                showfliers=False,
                widths=0.45,
                patch_artist=True,
            )
            for patch, label in zip(bp["boxes"], model_labels):
                group: str = self._group_from_label(label)
                base_color: str = self.GROUP_COLORS.get(group, "#8fbcd4")
                is_base_color: bool = use_base_color[label]
                line_color: str = self._stronger_family_color(base_color) if is_base_color else base_color
                patch.set_facecolor(base_color)
                patch.set_alpha(1.0 if is_base_color else 0.35)
                patch.set_linestyle("-" if is_base_color else "--")
                patch.set_edgecolor(line_color)
                patch.set_linewidth(1.0)
            for median, label in zip(bp["medians"], model_labels):
                group: str = self._group_from_label(label)
                base_color: str = self.GROUP_COLORS.get(group, "#8fbcd4")
                is_base_color: bool = use_base_color[label]
                line_color: str = self._stronger_family_color(base_color) if is_base_color else base_color
                median.set_color(line_color)
                median.set_linestyle("-" if is_base_color else "--")
                median.set_linewidth(1.0)
            whisker_labels: list[str] = [label for label in model_labels for _ in range(2)]
            for whisker, label in zip(bp["whiskers"], whisker_labels):
                group: str = self._group_from_label(label)
                base_color: str = self.GROUP_COLORS.get(group, "#8fbcd4")
                is_base_color: bool = use_base_color[label]
                line_color: str = self._stronger_family_color(base_color) if is_base_color else base_color
                whisker.set_color(line_color)
                whisker.set_linestyle("-" if is_base_color else "--")
                whisker.set_linewidth(1.0)
            for cap, label in zip(bp["caps"], whisker_labels):
                group: str = self._group_from_label(label)
                base_color: str = self.GROUP_COLORS.get(group, "#8fbcd4")
                is_base_color: bool = use_base_color[label]
                line_color: str = self._stronger_family_color(base_color) if is_base_color else base_color
                cap.set_color(line_color)
                cap.set_linestyle("-" if is_base_color else "--")
                cap.set_linewidth(1.0)

            ax.set_ylabel(metric_title, fontsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.tick_params(axis="y", labelsize=10)
            if idx == 0:
                ax.set_ylim((0.011, 0.017))
            elif idx == 1:
                ax.set_ylim((0.015, 0.030))
            elif idx == 2:
                ax.set_ylim((0.007, 0.014))


        axes[0].set_title("ERA5 MAE Distribution by Model", fontsize=13, pad=26)
        legend_handles: list[Patch] = [
            Patch(facecolor=color, label=group.upper())
            for group, color in self.GROUP_COLORS.items()
        ]
        axes[0].legend(
            handles=legend_handles,
            loc="lower right",
            bbox_to_anchor=(1.01, 0.95),
            ncol=4,
            frameon=False,
            fontsize=9,
        )
        axes[-1].tick_params(axis="x", labelsize=9)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return self.output_path

    def run(self) -> Path:
        labels, values = self.collect_all()
        return self.plot(model_labels=labels, model_values=values)


def main(root: str) -> None:
    builder = MAEBoxplotFigureBuilder(root=Path(root))
    output_path: Path = builder.run()
    print(f"[fig3] Saved: {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    args: Namespace = parser.parse_args()
    main(root=args.root)

# Example: python reports/fig3.py --root=/scratch/zgp2ps/s2s_results
