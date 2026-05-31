from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch


SIMCAST_000_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/finetune/diffusion_v23_cosine_eta000_100steps_100members_guidancescale200")
SIMCAST_050_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/finetune/diffusion_v23_cosine_eta050_100steps_100members_guidancescale200")
SIMCAST_100_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/finetune/diffusion_v23_cosine_eta100_100steps_100members_guidancescale200")
ECMWF_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/ecmwfs2s_28/")
TARGET_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/")


class AccBarFigureBuilder:

    BOOTSTRAP_SAMPLES: int = 1000
    BOOTSTRAP_CONFIDENCE: float = 0.975
    BOOTSTRAP_SEED: int = 7341
    WEIGHTED_COLORS: dict[str, str] = {
        "simcast": "#8ec1da",
        "ecmwf": "#e69a9d",
    }
    MODEL_SPECS: list[tuple[str, Path, str, str]] = [
        (r"$\eta=0.00$", SIMCAST_000_ROOT, "era5", "#3b6e8f"),
        (r"$\eta=0.50$", SIMCAST_050_ROOT, "era5", "#3b6e8f"),
        (r"$\eta=1.00$", SIMCAST_100_ROOT, "era5", "#3b6e8f"),
        ("ECMWF-S2S", ECMWF_ROOT, "era5", "#c44e52"),
    ]

    def __init__(self, target_dir: Path) -> None:
        for model_name, root, _, _ in self.MODEL_SPECS:
            if not root.exists():
                raise FileNotFoundError(f"{model_name} root directory does not exist: {root}")
        self.target_dir: Path = target_dir
        self.output_path: Path = self.target_dir.joinpath("fig10c.png")

    @staticmethod
    def _sample_key(payload: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(payload["output_name"]),
            str(payload["out_startdate"]),
            str(payload["out_enddate"]),
        )

    def _load_aggregate_payloads(self, root: Path, dataset: str) -> list[dict[str, Any]]:
        tensor_dir: Path = root.joinpath(dataset, "tensors")
        if not tensor_dir.exists():
            raise FileNotFoundError(f"Missing tensor directory: {tensor_dir}")
        payloads: list[dict[str, Any]] = []
        for path in sorted(tensor_dir.glob("*_ens_aggregate.pt"))[:100]:
            payloads.append(torch.load(path, map_location="cpu"))
        if not payloads:
            raise FileNotFoundError(f"No aggregate tensor files found in: {tensor_dir}")
        return payloads

    def _load_aggregate_samples(self, root: Path, dataset: str) -> dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]]:
        payloads = self._load_aggregate_payloads(root=root, dataset=dataset)
        samples: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for payload in payloads:
            prediction = torch.as_tensor(payload["ensemble_mean"], dtype=torch.float32)
            groundtruth = torch.as_tensor(payload["groundtruth"], dtype=torch.float32)
            assert prediction.ndim == 2
            assert groundtruth.shape == prediction.shape
            samples[self._sample_key(payload)] = (prediction, groundtruth)
        if not samples:
            raise ValueError(f"No aggregate samples found under {root}")
        return samples

    @staticmethod
    def _sample_acc(predictions: torch.Tensor, groundtruths: torch.Tensor) -> torch.Tensor:
        assert predictions.shape == groundtruths.shape
        predictions = predictions.flatten(start_dim=1)
        groundtruths = groundtruths.flatten(start_dim=1)
        numerator = torch.sum(predictions * groundtruths, dim=1)
        denominator = torch.sqrt(
            torch.sum(predictions ** 2, dim=1) * torch.sum(groundtruths ** 2, dim=1)
        )
        acc = numerator / denominator
        return torch.where(denominator > 0, acc, torch.full_like(acc, torch.nan))

    @staticmethod
    def _latitude_weights(height: int, width: int, device: torch.device) -> torch.Tensor:
        latitudes = torch.linspace(-90.0, 90.0, steps=height, dtype=torch.float32, device=device)
        weights_1d = torch.cos(torch.deg2rad(latitudes)).clamp_min(0.0)
        return weights_1d.unsqueeze(dim=1).expand(height, width)

    @classmethod
    def _sample_latitude_weighted_acc(cls, predictions: torch.Tensor, groundtruths: torch.Tensor) -> torch.Tensor:
        assert predictions.shape == groundtruths.shape
        weights = cls._latitude_weights(
            height=predictions.shape[-2],
            width=predictions.shape[-1],
            device=predictions.device,
        )
        numerator = torch.sum(weights * predictions * groundtruths, dim=(-2, -1))
        denominator = torch.sqrt(
            torch.sum(weights * predictions ** 2, dim=(-2, -1))
            * torch.sum(weights * groundtruths ** 2, dim=(-2, -1))
        )
        acc = numerator / denominator
        return torch.where(denominator > 0, acc, torch.full_like(acc, torch.nan))

    @classmethod
    def _bootstrap_significant_difference(cls, left: torch.Tensor, right: torch.Tensor) -> bool:
        left = left[torch.isfinite(left)]
        right = right[torch.isfinite(right)]
        if left.numel() == 0:
            return False
        if right.numel() == 0:
            return False
        left_generator = torch.Generator(device=left.device).manual_seed(cls.BOOTSTRAP_SEED)
        right_generator = torch.Generator(device=right.device).manual_seed(cls.BOOTSTRAP_SEED + 1)
        left_n_samples = left.shape[0]
        right_n_samples = right.shape[0]
        positive_count = 0
        negative_count = 0
        for _ in range(cls.BOOTSTRAP_SAMPLES):
            left_sample_indices = torch.randint(
                low=0,
                high=left_n_samples,
                size=(left_n_samples,),
                device=left.device,
                generator=left_generator,
            )
            right_sample_indices = torch.randint(
                low=0,
                high=right_n_samples,
                size=(right_n_samples,),
                device=right.device,
                generator=right_generator,
            )
            difference = (
                left.index_select(dim=0, index=left_sample_indices).mean()
                - right.index_select(dim=0, index=right_sample_indices).mean()
            )
            positive_count += int(difference > 0)
            negative_count += int(difference < 0)
        positive_fraction = positive_count / cls.BOOTSTRAP_SAMPLES
        negative_fraction = negative_count / cls.BOOTSTRAP_SAMPLES
        return positive_fraction >= cls.BOOTSTRAP_CONFIDENCE or negative_fraction >= cls.BOOTSTRAP_CONFIDENCE

    def collect(self) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], bool]]:
        samples_by_model: dict[str, dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]]] = {}
        for model_name, root, dataset, _ in self.MODEL_SPECS:
            samples_by_model[model_name] = self._load_aggregate_samples(root=root, dataset=dataset)
            print(f"{model_name}: samples={len(samples_by_model[model_name])}")

        acc_by_model: dict[tuple[str, str], torch.Tensor] = {}
        values: dict[str, dict[str, float]] = {}
        for model_name, _, _, _ in self.MODEL_SPECS:
            model_keys = sorted(samples_by_model[model_name])
            predictions = torch.stack([samples_by_model[model_name][key][0] for key in model_keys], dim=0)
            groundtruths = torch.stack([samples_by_model[model_name][key][1] for key in model_keys], dim=0)
            sample_acc = self._sample_acc(predictions=predictions, groundtruths=groundtruths)
            sample_weighted_acc = self._sample_latitude_weighted_acc(
                predictions=predictions,
                groundtruths=groundtruths,
            )
            acc_by_model[(model_name, "acc")] = sample_acc
            acc_by_model[(model_name, "weighted_acc")] = sample_weighted_acc
            acc_value = float(torch.nanmean(sample_acc).item())
            weighted_acc_value = float(torch.nanmean(sample_weighted_acc).item())
            print(f"{model_name}: predictions.shape={tuple(predictions.shape)}")
            print(f"{model_name}: acc={acc_value:.6f}")
            print(f"{model_name}: latitude_weighted_acc={weighted_acc_value:.6f}")
            values[model_name] = {
                "acc": acc_value,
                "weighted_acc": weighted_acc_value,
            }

        ecmwf_name = "ECMWF-S2S"
        significant: dict[tuple[str, str], bool] = {
            (ecmwf_name, "acc"): True,
            (ecmwf_name, "weighted_acc"): True,
        }
        for model_name, _, _, _ in self.MODEL_SPECS:
            if model_name == ecmwf_name:
                continue
            for metric_key, metric_label in (("acc", "ACC"), ("weighted_acc", "latitude-weighted ACC")):
                significant[(model_name, metric_key)] = self._bootstrap_significant_difference(
                    left=acc_by_model[(model_name, metric_key)],
                    right=acc_by_model[(ecmwf_name, metric_key)],
                )
                print(
                    f"{model_name} - {ecmwf_name} {metric_label}: "
                    f"significant={significant[(model_name, metric_key)]}"
                )
        return values, significant

    def plot(self, values: dict[str, dict[str, float]], significant: dict[tuple[str, str], bool]) -> Path:
        labels = [model_name for model_name, _, _, _ in self.MODEL_SPECS]
        unweighted_colors = [color for _, _, _, color in self.MODEL_SPECS]
        weighted_colors = [
            self.WEIGHTED_COLORS["ecmwf"] if label == "ECMWF-S2S" else self.WEIGHTED_COLORS["simcast"]
            for label in labels
        ]
        acc_heights = [values[label]["acc"] for label in labels]
        weighted_acc_heights = [values[label]["weighted_acc"] for label in labels]

        fig, ax = plt.subplots(figsize=(7.8, 3.5))
        x = np.arange(len(labels), dtype=np.float64)
        ax.set_axisbelow(True)
        bar_width = 0.32
        acc_bars = ax.bar(
            x - bar_width / 2.0,
            acc_heights,
            color=unweighted_colors,
            width=bar_width,
            edgecolor="none",
            linewidth=0.0,
            zorder=3,
            label="Unweighted",
        )
        weighted_bars = ax.bar(
            x + bar_width / 2.0,
            weighted_acc_heights,
            color=weighted_colors,
            width=bar_width,
            edgecolor="none",
            linewidth=0.0,
            zorder=3,
            label="Latitude-weighted",
        )
        for bars, metric_key in ((acc_bars, "acc"), (weighted_bars, "weighted_acc")):
            for bar, label in zip(bars, labels):
                if label != "ECMWF-S2S" and significant.get((label, metric_key), False):
                    bar.set_edgecolor("#c44e52")
                    bar.set_linewidth(1.4)
                    bar.set_linestyle("--")
                    bar.set_zorder(4)
        ax.axhline(0.0, linestyle="--", color="0.4", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("ACC", fontsize=13)
        ax.set_ylim(0., 0.4)
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
        ax.tick_params(axis="both", labelsize=11)
        unweighted_handle = Patch(
            facecolor="#3b6e8f",
            edgecolor="none",
            linewidth=0.0,
            label="Unweighted",
        )
        weighted_handle = Patch(
            facecolor=self.WEIGHTED_COLORS["simcast"],
            edgecolor="none",
            linewidth=0.0,
            label="Latitude-weighted",
        )
        significance_handle = Patch(
            facecolor="none",
            edgecolor="#c44e52",
            linewidth=1.4,
            linestyle="--",
            label="Statistically Significant",
        )
        ax.legend(
            handles=[unweighted_handle, weighted_handle, significance_handle],
            frameon=False,
            fontsize=10,
            loc="upper right",
        )

        fig.tight_layout()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.output_path, dpi=800, bbox_inches="tight")
        plt.close(fig)
        return self.output_path

    def run(self) -> Path:
        values, significant = self.collect()
        return self.plot(values=values, significant=significant)


def main() -> None:
    builder = AccBarFigureBuilder(target_dir=TARGET_ROOT)
    output_path = builder.run()
    print(f"[fig10c] Saved: {output_path}")


if __name__ == "__main__":
    main()


# python reports/fig10c.py
