from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch


SIMCAST_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/finetune/diffusion_v23_cosine_eta000_100steps_100members_guidancescale200")
ECMWF_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/ecmwfs2s_28/")
TARGET_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/")


class AccComponentBarFigureBuilder:

    BOOTSTRAP_SAMPLES: int = 1000
    BOOTSTRAP_CONFIDENCE: float = 0.975
    BOOTSTRAP_SEED: int = 7341
    MODEL_SPECS: list[tuple[str, Path, str, str]] = [
        ("SimCast-S2S", SIMCAST_ROOT, "era5", "#3b6e8f"),
        ("ECMWF-S2S", ECMWF_ROOT, "era5", "#c44e52"),
    ]
    WEIGHTED_COLORS: dict[str, str] = {
        "SimCast-S2S": "#8ec1da",
        "ECMWF-S2S": "#e69a9d",
    }
    COMPONENTS: tuple[tuple[str, str], ...] = (
        ("numerator", "Numerator"),
        ("prediction_norm", "Prediction Norm"),
        ("truth_norm", "Truth Norm"),
    )

    def __init__(self, target_dir: Path) -> None:
        for model_name, root, _, _ in self.MODEL_SPECS:
            if not root.exists():
                raise FileNotFoundError(f"{model_name} root directory does not exist: {root}")
        self.target_dir: Path = target_dir
        self.output_path: Path = self.target_dir.joinpath("fig10d.png")

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
        for path in sorted(tensor_dir.glob("*_ens_aggregate.pt")):
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
    def _latitude_weights(height: int, width: int, device: torch.device) -> torch.Tensor:
        latitudes = torch.linspace(-90.0, 90.0, steps=height, dtype=torch.float32, device=device)
        weights_1d = torch.cos(torch.deg2rad(latitudes)).clamp_min(0.0)
        return weights_1d.unsqueeze(dim=1).expand(height, width)

    @staticmethod
    def _components(predictions: torch.Tensor, groundtruths: torch.Tensor) -> dict[str, torch.Tensor]:
        assert predictions.shape == groundtruths.shape
        numerator = torch.sum(predictions * groundtruths, dim=(-2, -1))
        prediction_norm = torch.sqrt(torch.sum(predictions ** 2, dim=(-2, -1)))
        truth_norm = torch.sqrt(torch.sum(groundtruths ** 2, dim=(-2, -1)))
        return {
            "numerator": numerator,
            "prediction_norm": prediction_norm,
            "truth_norm": truth_norm,
        }

    @classmethod
    def _weighted_components(cls, predictions: torch.Tensor, groundtruths: torch.Tensor) -> dict[str, torch.Tensor]:
        assert predictions.shape == groundtruths.shape
        weights = cls._latitude_weights(
            height=predictions.shape[-2],
            width=predictions.shape[-1],
            device=predictions.device,
        )
        numerator = torch.sum(weights * predictions * groundtruths, dim=(-2, -1))
        prediction_norm = torch.sqrt(torch.sum(weights * predictions ** 2, dim=(-2, -1)))
        truth_norm = torch.sqrt(torch.sum(weights * groundtruths ** 2, dim=(-2, -1)))
        return {
            "numerator": numerator,
            "prediction_norm": prediction_norm,
            "truth_norm": truth_norm,
        }

    @classmethod
    def _bootstrap_significant_difference(cls, left: torch.Tensor, right: torch.Tensor) -> bool:
        left = left[torch.isfinite(left)]
        right = right[torch.isfinite(right)]
        if left.numel() == 0 or right.numel() == 0:
            return False
        left_generator = torch.Generator(device=left.device).manual_seed(cls.BOOTSTRAP_SEED)
        right_generator = torch.Generator(device=right.device).manual_seed(cls.BOOTSTRAP_SEED + 1)
        positive_count = 0
        negative_count = 0
        for _ in range(cls.BOOTSTRAP_SAMPLES):
            left_indices = torch.randint(
                low=0,
                high=left.shape[0],
                size=(left.shape[0],),
                device=left.device,
                generator=left_generator,
            )
            right_indices = torch.randint(
                low=0,
                high=right.shape[0],
                size=(right.shape[0],),
                device=right.device,
                generator=right_generator,
            )
            difference = left.index_select(dim=0, index=left_indices).mean() - right.index_select(dim=0, index=right_indices).mean()
            positive_count += int(difference > 0)
            negative_count += int(difference < 0)
        return (
            positive_count / cls.BOOTSTRAP_SAMPLES >= cls.BOOTSTRAP_CONFIDENCE
            or negative_count / cls.BOOTSTRAP_SAMPLES >= cls.BOOTSTRAP_CONFIDENCE
        )

    def collect(self) -> tuple[dict[str, dict[str, dict[str, float]]], dict[tuple[str, str], bool]]:
        values: dict[str, dict[str, dict[str, float]]] = {}
        component_samples: dict[tuple[str, str, str], torch.Tensor] = {}

        for model_name, root, dataset, _ in self.MODEL_SPECS:
            samples = self._load_aggregate_samples(root=root, dataset=dataset)
            keys = sorted(samples)
            predictions = torch.stack([samples[key][0] for key in keys], dim=0)
            groundtruths = torch.stack([samples[key][1] for key in keys], dim=0)
            unweighted = self._components(predictions=predictions, groundtruths=groundtruths)
            weighted = self._weighted_components(predictions=predictions, groundtruths=groundtruths)

            values[model_name] = {"unweighted": {}, "weighted": {}}
            for component_key, _ in self.COMPONENTS:
                component_samples[(model_name, "unweighted", component_key)] = unweighted[component_key]
                component_samples[(model_name, "weighted", component_key)] = weighted[component_key]
                values[model_name]["unweighted"][component_key] = float(torch.nanmean(unweighted[component_key]).item())
                values[model_name]["weighted"][component_key] = float(torch.nanmean(weighted[component_key]).item())

            print(f"{model_name}: samples={len(keys)} predictions.shape={tuple(predictions.shape)}")
            for component_key, component_label in self.COMPONENTS:
                print(
                    f"{model_name}: {component_label}="
                    f"{values[model_name]['unweighted'][component_key]:.6g}, "
                    f"weighted={values[model_name]['weighted'][component_key]:.6g}"
                )

        significant: dict[tuple[str, str], bool] = {}
        for weight_key in ("unweighted", "weighted"):
            for component_key, component_label in self.COMPONENTS:
                significant[(weight_key, component_key)] = self._bootstrap_significant_difference(
                    left=component_samples[("SimCast-S2S", weight_key, component_key)],
                    right=component_samples[("ECMWF-S2S", weight_key, component_key)],
                )
                print(
                    f"SimCast-S2S - ECMWF-S2S {component_label} {weight_key}: "
                    f"significant={significant[(weight_key, component_key)]}"
                )
        return values, significant

    def plot(
        self,
        values: dict[str, dict[str, dict[str, float]]],
        significant: dict[tuple[str, str], bool],
    ) -> Path:
        component_labels = [label for _, label in self.COMPONENTS]
        x = np.arange(len(component_labels), dtype=np.float64)
        bar_width = 0.18
        offsets = {
            ("SimCast-S2S", "unweighted"): -1.5 * bar_width,
            ("SimCast-S2S", "weighted"): -0.5 * bar_width,
            ("ECMWF-S2S", "unweighted"): 0.5 * bar_width,
            ("ECMWF-S2S", "weighted"): 1.5 * bar_width,
        }
        colors = {
            ("SimCast-S2S", "unweighted"): "#3b6e8f",
            ("SimCast-S2S", "weighted"): self.WEIGHTED_COLORS["SimCast-S2S"],
            ("ECMWF-S2S", "unweighted"): "#c44e52",
            ("ECMWF-S2S", "weighted"): self.WEIGHTED_COLORS["ECMWF-S2S"],
        }

        fig, ax = plt.subplots(figsize=(7.6, 3.7))
        ax.set_axisbelow(True)
        for model_name, _, _, _ in self.MODEL_SPECS:
            for weight_key in ("unweighted", "weighted"):
                heights = [values[model_name][weight_key][component_key] for component_key, _ in self.COMPONENTS]
                bars = ax.bar(
                    x + offsets[(model_name, weight_key)],
                    heights,
                    width=bar_width,
                    color=colors[(model_name, weight_key)],
                    edgecolor="none",
                    linewidth=0.0,
                    zorder=3,
                )
                if model_name == "SimCast-S2S":
                    for bar, (component_key, _) in zip(bars, self.COMPONENTS):
                        if significant.get((weight_key, component_key), False):
                            bar.set_edgecolor("#c44e52")
                            bar.set_linewidth(1.4)
                            bar.set_linestyle("--")
                            bar.set_zorder(4)

        ax.axhline(0.0, linestyle="--", color="0.4", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(component_labels)
        ax.set_ylabel("ACC Component", fontsize=13)
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
        ax.tick_params(axis="both", labelsize=11)

        legend_handles = [
            Patch(facecolor="#3b6e8f", edgecolor="none", label="SimCast-S2S Unweighted"),
            Patch(facecolor=self.WEIGHTED_COLORS["SimCast-S2S"], edgecolor="none", label="SimCast-S2S Latitude-weighted"),
            Patch(facecolor="#c44e52", edgecolor="none", label="ECMWF-S2S Unweighted"),
            Patch(facecolor=self.WEIGHTED_COLORS["ECMWF-S2S"], edgecolor="none", label="ECMWF-S2S Latitude-weighted"),
            Patch(facecolor="none", edgecolor="#c44e52", linewidth=1.4, linestyle="--", label="Statistically Significant"),
        ]
        ax.legend(handles=legend_handles, frameon=False, fontsize=9, loc="best")

        fig.tight_layout()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.output_path, dpi=800, bbox_inches="tight")
        plt.close(fig)
        return self.output_path

    def run(self) -> Path:
        values, significant = self.collect()
        return self.plot(values=values, significant=significant)


def main() -> None:
    builder = AccComponentBarFigureBuilder(target_dir=TARGET_ROOT)
    output_path = builder.run()
    print(f"[fig10d] Saved: {output_path}")


if __name__ == "__main__":
    main()


# python reports/fig10d.py
