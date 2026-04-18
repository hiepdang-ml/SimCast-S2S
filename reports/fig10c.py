from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


DDPM_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/train_cesm2/diffusion_eta100/")
ECMWF_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/ecmwfs2s/")
TARGET_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/")


class WeightedAccBarFigureBuilder:

    MODEL_SPECS: list[tuple[str, Path, str, str]] = [
        ("Diffusion", DDPM_ROOT, "cesm2", "#3b6e8f"),
        ("ECMWF-S2S", ECMWF_ROOT, "era5", "#c44e52"),
    ]

    def __init__(self, target_dir: Path) -> None:
        for model_name, root, _, _ in self.MODEL_SPECS:
            if not root.exists():
                raise FileNotFoundError(f"{model_name} root directory does not exist: {root}")
        self.target_dir: Path = target_dir
        self.output_path: Path = self.target_dir.joinpath("fig10c.png")

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

    def _load_aggregate_samples(self, root: Path, dataset: str) -> tuple[torch.Tensor, torch.Tensor]:
        payloads = self._load_aggregate_payloads(root=root, dataset=dataset)
        predictions: list[torch.Tensor] = []
        groundtruths: list[torch.Tensor] = []
        for payload in payloads:
            prediction = torch.as_tensor(payload["ensemble_mean"], dtype=torch.float32)
            groundtruth = torch.as_tensor(payload["groundtruth"], dtype=torch.float32)
            assert prediction.ndim == 2
            assert groundtruth.shape == prediction.shape
            predictions.append(prediction)
            groundtruths.append(groundtruth)
        if not predictions:
            raise ValueError(f"No aggregate samples found under {root}")
        return torch.stack(predictions, dim=0), torch.stack(groundtruths, dim=0)

    @staticmethod
    def _latitude_weights(height: int, width: int) -> torch.Tensor:
        latitudes = torch.linspace(-90.0, 90.0, steps=height, dtype=torch.float32)
        weights_1d = torch.cos(torch.deg2rad(latitudes)).clamp_min(0.0)
        return weights_1d.unsqueeze(dim=1).expand(height, width)

    @staticmethod
    def _acc_map(predictions: torch.Tensor, groundtruths: torch.Tensor) -> torch.Tensor:
        assert predictions.shape == groundtruths.shape
        numerator = torch.sum(predictions * groundtruths, dim=0)
        denominator = torch.sqrt(
            torch.sum(predictions ** 2, dim=0) * torch.sum(groundtruths ** 2, dim=0)
        )
        acc = numerator / denominator
        return torch.where(denominator > 0, acc, torch.full_like(acc, torch.nan))

    @classmethod
    def _latitude_weighted_mean(cls, field: torch.Tensor) -> float:
        assert field.ndim == 2
        weights = cls._latitude_weights(height=field.shape[0], width=field.shape[1]).to(field.device)
        valid = torch.isfinite(field)
        if not torch.any(valid):
            return float("nan")
        weighted_sum = torch.sum(field[valid] * weights[valid])
        total_weight = torch.sum(weights[valid])
        return float((weighted_sum / total_weight).item())

    def collect(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for model_name, root, dataset, _ in self.MODEL_SPECS:
            predictions, groundtruths = self._load_aggregate_samples(root=root, dataset=dataset)
            acc_map = self._acc_map(predictions=predictions, groundtruths=groundtruths)
            weighted_acc = self._latitude_weighted_mean(acc_map)
            print(f"{model_name}: predictions.shape={tuple(predictions.shape)}")
            print(f"{model_name}: weighted_acc={weighted_acc:.6f}")
            values[model_name] = weighted_acc
        return values

    def plot(self, values: dict[str, float]) -> Path:
        labels = [model_name for model_name, _, _, _ in self.MODEL_SPECS]
        colors = [color for _, _, _, color in self.MODEL_SPECS]
        heights = [values[label] for label in labels]

        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        x = np.arange(len(labels), dtype=np.float64)
        bars = ax.bar(x, heights, color=colors, width=0.62, edgecolor="black", linewidth=0.8)
        ax.axhline(0.0, linestyle="--", color="0.4", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Latitude-Weighted ACC", fontsize=13)
        ax.set_ylim(0., 0.5)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="both", labelsize=11)

        for bar, value in zip(bars, heights):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + (0.03 if value >= 0 else -0.05),
                f"{value:.3f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=10,
            )

        fig.tight_layout()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.output_path, dpi=800, bbox_inches="tight")
        plt.close(fig)
        return self.output_path

    def run(self) -> Path:
        return self.plot(values=self.collect())


def main() -> None:
    builder = WeightedAccBarFigureBuilder(target_dir=TARGET_ROOT)
    output_path = builder.run()
    print(f"[fig10c] Saved: {output_path}")


if __name__ == "__main__":
    main()


# python reports/fig10c.py
