from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap


DDPM_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/finetune/diffusion_v23_cosine_eta100_rank64/")
ECMWF_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/ecmwfs2s/")
TARGET_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/")


class ProbabilityAnomalyCorrelationMapFigureBuilder:

    EVENT_PERCENTILE: float = 50.0
    DATA_CMAP = LinearSegmentedColormap.from_list(
        "ref_blue_white_red",
        [
            (0.00, "#4f5bd5"),
            (0.18, "#7f9ff0"),
            (0.36, "#b9d0f7"),
            (0.50, "#f2f2f2"),
            (0.64, "#f3c7b3"),
            (0.82, "#ef8e72"),
            (1.00, "#c91f2f"),
        ],
    )
    DATA_VMIN: float = -1.0
    DATA_VMAX: float = 1.0
    DIFF_VMIN: float = -0.5
    DIFF_VMAX: float = 0.5

    def __init__(self, ddpm_root: Path, ecmwf_root: Path, target_dir: Path) -> None:
        if not ddpm_root.exists():
            raise FileNotFoundError(f"DDPM root directory does not exist: {ddpm_root}")
        if not ecmwf_root.exists():
            raise FileNotFoundError(f"ECMWF root directory does not exist: {ecmwf_root}")
        self.dataset: str = "era5"
        self.ddpm_root: Path = ddpm_root
        self.ecmwf_root: Path = ecmwf_root
        self.target_dir: Path = target_dir
        self.output_path: Path = self.target_dir.joinpath("fig10e.png")

    @staticmethod
    def _sample_key(payload: dict[str, Any]) -> tuple[str, str]:
        return (
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

    def _load_grouped_member_samples(self, root: Path) -> tuple[list[torch.Tensor], torch.Tensor]:
        payloads = self._load_member_payloads(root=root)
        predictions: list[torch.Tensor] = []
        groundtruths: list[torch.Tensor] = []
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for payload in payloads:
            grouped.setdefault(self._sample_key(payload), []).append(payload)
        for key in sorted(grouped.keys()):
            sample_payloads = sorted(grouped[key], key=lambda item: int(item["ensemble_member"]))
            prediction_members = torch.stack(
                [torch.as_tensor(item["prediction"], dtype=torch.float32) for item in sample_payloads],
                dim=0,
            )
            groundtruth = torch.as_tensor(sample_payloads[0]["groundtruth"], dtype=torch.float32)
            assert prediction_members.ndim == 3
            assert groundtruth.shape == prediction_members.shape[1:]
            predictions.append(prediction_members)
            groundtruths.append(groundtruth)
        if not predictions:
            raise ValueError(f"No grouped member samples found under {root}")
        return predictions, torch.stack(groundtruths, dim=0)

    @staticmethod
    def _latitude_weights(height: int, width: int) -> torch.Tensor:
        latitudes = torch.linspace(-90.0, 90.0, steps=height, dtype=torch.float32)
        weights_1d = torch.cos(torch.deg2rad(latitudes)).clamp_min(0.0)
        return weights_1d.unsqueeze(dim=1).expand(height, width)

    @classmethod
    def _weighted_percentile_threshold(
        cls,
        groundtruths: torch.Tensor,
        percentile: float,
    ) -> float:
        weights = cls._latitude_weights(height=groundtruths.shape[-2], width=groundtruths.shape[-1])
        values = groundtruths.reshape(-1)
        repeated_weights = weights.unsqueeze(dim=0).expand_as(groundtruths).reshape(-1)
        order = torch.argsort(values)
        values_sorted = values[order]
        weights_sorted = repeated_weights[order]
        cumulative = torch.cumsum(weights_sorted, dim=0)
        cutoff = (percentile / 100.0) * cumulative[-1]
        idx = int(torch.searchsorted(cumulative, cutoff, right=False).item())
        idx = min(max(idx, 0), values_sorted.shape[0] - 1)
        return float(values_sorted[idx].item())

    @staticmethod
    def _pac_map(
        predictions: list[torch.Tensor],
        groundtruths: torch.Tensor,
        threshold: float,
        climatology: float,
    ) -> torch.Tensor:
        event_probability = torch.stack(
            [(prediction > threshold).to(dtype=torch.float32).mean(dim=0) for prediction in predictions],
            dim=0,
        )
        event_observation = (groundtruths > threshold).to(dtype=torch.float32)
        forecast_anomaly = event_probability - climatology
        observation_anomaly = event_observation - climatology
        numerator = torch.sum(forecast_anomaly * observation_anomaly, dim=0)
        denominator = torch.sqrt(
            torch.sum(forecast_anomaly ** 2, dim=0) * torch.sum(observation_anomaly ** 2, dim=0)
        )
        pac = numerator / denominator
        return torch.where(denominator > 0, pac, torch.full_like(pac, torch.nan))

    def collect(self) -> dict[str, torch.Tensor]:
        ddpm_predictions, ddpm_groundtruths = self._load_grouped_member_samples(root=self.ddpm_root)
        ecmwf_predictions, ecmwf_groundtruths = self._load_grouped_member_samples(root=self.ecmwf_root)

        threshold = self._weighted_percentile_threshold(
            groundtruths=ddpm_groundtruths,
            percentile=self.EVENT_PERCENTILE,
        )
        climatology = 1.0 - self.EVENT_PERCENTILE / 100.0
        print(f"event_percentile: {self.EVENT_PERCENTILE}")
        print(f"threshold: {threshold:.6f}")
        print(f"ddpm_predictions.shape: {len(ddpm_predictions)} x {tuple(ddpm_predictions[0].shape)}")
        print("DDPM")
        ddpm_pac = self._pac_map(
            predictions=ddpm_predictions,
            groundtruths=ddpm_groundtruths,
            threshold=threshold,
            climatology=climatology,
        )
        print("ECMWFS2S")
        ecmwf_pac = self._pac_map(
            predictions=ecmwf_predictions,
            groundtruths=ecmwf_groundtruths,
            threshold=threshold,
            climatology=climatology,
        )
        print(f"ddpm_pac.mean(): {ddpm_pac.nanmean()}")
        print(f"ecmwf_pac.mean(): {ecmwf_pac.nanmean()}")

        return {
            "pac_ecmwf": ecmwf_pac,
            "pac_ddpm": ddpm_pac,
            "pac_diff": ddpm_pac - ecmwf_pac,
        }

    @staticmethod
    def _contourf(ax: Any, frame: np.ndarray, levels: np.ndarray, cmap: Any, extend: str = "both") -> Any:
        y = np.arange(frame.shape[0], dtype=np.float64)
        x = np.arange(frame.shape[1], dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        return ax.contourf(xx, yy, frame, levels=levels, cmap=cmap, extend=extend)

    def plot(self, maps: dict[str, torch.Tensor]) -> Path:
        fig = plt.figure(figsize=(16, 4.2))
        gs = fig.add_gridspec(
            nrows=2,
            ncols=3,
            height_ratios=[1.0, 0.045],
            hspace=0.08,
            wspace=0.05,
        )
        axs = [fig.add_subplot(gs[0, idx]) for idx in range(3)]
        cax1 = fig.add_subplot(gs[1, :2])
        cax2 = fig.add_subplot(gs[1, 2])
        titles = ["ECMWF", "DDPM", "DDPM - ECMWF"]
        keys = ("pac_ecmwf", "pac_ddpm", "pac_diff")
        data_levels = np.linspace(self.DATA_VMIN, self.DATA_VMAX, 17)
        diff_levels = np.linspace(self.DIFF_VMIN, self.DIFF_VMAX, 17)

        data_mappable = None
        diff_mappable = None
        for ax, title, key in zip(axs, titles, keys):
            frame = maps[key].cpu().numpy()
            if key == "pac_diff":
                im = self._contourf(ax=ax, frame=frame, levels=diff_levels, cmap=self.DATA_CMAP)
                diff_mappable = im
            else:
                im = self._contourf(ax=ax, frame=frame, levels=data_levels, cmap=self.DATA_CMAP)
                data_mappable = im
            ax.set_title(title, fontsize=16, pad=8.0)
            ax.set_xticks([])
            ax.set_yticks([])

        assert data_mappable is not None and diff_mappable is not None
        cbar1 = fig.colorbar(data_mappable, cax=cax1, orientation="horizontal")
        cbar1.set_ticks(np.linspace(self.DATA_VMIN, self.DATA_VMAX, 9).tolist())
        cbar2 = fig.colorbar(diff_mappable, cax=cax2, orientation="horizontal")
        cbar2.set_ticks(np.linspace(self.DIFF_VMIN, self.DIFF_VMAX, 5).tolist())

        self.target_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.output_path, dpi=800, bbox_inches="tight")
        plt.close(fig)
        return self.output_path

    def run(self) -> Path:
        return self.plot(maps=self.collect())


def main() -> None:
    builder = ProbabilityAnomalyCorrelationMapFigureBuilder(
        ddpm_root=DDPM_ROOT,
        ecmwf_root=ECMWF_ROOT,
        target_dir=TARGET_ROOT,
    )
    output_path = builder.run()
    print(f"[fig10e] Saved: {output_path}")


if __name__ == "__main__":
    main()


# python reports/fig10e.py
