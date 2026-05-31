from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from common.configs import MetaData
from matplotlib.lines import Line2D

SIMCAST_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/finetune/diffusion_v23_cosine_eta100_100steps_100members_guidancescale200")
ECMWF_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/ecmwfs2s_28/")
TARGET_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/")


class ThresholdBrierSkillScoreFigureBuilder:

    REGIONS: list[tuple[str, str]] = [
        ("global", "Global"),
        ("tropical", "Tropical"),
        ("extratropical", "Extratropical"),
    ]

    MODEL_SPECS: list[tuple[str, Path, Any]] = [
        ("SimCast-S2S", SIMCAST_ROOT, "#3b6e8f"),
        ("ECMWF-S2S", ECMWF_ROOT, "#c44e52"),
    ]
    RIGHT_TAIL_PERCENTILES: np.ndarray = np.asarray(
        [90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0],
        dtype=np.float64
    )
    LEFT_TAIL_PERCENTILES: np.ndarray = np.asarray(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        dtype=np.float64,
    )
    TAIL_SPECS: list[tuple[str, str, str, np.ndarray]] = [
        ("left", "Left tail", "lower", LEFT_TAIL_PERCENTILES),
        ("right", "Right tail", "upper", RIGHT_TAIL_PERCENTILES),
    ]
    TROPICAL_LAT_RANGE: tuple[float, float] = (-25.0, 25.0)
    MAX_SIMCAST_GROUPS: int = 70
    MIN_GROUNDTRUTH_VARIANCE: float = 0.
    MIN_EVENT_COUNT: int = 10
    MIN_NON_EVENT_COUNT: int = 10

    def __init__(self, target_dir: Path) -> None:
        for model_name, root, _ in self.MODEL_SPECS:
            if not root.exists():
                raise FileNotFoundError(f"{model_name} root directory does not exist: {root}")
        self.dataset: Literal["cesm2", "era5"] = "era5"
        self.target_dir: Path = target_dir
        self.output_path: Path = self.target_dir.joinpath("fig7d.png")

    @staticmethod
    def _sample_key(payload: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(payload["output_name"]),
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

    def _load_grouped_samples(self, root: Path) -> dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]]:
        payloads = self._load_member_payloads(root=root)
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for payload in payloads:
            grouped.setdefault(self._sample_key(payload), []).append(payload)

        samples: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for key, sample_payloads in grouped.items():
            members = sorted(sample_payloads, key=lambda item: int(item["ensemble_member"]))
            prediction_tensor = torch.stack(
                [torch.as_tensor(item["prediction"], dtype=torch.float32) for item in members],
                dim=0,
            )
            groundtruth = torch.as_tensor(members[0]["groundtruth"], dtype=torch.float32)
            assert prediction_tensor.ndim == 3
            assert groundtruth.shape == prediction_tensor.shape[1:]
            samples[key] = (prediction_tensor, groundtruth)

        if not samples:
            raise ValueError(f"No grouped samples found under {root}")
        return samples

    def _load_model_tensors(self, root: Path, model_name: str) -> tuple[list[torch.Tensor], torch.Tensor]:
        samples = self._load_grouped_samples(root=root)
        selected_keys = sorted(samples)
        if model_name == "SimCast-S2S":
            selected_keys = selected_keys[: self.MAX_SIMCAST_GROUPS]
        print(f"[fig7d] {model_name} groups: {len(selected_keys)}")

        prediction_list: list[torch.Tensor] = []
        groundtruth_list: list[torch.Tensor] = []
        for key in selected_keys:
            prediction, groundtruth = samples[key]
            prediction_list.append(prediction)
            groundtruth_list.append(groundtruth)
        return prediction_list, torch.stack(groundtruth_list, dim=0)

    @classmethod
    def _regional_slice(cls, values: torch.Tensor, region: str) -> torch.Tensor:
        assert values.ndim in (2, 3)
        n_lat = values.shape[-2]
        latitudes = torch.linspace(-90.0, 90.0, steps=n_lat, dtype=torch.float32, device=values.device)
        tropical_mask = (latitudes >= cls.TROPICAL_LAT_RANGE[0]) & (latitudes <= cls.TROPICAL_LAT_RANGE[1])
        if region == "tropical":
            lat_mask = tropical_mask
        elif region == "extratropical":
            lat_mask = ~tropical_mask
        elif region == "global":
            lat_mask = torch.ones_like(tropical_mask, dtype=torch.bool)
        else:
            raise ValueError(f"Unsupported region: {region}")
        if values.ndim == 2:
            return values[lat_mask, :]
        return values[:, lat_mask, :]

    @staticmethod
    def _event_indicator(values: torch.Tensor, threshold: torch.Tensor, tail: str) -> torch.Tensor:
        if tail == "upper":
            return (values > threshold).to(dtype=torch.float32)
        if tail == "lower":
            return (values < threshold).to(dtype=torch.float32)
        raise ValueError(f"Unsupported tail: {tail}")

    @staticmethod
    def _climatology_probability(percentile: float, tail: str) -> float:
        if tail == "upper":
            return 1.0 - percentile / 100.0
        if tail == "lower":
            return percentile / 100.0
        raise ValueError(f"Unsupported tail: {tail}")

    @classmethod
    def _minimum_event_count(cls, n_samples: int, percentile: float, tail: str) -> int:
        if (tail == "lower" and percentile >= 10.0) or (tail == "upper" and percentile <= 90.0):
            return cls.MIN_EVENT_COUNT
        climatology_probability = cls._climatology_probability(percentile=percentile, tail=tail)
        expected_event_count = int(round(float(n_samples) * climatology_probability))
        return max(1, min(cls.MIN_EVENT_COUNT, expected_event_count))

    @staticmethod
    def _percentile_threshold_map(samples: torch.Tensor, percentile: float) -> torch.Tensor:
        assert samples.ndim == 3
        return torch.quantile(samples, q=percentile / 100.0, dim=0)

    def _load_historical_output_climatology(self) -> torch.Tensor:
        train_metadata = MetaData(dataset_name=self.dataset, tp="train")
        output_name = train_metadata.output_vars[0]
        tensor_dir = train_metadata.write_directory.joinpath("output", output_name)
        if not tensor_dir.exists():
            raise FileNotFoundError(f"Missing historical climatology tensor directory: {tensor_dir}")

        climatology_samples: list[torch.Tensor] = []
        for path in sorted(tensor_dir.glob("*.pt")):
            _, output_tensor = torch.load(path, map_location="cpu", weights_only=False)
            output_tensor = torch.as_tensor(output_tensor, dtype=torch.float32)
            assert output_tensor.shape == (train_metadata.n_output_days, *train_metadata.resolution)
            climatology_samples.append(output_tensor.mean(dim=0))

        if not climatology_samples:
            raise FileNotFoundError(f"No historical climatology output tensors found in: {tensor_dir}")
        climatology = torch.stack(climatology_samples, dim=0)
        print(f"[fig7d] Historical climatology samples: {climatology.shape[0]}")
        return climatology

    @classmethod
    def _bss_components(
        cls,
        predictions: list[torch.Tensor],
        groundtruths: torch.Tensor,
        threshold: torch.Tensor,
        percentile: float,
        tail: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(predictions) == groundtruths.shape[0]
        event_probability = torch.stack(
            [cls._event_indicator(prediction, threshold=threshold, tail=tail).mean(dim=0) for prediction in predictions],
            dim=0,
        )
        event_observation = cls._event_indicator(groundtruths, threshold=threshold, tail=tail)
        climatology_probability = cls._climatology_probability(percentile=percentile, tail=tail)
        brier_score = (event_probability - event_observation) ** 2
        reference_score = (climatology_probability - event_observation) ** 2
        event_count = event_observation.sum(dim=0)
        non_event_count = event_observation.shape[0] - event_count
        min_event_count = cls._minimum_event_count(
            n_samples=event_observation.shape[0],
            percentile=percentile,
            tail=tail,
        )
        valid_cells = (
            (groundtruths.var(dim=0, unbiased=False) > cls.MIN_GROUNDTRUTH_VARIANCE)
            & (event_count >= min_event_count)
            & (non_event_count >= cls.MIN_NON_EVENT_COUNT)
            & (reference_score.mean(dim=0) > 0)
        )
        return brier_score, reference_score, valid_cells

    @staticmethod
    def _bss_from_components(score: torch.Tensor, reference_score: torch.Tensor, valid_cells: torch.Tensor) -> torch.Tensor:
        mean_score = score.mean(dim=0)
        mean_reference_score = reference_score.mean(dim=0)
        skill = 1.0 - mean_score / mean_reference_score
        return torch.where(valid_cells, skill, torch.zeros_like(skill))

    @classmethod
    def _regional_mean(cls, values: torch.Tensor, region: str) -> float:
        return float(cls._regional_slice(values, region=region).mean().item())

    def collect(self) -> dict[str, dict[str, dict[str, np.ndarray]]]:
        scores: dict[str, dict[str, dict[str, np.ndarray]]] = {
            tail_key: {region: {} for region, _ in self.REGIONS}
            for tail_key, _, _, _ in self.TAIL_SPECS
        }
        historical_climatology = self._load_historical_output_climatology()
        thresholds = {
            float(percentile): self._percentile_threshold_map(
                samples=historical_climatology,
                percentile=float(percentile),
            )
            for percentile in np.concatenate([self.LEFT_TAIL_PERCENTILES, self.RIGHT_TAIL_PERCENTILES]).tolist()
        }
        for model_name, root, _ in self.MODEL_SPECS:
            predictions, groundtruths = self._load_model_tensors(root=root, model_name=model_name)
            total_cell_count = int(groundtruths.shape[-2] * groundtruths.shape[-1])
            low_variance_count = int(
                (groundtruths.var(dim=0, unbiased=False) <= self.MIN_GROUNDTRUTH_VARIANCE).sum().item()
            )
            print(f"[fig7d] {model_name} masked low-variance cells: {low_variance_count}/{total_cell_count}")
            for tail_key, _, tail, percentiles in self.TAIL_SPECS:
                region_scores: dict[str, list[float]] = {region: [] for region, _ in self.REGIONS}
                for region, _ in self.REGIONS:
                    region_scores[region] = []
                for percentile in percentiles.tolist():
                    threshold = thresholds[float(percentile)]
                    event_observation = self._event_indicator(groundtruths, threshold=threshold, tail=tail)
                    min_event_count = self._minimum_event_count(
                        n_samples=event_observation.shape[0],
                        percentile=float(percentile),
                        tail=tail,
                    )
                    event_mask = (
                        (event_observation.sum(dim=0) < min_event_count)
                        | ((event_observation.shape[0] - event_observation.sum(dim=0)) < self.MIN_NON_EVENT_COUNT)
                    )
                    print(
                        f"[fig7d] {tail_key} p{percentile:.0f} {model_name} "
                        f"masked rare-event cells (min events={min_event_count}): "
                        f"{int(event_mask.sum().item())}/{total_cell_count}"
                    )
                    score, reference_score, valid_cells = self._bss_components(
                        predictions=predictions,
                        groundtruths=groundtruths,
                        threshold=threshold,
                        percentile=float(percentile),
                        tail=tail,
                    )
                    bss_map = self._bss_from_components(
                        score=score,
                        reference_score=reference_score,
                        valid_cells=valid_cells,
                    )
                    for region, _ in self.REGIONS:
                        region_scores[region].append(self._regional_mean(values=bss_map, region=region))
                for region, _ in self.REGIONS:
                    scores[tail_key][region][model_name] = np.asarray(region_scores[region], dtype=np.float64)
        return scores

    def plot(self, scores: dict[str, dict[str, dict[str, np.ndarray]]]) -> Path:
        fig = plt.figure(figsize=(13.0, 7.2))
        gs = fig.add_gridspec(nrows=3, ncols=3, height_ratios=[1.0, 1.0, 0.18], hspace=0.3, wspace=0.05)
        axs: list[list[Any]] = []
        for row_idx in range(2):
            row_axes: list[Any] = [fig.add_subplot(gs[row_idx, 0])]
            for col_idx in range(3):
                if col_idx == 0:
                    continue
                row_axes.append(fig.add_subplot(gs[row_idx, col_idx], sharey=row_axes[0]))
            axs.append(row_axes)
        legend_ax = fig.add_subplot(gs[2, :])
        legend_handles: list[Any] | None = None
        for row_idx, (tail_key, tail_label, _, percentiles) in enumerate(self.TAIL_SPECS):
            for col_idx, (region, region_label) in enumerate(self.REGIONS):
                ax = axs[row_idx][col_idx]
                subplot_label = chr(ord("a") + row_idx * 3 + col_idx)
                current_handles: list[Any] = []
                for model_name, _, color in self.MODEL_SPECS:
                    line, = ax.plot(
                        percentiles,
                        scores[tail_key][region][model_name],
                        marker="s",
                        markersize=3,
                        linewidth=1.5,
                        color=color,
                        label=model_name,
                    )
                    current_handles.append(line)
                if legend_handles is None:
                    legend_handles = current_handles
                ax.axhline(0.0, linestyle="--", color="0.4", linewidth=1.2)
                ax.set_title(f"({subplot_label}) {region_label}", fontsize=14, loc="left", pad=8.0)
                ax.set_xticks(percentiles.tolist())
                ax.set_xticklabels([f"{percentile:.0f}" for percentile in percentiles.tolist()])
                if tail_key == "left":
                    ax.set_xlabel("Lower Percentile Threshold", fontsize=13)
                else:
                    ax.set_xlabel("Upper Percentile Threshold", fontsize=13)
                ax.grid(True, linestyle="-", alpha=0.3, linewidth=0.3)
                ax.tick_params(axis="both", labelsize=12)
                if col_idx == 0:
                    ax.set_ylabel(f"{tail_label}\nBrier Skill Score", fontsize=14)
                else:
                    ax.tick_params(axis="y", labelleft=False)

        legend_ax.axis("off")
        assert legend_handles is not None
        legend_handles = legend_handles + [
            Line2D([0], [0], color="0.4", linestyle="--", linewidth=1.2, label="No Skill")
        ]
        legend_ax.legend(
            handles=legend_handles,
            labels=[model_name for model_name, _, _ in self.MODEL_SPECS] + ["No Skill"],
            ncol=8,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.35),
            frameon=False,
            fontsize=12,
            handlelength=2.0,
            columnspacing=1.1,
        )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.output_path, dpi=800, bbox_inches="tight")
        plt.close(fig)
        return self.output_path

    def run(self) -> Path:
        return self.plot(scores=self.collect())


def main() -> None:
    builder = ThresholdBrierSkillScoreFigureBuilder(target_dir=TARGET_ROOT)
    output_path = builder.run()
    print(f"[fig7d] Saved: {output_path}")


if __name__ == "__main__":
    main()


# python reports/fig7d.py
