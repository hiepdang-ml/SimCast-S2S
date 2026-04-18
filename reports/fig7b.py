from pathlib import Path
from typing import Any

import cmocean
import matplotlib.pyplot as plt
import numpy as np
import torch

ECMWF_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/ecmwfs2s/")
TARGET_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/")


class ThresholdBrierSkillScoreFigureBuilder:

    BOTH_TAIL_PERCENTILE_PAIRS: list[tuple[float, float]] = [
        (30.0, 70.0),
        (25.0, 75.0),
        (20.0, 80.0),
        (15.0, 85.0),
        (10.0, 90.0),
        (5.0, 95.0),
    ]

    REGIONS: list[tuple[str, str]] = [
        ("global", "Global"),
        ("tropical", "Tropical"),
        ("extratropical", "Extratropical"),
    ]

    MODEL_SPECS: list[tuple[str, Path, Any]] = [
        (
            "eta=0.00",
            Path("/scratch/zgp2ps/s2s_results/finetune/diffusion_v23_cosine_eta000_rank64/"),
            cmocean.cm.algae(0.20),
        ),
        (
            "eta=0.20",
            Path("/scratch/zgp2ps/s2s_results/finetune/diffusion_v23_cosine_eta020_rank64/"),
            cmocean.cm.algae(0.32),
        ),
        (
            "eta=0.40",
            Path("/scratch/zgp2ps/s2s_results/finetune/diffusion_v23_cosine_eta040_rank64/"),
            cmocean.cm.algae(0.44),
        ),
        (
            "eta=0.60",
            Path("/scratch/zgp2ps/s2s_results/finetune/diffusion_v23_cosine_eta060_rank64/"),
            cmocean.cm.algae(0.56),
        ),
        (
            "eta=0.80",
            Path("/scratch/zgp2ps/s2s_results/finetune/diffusion_v23_cosine_eta080_rank64/"),
            cmocean.cm.algae(0.68),
        ),
        (
            "eta=1.00",
            Path("/scratch/zgp2ps/s2s_results/finetune/diffusion_v23_cosine_eta100_rank64/"),
            cmocean.cm.algae(0.80),
        ),
        ("ECMWF-S2S", ECMWF_ROOT, cmocean.cm.solar(0.82)),
    ]
    RIGHT_TAIL_PERCENTILES: np.ndarray = np.asarray([70.0, 75.0, 80.0, 85.0, 90.0, 95.0], dtype=np.float64)
    LEFT_TAIL_PERCENTILES: np.ndarray = np.asarray([30.0, 25.0, 20.0, 15.0, 10.0, 5.0], dtype=np.float64)
    TROPICAL_LAT_RANGE: tuple[float, float] = (-25.0, 25.0)

    def __init__(self, target_dir: Path) -> None:
        for model_name, root, _ in self.MODEL_SPECS:
            if not root.exists():
                raise FileNotFoundError(f"{model_name} root directory does not exist: {root}")
        self.dataset: str = "era5"
        self.target_dir: Path = target_dir
        self.output_path: Path = self.target_dir.joinpath("fig7b.png")

    @staticmethod
    def _sample_key(payload: dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(payload["in_startdate"]),
            str(payload["in_enddate"]),
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

    def _collect_model_tensors(self, root: Path) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        payloads = self._load_member_payloads(root=root)
        grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        for payload in payloads:
            grouped.setdefault(self._sample_key(payload), []).append(payload)

        prediction_list: list[torch.Tensor] = []
        groundtruth_list: list[torch.Tensor] = []
        for sample_payloads in grouped.values():
            sample_payloads = sorted(sample_payloads, key=lambda item: int(item["ensemble_member"]))
            prediction_members = [
                torch.as_tensor(item["prediction"], dtype=torch.float32)
                for item in sample_payloads
            ]
            prediction_list.append(torch.stack(prediction_members, dim=0))
            groundtruth_list.append(torch.as_tensor(sample_payloads[0]["groundtruth"], dtype=torch.float32))

        if not prediction_list:
            raise ValueError(f"No grouped samples found under {root}")
        return prediction_list, groundtruth_list

    @staticmethod
    def _latitude_weights(height: int, width: int) -> torch.Tensor:
        latitudes = torch.linspace(-90.0, 90.0, steps=height, dtype=torch.float32)
        weights_1d = torch.cos(torch.deg2rad(latitudes)).clamp_min(0.0)
        weights_2d = weights_1d.unsqueeze(dim=1).expand(height, width)
        return weights_2d

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
        if tail == "right":
            return (values > threshold).to(dtype=torch.float32)
        if tail == "left":
            return (values < threshold).to(dtype=torch.float32)
        raise ValueError(f"Unsupported tail: {tail}")

    @staticmethod
    def _both_tail_event_indicator(
        values: torch.Tensor,
        lower_threshold: torch.Tensor,
        upper_threshold: torch.Tensor,
    ) -> torch.Tensor:
        return ((values < lower_threshold) | (values > upper_threshold)).to(dtype=torch.float32)

    @classmethod
    def _weighted_percentile_threshold(
        cls,
        groundtruths: list[torch.Tensor],
        percentile: float,
    ) -> torch.Tensor:
        flat_values: list[torch.Tensor] = []
        flat_weights: list[torch.Tensor] = []
        for groundtruth in groundtruths:
            weights = cls._latitude_weights(height=groundtruth.shape[0], width=groundtruth.shape[1])
            flat_values.append(groundtruth.reshape(-1))
            flat_weights.append(weights.reshape(-1))
        values = torch.cat(flat_values, dim=0)
        weights = torch.cat(flat_weights, dim=0)
        order = torch.argsort(values)
        values_sorted = values[order]
        weights_sorted = weights[order]
        cumulative = torch.cumsum(weights_sorted, dim=0)
        cutoff = (percentile / 100.0) * cumulative[-1]
        idx = int(torch.searchsorted(cumulative, cutoff, right=False).item())
        idx = min(max(idx, 0), values_sorted.shape[0] - 1)
        threshold_value = values_sorted[idx]
        return torch.as_tensor(threshold_value, dtype=torch.float32)

    @classmethod
    def _sample_brier_skill_scores(
        cls,
        predictions: list[torch.Tensor],
        groundtruths: list[torch.Tensor],
        threshold: torch.Tensor,
        percentile: float,
        tail: str,
        region: str,
    ) -> np.ndarray:
        if len(predictions) != len(groundtruths):
            raise ValueError("predictions and groundtruths must have the same number of samples")

        sample_weights = [cls._latitude_weights(height=groundtruth.shape[0], width=groundtruth.shape[1]) for groundtruth in groundtruths]
        climatology = 1.0 - percentile / 100.0 if tail == "right" else percentile / 100.0
        scores: list[float] = []
        for prediction_members, groundtruth, weights in zip(predictions, groundtruths, sample_weights):
            event_probability = cls._event_indicator(prediction_members, threshold=threshold, tail=tail).mean(dim=0)
            event_observation = cls._event_indicator(groundtruth, threshold=threshold, tail=tail)
            event_probability = cls._regional_slice(event_probability, region=region)
            event_observation = cls._regional_slice(event_observation, region=region)
            weights = cls._regional_slice(weights, region=region)
            brier_score = torch.sum(((event_probability - event_observation) ** 2) * weights) / torch.sum(weights)
            reference_score = torch.sum(((climatology - event_observation) ** 2) * weights) / torch.sum(weights)
            scores.append(float((1.0 - brier_score / reference_score).item()))
        return np.asarray(scores, dtype=np.float64)

    @classmethod
    def _sample_brier_skill_scores_both(
        cls,
        predictions: list[torch.Tensor],
        groundtruths: list[torch.Tensor],
        lower_threshold: torch.Tensor,
        upper_threshold: torch.Tensor,
        lower_percentile: float,
        upper_percentile: float,
        region: str,
    ) -> np.ndarray:
        if len(predictions) != len(groundtruths):
            raise ValueError("predictions and groundtruths must have the same number of samples")

        sample_weights = [cls._latitude_weights(height=groundtruth.shape[0], width=groundtruth.shape[1]) for groundtruth in groundtruths]
        climatology = lower_percentile / 100.0 + (1.0 - upper_percentile / 100.0)
        scores: list[float] = []
        for prediction_members, groundtruth, weights in zip(predictions, groundtruths, sample_weights):
            event_probability = cls._both_tail_event_indicator(
                prediction_members,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
            ).mean(dim=0)
            event_observation = cls._both_tail_event_indicator(
                groundtruth,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
            )
            event_probability = cls._regional_slice(event_probability, region=region)
            event_observation = cls._regional_slice(event_observation, region=region)
            weights = cls._regional_slice(weights, region=region)
            brier_score = torch.sum(((event_probability - event_observation) ** 2) * weights) / torch.sum(weights)
            reference_score = torch.sum(((climatology - event_observation) ** 2) * weights) / torch.sum(weights)
            scores.append(float((1.0 - brier_score / reference_score).item()))
        return np.asarray(scores, dtype=np.float64)

    def collect(self) -> dict[str, dict[str, dict[str, np.ndarray]]]:
        scores: dict[str, dict[str, dict[str, np.ndarray]]] = {
            region: {"right": {}, "left": {}, "both": {}}
            for region, _ in self.REGIONS
        }
        for model_name, root, _ in self.MODEL_SPECS:
            predictions, groundtruths = self._collect_model_tensors(root=root)
            for region, _ in self.REGIONS:
                for tail, percentiles in (("right", self.RIGHT_TAIL_PERCENTILES), ("left", self.LEFT_TAIL_PERCENTILES)):
                    mean_scores: list[float] = []
                    for percentile in percentiles.tolist():
                        threshold = self._weighted_percentile_threshold(
                            groundtruths=groundtruths,
                            percentile=float(percentile),
                        )
                        sample_scores = self._sample_brier_skill_scores(
                            predictions=predictions,
                            groundtruths=groundtruths,
                            threshold=threshold,
                            percentile=float(percentile),
                            tail=tail,
                            region=region,
                        )
                        mean_scores.append(float(np.nanmean(sample_scores)))
                    scores[region][tail][model_name] = np.asarray(mean_scores, dtype=np.float64)
                mean_both_scores: list[float] = []
                for lower_percentile, upper_percentile in self.BOTH_TAIL_PERCENTILE_PAIRS:
                    lower_threshold = self._weighted_percentile_threshold(
                        groundtruths=groundtruths,
                        percentile=lower_percentile,
                    )
                    upper_threshold = self._weighted_percentile_threshold(
                        groundtruths=groundtruths,
                        percentile=upper_percentile,
                    )
                    sample_scores = self._sample_brier_skill_scores_both(
                        predictions=predictions,
                        groundtruths=groundtruths,
                        lower_threshold=lower_threshold,
                        upper_threshold=upper_threshold,
                        lower_percentile=lower_percentile,
                        upper_percentile=upper_percentile,
                        region=region,
                    )
                    mean_both_scores.append(float(np.nanmean(sample_scores)))
                scores[region]["both"][model_name] = np.asarray(mean_both_scores, dtype=np.float64)
        return scores

    def plot(self, scores: dict[str, dict[str, dict[str, np.ndarray]]]) -> Path:
        fig, axs = plt.subplots(3, 3, figsize=(17.0, 11.6), sharey=True)
        both_tail_positions = np.arange(len(self.BOTH_TAIL_PERCENTILE_PAIRS), dtype=np.float64)
        both_tail_labels = [f"{int(lower)}/{int(upper)}" for lower, upper in self.BOTH_TAIL_PERCENTILE_PAIRS]
        tail_specs = [
            ("left", self.LEFT_TAIL_PERCENTILES, "Left Tail", "< threshold"),
            ("right", self.RIGHT_TAIL_PERCENTILES, "Right Tail", "> threshold"),
            ("both", both_tail_positions, "Both Tails", "< lower or > upper"),
        ]
        for row_idx, (region, region_label) in enumerate(self.REGIONS):
            for col_idx, (tail, percentiles, tail_label, event_label) in enumerate(tail_specs):
                ax = axs[row_idx, col_idx]
                title = f"{region_label} {tail_label} ({event_label})"
                subplot_label = chr(ord("a") + row_idx * 3 + col_idx)
                for model_name, _, color in self.MODEL_SPECS:
                    ax.plot(
                        percentiles,
                        scores[region][tail][model_name],
                        marker="o",
                        markersize=5,
                        linewidth=2.0,
                        color=color,
                        label=model_name,
                    )
                ax.axhline(0.0, linestyle="--", color="0.4", linewidth=1.2)
                ax.set_title(f"({subplot_label}) {title}", fontsize=14, loc="left", pad=8.0)
                ax.grid(True, linestyle="--", alpha=0.35)
                ax.tick_params(axis="both", labelsize=10)
                if tail == "both":
                    ax.set_xticks(both_tail_positions)
                    ax.set_xticklabels(both_tail_labels)

        axs[-1, 0].set_xlabel("Percentile", fontsize=13)
        axs[-1, 1].set_xlabel("Percentile", fontsize=13)
        axs[-1, 2].set_xlabel("Percentile Pair", fontsize=13)
        for ax in axs[:, 0]:
            ax.set_ylabel("Latitude-Weighted Brier Skill Score", fontsize=13)
        axs[0, 2].legend(frameon=False, fontsize=10, loc="best")
        fig.tight_layout()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.output_path, dpi=500, bbox_inches="tight")
        plt.close(fig)
        return self.output_path

    def run(self) -> Path:
        return self.plot(scores=self.collect())


def main() -> None:
    builder = ThresholdBrierSkillScoreFigureBuilder(target_dir=TARGET_ROOT)
    output_path = builder.run()
    print(f"[fig7b] Saved: {output_path}")


if __name__ == "__main__":
    main()


# python reports/fig7b.py
