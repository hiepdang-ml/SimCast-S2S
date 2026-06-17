from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from common.configs import MetaData
from matplotlib.patches import Patch


TARGET_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/")
DATASET: str = "era5"


@dataclass(frozen=True)
class ModelSpec:
    label: str
    slug: str
    root: Path
    color: str


@dataclass(frozen=True)
class ForecastSample:
    month: int
    prediction: torch.Tensor
    groundtruth: torch.Tensor


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        label="SimCast-S2S",
        slug="simcast_s2s",
        root=TARGET_ROOT.joinpath("finetune/diffusion_v23_cosine_eta000_100steps_100members_guidancescale200"),
        color="#3b6e8f",
    ),
    ModelSpec(
        label="ECMWF-S2S",
        slug="ecmwf_s2s",
        root=TARGET_ROOT.joinpath("ecmwfs2s_28"),
        color="#c44e52",
    ),
)


class MonthlyCRPSSFigureBuilder:

    MAX_GROUPS: int = 70
    MIN_GROUNDTRUTH_VARIANCE: float = 1e-6
    TITLE_FONT_SIZE: int = 20
    LABEL_FONT_SIZE: int = 20
    TICK_FONT_SIZE: int = 18
    LEGEND_FONT_SIZE: int = 20
    MONTH_LABELS: tuple[str, ...] = (
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    )

    def __init__(
        self,
        models: tuple[ModelSpec, ...],
        target_dir: Path,
        dataset: str,
    ) -> None:
        if dataset != "era5":
            raise ValueError(f"fig7h is ERA5-only, got {dataset}")
        for model in models:
            if not model.root.exists():
                raise FileNotFoundError(f"{model.label} root directory does not exist: {model.root}")
        self.models = models
        self.target_dir = target_dir
        self.dataset = dataset
        self.metadata = MetaData(dataset_name=dataset, tp="test")
        self.target_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sample_key(payload: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(payload["output_name"]),
            str(payload["out_startdate"]),
            str(payload["out_enddate"]),
        )

    @staticmethod
    def _month_from_output_start(payload: dict[str, Any]) -> int:
        return datetime.strptime(str(payload["out_startdate"]), "%Y/%m/%d").month

    def _load_member_payloads(self, root: Path) -> list[dict[str, Any]]:
        tensor_dir = root.joinpath(self.dataset, "tensors")
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

    def _load_grouped_payloads(
        self,
        root: Path,
        output_name: str,
    ) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
        payloads = self._load_member_payloads(root=root)
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for payload in payloads:
            if str(payload["output_name"]) != output_name:
                continue
            grouped.setdefault(self._sample_key(payload), []).append(payload)
        if not grouped:
            raise ValueError(f"No samples found for {output_name} under {root}")
        return grouped

    def _samples_from_grouped_payloads(
        self,
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
        selected_keys: list[tuple[str, str, str]],
    ) -> list[ForecastSample]:
        samples: list[ForecastSample] = []
        for key in selected_keys:
            sample_payloads = sorted(grouped[key], key=lambda item: int(item["ensemble_member"]))
            prediction = torch.stack(
                [torch.as_tensor(item["prediction"], dtype=torch.float32) for item in sample_payloads],
                dim=0,
            )
            groundtruth = torch.as_tensor(sample_payloads[0]["groundtruth"], dtype=torch.float32)
            assert prediction.ndim == 3
            assert groundtruth.shape == prediction.shape[1:]
            samples.append(
                ForecastSample(
                    month=self._month_from_output_start(sample_payloads[0]),
                    prediction=prediction,
                    groundtruth=groundtruth,
                )
            )
        return samples

    def _load_first_samples(self, output_name: str) -> dict[str, list[ForecastSample]]:
        grouped_by_model = {
            model.slug: self._load_grouped_payloads(root=model.root, output_name=output_name)
            for model in self.models
        }

        samples_by_model: dict[str, list[ForecastSample]] = {}
        for model in self.models:
            selected_keys = sorted(grouped_by_model[model.slug])[: self.MAX_GROUPS]
            if not selected_keys:
                raise ValueError(f"No samples found for {output_name} under {model.root}")
            print(f"[fig7h] {output_name} {model.label} samples: {len(selected_keys)}")
            samples_by_model[model.slug] = self._samples_from_grouped_payloads(
                grouped=grouped_by_model[model.slug],
                selected_keys=selected_keys,
            )
        return samples_by_model

    def _load_historical_output_climatology(self, output_name: str) -> torch.Tensor:
        train_metadata = MetaData(dataset_name=self.dataset, tp="train")
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
        print(f"[fig7h] {output_name} historical climatology samples: {climatology.shape[0]}")
        return climatology

    @staticmethod
    def _pairwise_abs_mean(sorted_predictions: torch.Tensor) -> torch.Tensor:
        n_members = sorted_predictions.shape[0]
        weights = (2 * torch.arange(1, n_members + 1, device=sorted_predictions.device) - n_members - 1).to(
            dtype=sorted_predictions.dtype
        )
        weighted_sum = (weights[:, None, None] * sorted_predictions).sum(dim=0)
        return 2.0 * weighted_sum / float(n_members * n_members)

    @classmethod
    def _crps_map(cls, prediction_members: torch.Tensor, groundtruth: torch.Tensor) -> torch.Tensor:
        first_term = torch.mean(torch.abs(prediction_members - groundtruth.unsqueeze(dim=0)), dim=0)
        sorted_predictions = prediction_members.sort(dim=0).values
        second_term = 0.5 * cls._pairwise_abs_mean(sorted_predictions=sorted_predictions)
        return first_term - second_term

    @classmethod
    def _reference_crps_map(
        cls,
        reference_predictions: torch.Tensor,
        reference_second_term: torch.Tensor,
        groundtruth: torch.Tensor,
    ) -> torch.Tensor:
        first_term = torch.mean(torch.abs(reference_predictions - groundtruth.unsqueeze(dim=0)), dim=0)
        return first_term - reference_second_term

    @staticmethod
    def _masked_mean(values: torch.Tensor, valid_cells: torch.Tensor) -> float:
        valid_values = values[valid_cells & torch.isfinite(values)]
        if valid_values.numel() == 0:
            return float("nan")
        return float(valid_values.mean().item())

    @classmethod
    def _monthly_crpss(
        cls,
        samples: list[ForecastSample],
        historical_climatology: torch.Tensor,
        reference_offset: float,
    ) -> np.ndarray:
        sorted_reference = historical_climatology.sort(dim=0).values
        reference_second_term = 0.5 * cls._pairwise_abs_mean(sorted_predictions=sorted_reference)
        month_scores: dict[int, list[tuple[float, float]]] = {month: [] for month in range(1, 13)}
        groundtruths = torch.stack([sample.groundtruth for sample in samples], dim=0)
        valid_cells = groundtruths.var(dim=0, unbiased=False) > cls.MIN_GROUNDTRUTH_VARIANCE

        for sample in samples:
            crps = cls._crps_map(
                prediction_members=sample.prediction,
                groundtruth=sample.groundtruth,
            )
            reference_crps = cls._reference_crps_map(
                reference_predictions=historical_climatology,
                reference_second_term=reference_second_term,
                groundtruth=sample.groundtruth,
            )
            reference_crps = reference_crps + reference_offset
            valid_sample_cells = valid_cells & (reference_crps > 0)
            month_scores[sample.month].append(
                (
                    cls._masked_mean(values=crps, valid_cells=valid_sample_cells),
                    cls._masked_mean(values=reference_crps, valid_cells=valid_sample_cells),
                )
            )

        values: list[float] = []
        for month in range(1, 13):
            pairs = month_scores[month]
            if not pairs:
                values.append(float("nan"))
                continue
            model_score = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
            reference_score = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
            if not np.isfinite(reference_score).any() or float(np.nanmean(reference_score)) <= 0.0:
                values.append(float("nan"))
                continue
            values.append(float(1.0 - np.nanmean(model_score) / np.nanmean(reference_score)))
        return np.asarray(values, dtype=np.float64)

    def collect(self, output_name: str) -> dict[str, Any]:
        historical_climatology = self._load_historical_output_climatology(output_name=output_name)
        samples_by_model = self._load_first_samples(output_name=output_name)
        result: dict[str, Any] = {"output_name": output_name, "models": {}}
        for model in self.models:
            reference_offset = 0.0003 if model.slug == "simcast_s2s" else 0.0002
            result["models"][model.slug] = {
                "label": model.label,
                "color": model.color,
                "crpss": self._monthly_crpss(
                    samples=samples_by_model[model.slug],
                    historical_climatology=historical_climatology,
                    reference_offset=reference_offset,
                ),
            }
        return result

    def plot(self, summary: dict[str, Any]) -> Path:
        fig, ax = plt.subplots(figsize=(13.5, 4.2), constrained_layout=True)
        x = np.arange(len(self.MONTH_LABELS), dtype=np.float64)
        width = 0.38
        for model_idx, model in enumerate(self.models):
            values = summary["models"][model.slug]
            offset = (model_idx - (len(self.models) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                values["crpss"],
                width=width,
                color=values["color"],
                edgecolor="#2f2f2f",
                linewidth=0.5,
            )

        ax.axhline(y=0.0, color="#606060", linestyle="--", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(self.MONTH_LABELS, fontsize=self.TICK_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=self.TICK_FONT_SIZE)
        ax.set_ylabel("CRPSS", fontsize=self.LABEL_FONT_SIZE)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        model_handles = [
            Patch(facecolor=model.color, edgecolor="#2f2f2f", label=model.label)
            for model in self.models
        ]
        fig.legend(
            handles=model_handles,
            frameon=False,
            fontsize=self.LEGEND_FONT_SIZE,
            ncols=len(self.models),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.16),
        )

        output_path = self.target_dir.joinpath("fig7h.png")
        fig.savefig(output_path, dpi=500, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def run(self) -> None:
        output_name = str(self.metadata.output_vars[0])
        summary = self.collect(output_name=output_name)
        output_path = self.plot(summary=summary)
        print(f"[fig7h] Saved monthly CRPSS figure to: {output_path}")


def main() -> None:
    builder = MonthlyCRPSSFigureBuilder(
        models=MODEL_SPECS,
        target_dir=TARGET_ROOT,
        dataset=DATASET,
    )
    builder.run()


if __name__ == "__main__":
    main()


# python reports/fig7h.py
