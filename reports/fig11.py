from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap


DDPM_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/finetune/diffusion_v23_cosine_eta000_rank64/")
ECMWF_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/ecmwfs2s/")
TARGET_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/")


class BssThresholdFigureBuilder:

    EVENT_SPECS: list[tuple[str, float, str]] = [
        ("lt", -0.03, "BSS TP\n< -0.03"),
        ("gt", 0.03, "BSS TP\n> 0.03"),
    ]
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
    DATA_VMIN: float = -0.4
    DATA_VMAX: float = 0.4
    DIFF_VMIN: float = -0.2
    DIFF_VMAX: float = 0.2

    def __init__(self, ddpm_root: Path, ecmwf_root: Path, target_dir: Path) -> None:
        if not ddpm_root.exists():
            raise FileNotFoundError(f"DDPM root directory does not exist: {ddpm_root}")
        if not ecmwf_root.exists():
            raise FileNotFoundError(f"ECMWF root directory does not exist: {ecmwf_root}")
        self.dataset: str = "era5"
        self.ddpm_root: Path = ddpm_root
        self.ecmwf_root: Path = ecmwf_root
        self.target_dir: Path = target_dir
        self.output_path: Path = self.target_dir.joinpath("fig11.png")

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

    def _load_grouped_samples(self, root: Path) -> tuple[list[torch.Tensor], torch.Tensor]:
        payloads = self._load_member_payloads(root=root)
        grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        for payload in payloads:
            grouped.setdefault(self._sample_key(payload), []).append(payload)

        predictions: list[torch.Tensor] = []
        groundtruths: list[torch.Tensor] = []
        for sample_payloads in grouped.values():
            members = sorted(sample_payloads, key=lambda item: int(item["ensemble_member"]))
            prediction_tensor = torch.stack(
                [torch.as_tensor(item["prediction"], dtype=torch.float32) for item in members],
                dim=0,
            )
            groundtruth = torch.as_tensor(members[0]["groundtruth"], dtype=torch.float32)
            assert prediction_tensor.ndim == 3
            assert groundtruth.shape == prediction_tensor.shape[1:]
            predictions.append(prediction_tensor)
            groundtruths.append(groundtruth)

        if not predictions:
            raise ValueError(f"No grouped samples found under {root}")
        return predictions, torch.stack(groundtruths, dim=0)

    @staticmethod
    def _latitude_weights(height: int, width: int) -> torch.Tensor:
        latitudes = torch.linspace(-90.0, 90.0, steps=height, dtype=torch.float32)
        weights_1d = torch.cos(torch.deg2rad(latitudes)).clamp_min(0.0)
        weights_2d = weights_1d.unsqueeze(dim=1).expand(height, width)
        return weights_2d / weights_2d.mean()

    @classmethod
    def _event_indicator(
        cls,
        values: torch.Tensor,
        direction: str,
        threshold: float,
    ) -> torch.Tensor:
        if direction == "lt":
            return (values < threshold).to(dtype=torch.float32)
        if direction == "gt":
            return (values > threshold).to(dtype=torch.float32)
        raise ValueError(f"Unsupported event direction: {direction}")

    @classmethod
    def _bss_map(
        cls,
        predictions: list[torch.Tensor],
        groundtruths: torch.Tensor,
        direction: str,
        threshold: float,
    ) -> torch.Tensor:
        assert len(predictions) == groundtruths.shape[0]
        event_probability = torch.stack(
            [cls._event_indicator(prediction, direction=direction, threshold=threshold).mean(dim=0) for prediction in predictions],
            dim=0,
        )
        event_observation = cls._event_indicator(groundtruths, direction=direction, threshold=threshold)
        climatology = event_observation.mean(dim=0, keepdim=True)
        brier_score = torch.mean((event_probability - event_observation) ** 2, dim=0)
        reference_score = torch.mean((climatology - event_observation) ** 2, dim=0)
        skill = 1.0 - brier_score / reference_score
        weighted_skill = skill * cls._latitude_weights(height=skill.shape[0], width=skill.shape[1])
        return torch.where(reference_score > 0, weighted_skill, torch.full_like(weighted_skill, torch.nan))

    def collect(self) -> dict[str, torch.Tensor]:
        ddpm_predictions, ddpm_groundtruths = self._load_grouped_samples(root=self.ddpm_root)
        ecmwf_predictions, ecmwf_groundtruths = self._load_grouped_samples(root=self.ecmwf_root)
        maps: dict[str, torch.Tensor] = {}
        for direction, threshold, _ in self.EVENT_SPECS:
            threshold_tag = f"{direction}_{threshold:+.2f}"
            ddpm_bss = self._bss_map(
                predictions=ddpm_predictions,
                groundtruths=ddpm_groundtruths,
                direction=direction,
                threshold=threshold,
            )
            ecmwf_bss = self._bss_map(
                predictions=ecmwf_predictions,
                groundtruths=ecmwf_groundtruths,
                direction=direction,
                threshold=threshold,
            )
            maps[f"ecmwf_{threshold_tag}"] = ecmwf_bss
            maps[f"ddpm_{threshold_tag}"] = ddpm_bss
            maps[f"diff_{threshold_tag}"] = ddpm_bss - ecmwf_bss
        return maps

    def plot(self, maps: dict[str, torch.Tensor]) -> Path:
        fig = plt.figure(figsize=(13.5, 10.4))
        gs = fig.add_gridspec(
            nrows=3,
            ncols=3,
            height_ratios=[1.0, 1.0, 0.035],
            hspace=0.05,
            wspace=0.05,
        )
        axs = np.array([
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])],
        ])
        cax1 = fig.add_subplot(gs[2, :2])
        cax2 = fig.add_subplot(gs[2, 2])

        titles = ["ECMWF", "DDPM", "DDPM - ECMWF"]
        row_labels = [label for _, _, label in self.EVENT_SPECS]
        data_mappable = None
        diff_mappable = None

        for row_idx, (direction, threshold, _) in enumerate(self.EVENT_SPECS):
            threshold_tag = f"{direction}_{threshold:+.2f}"
            keys = (
                f"ecmwf_{threshold_tag}",
                f"ddpm_{threshold_tag}",
                f"diff_{threshold_tag}",
            )
            for col_idx, key in enumerate(keys):
                ax = axs[row_idx, col_idx]
                frame = maps[key].cpu().numpy()
                if col_idx < 2:
                    im = ax.imshow(
                        frame,
                        origin="lower",
                        cmap=self.DATA_CMAP,
                        vmin=self.DATA_VMIN,
                        vmax=self.DATA_VMAX,
                    )
                    data_mappable = im
                else:
                    im = ax.imshow(
                        frame,
                        origin="lower",
                        cmap=self.DATA_CMAP,
                        vmin=self.DIFF_VMIN,
                        vmax=self.DIFF_VMAX,
                    )
                    diff_mappable = im
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(titles[col_idx], fontsize=16, pad=8.0)
                if col_idx == 0:
                    ax.set_ylabel(row_labels[row_idx], fontsize=15, rotation=90)

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
    builder = BssThresholdFigureBuilder(ddpm_root=DDPM_ROOT, ecmwf_root=ECMWF_ROOT, target_dir=TARGET_ROOT)
    output_path = builder.run()
    print(f"[fig11] Saved: {output_path}")


if __name__ == "__main__":
    main()


# python reports/fig11.py
