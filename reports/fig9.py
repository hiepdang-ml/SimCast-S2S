from pathlib import Path
from typing import Any

import cmocean
import matplotlib.pyplot as plt
import numpy as np
import torch


DDIM_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/finetune/diffusion_v23_cosine_eta000_rank64/")
DDPM_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/finetune/diffusion_v23_cosine_eta100_rank64/")
ECMWF_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/ecmwfs2s/")
TARGET_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/")


class CrpsFigureBuilder:

    MODEL_SPECS: list[tuple[str, Path, Any]] = [
        ("DDIM", DDIM_ROOT, cmocean.cm.algae(0.55)),
        ("DDPM", DDPM_ROOT, cmocean.cm.amp(0.70)),
        ("ECMWF-S2S", ECMWF_ROOT, cmocean.cm.solar(0.82)),
    ]

    def __init__(self, target_dir: Path) -> None:
        for model_name, root, _ in self.MODEL_SPECS:
            if not root.exists():
                raise FileNotFoundError(f"{model_name} root directory does not exist: {root}")
        self.dataset: str = "era5"
        self.target_dir: Path = target_dir
        self.output_path: Path = self.target_dir.joinpath("fig9.png")

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
            prediction_members = [torch.as_tensor(item["prediction"], dtype=torch.float32) for item in sample_payloads]
            prediction_list.append(torch.stack(prediction_members, dim=0))
            groundtruth_list.append(torch.as_tensor(sample_payloads[0]["groundtruth"], dtype=torch.float32))

        if not prediction_list:
            raise ValueError(f"No grouped samples found under {root}")
        return prediction_list, groundtruth_list

    @staticmethod
    def _pairwise_abs_mean(sorted_predictions: torch.Tensor) -> torch.Tensor:
        n_members = sorted_predictions.shape[0]
        weights = (2 * torch.arange(1, n_members + 1, device=sorted_predictions.device) - n_members - 1).to(
            dtype=sorted_predictions.dtype
        )
        weighted_sum = (weights[:, None, None] * sorted_predictions).sum(dim=0)
        return 2.0 * weighted_sum / float(n_members * n_members)

    @classmethod
    def _sample_crps_scores(
        cls,
        predictions: list[torch.Tensor],
        groundtruths: list[torch.Tensor],
    ) -> np.ndarray:
        if len(predictions) != len(groundtruths):
            raise ValueError("predictions and groundtruths must have the same number of samples")

        scores: list[float] = []
        for prediction_members, groundtruth in zip(predictions, groundtruths):
            assert prediction_members.ndim == 3
            assert groundtruth.ndim == 2
            first_term = torch.mean(torch.abs(prediction_members - groundtruth.unsqueeze(dim=0)), dim=0)
            sorted_predictions = prediction_members.sort(dim=0).values
            second_term = 0.5 * cls._pairwise_abs_mean(sorted_predictions=sorted_predictions)
            crps_map = first_term - second_term
            scores.append(float(crps_map.mean().item()))
        return np.asarray(scores, dtype=np.float64)

    def collect(self) -> dict[str, np.ndarray]:
        scores: dict[str, np.ndarray] = {}
        for model_name, root, _ in self.MODEL_SPECS:
            predictions, groundtruths = self._collect_model_tensors(root=root)
            scores[model_name] = self._sample_crps_scores(predictions=predictions, groundtruths=groundtruths)
        return scores

    def plot(self, scores: dict[str, np.ndarray]) -> Path:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        model_names = [model_name for model_name, _, _ in self.MODEL_SPECS]
        colors = [color for _, _, color in self.MODEL_SPECS]
        positions = np.arange(1, len(model_names) + 1)
        values = [scores[model_name][~np.isnan(scores[model_name])].tolist() for model_name in model_names]

        bp = ax.boxplot(values, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.85)
            patch.set_linewidth(1.0)
        for artist_key, repeat in (("medians", 1), ("whiskers", 2), ("caps", 2)):
            for idx, artist in enumerate(bp[artist_key]):
                artist.set_color(colors[idx // repeat])
                artist.set_linewidth(1.0)

        ax.set_xlabel("Model", fontsize=13)
        ax.set_ylabel("CRPS", fontsize=13)
        ax.set_title("Continuous Ranked Probability Score", fontsize=14, pad=8.0)
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="both", labelsize=10)
        fig.tight_layout()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.output_path, dpi=500, bbox_inches="tight")
        plt.close(fig)
        return self.output_path

    def run(self) -> Path:
        return self.plot(scores=self.collect())


def main() -> None:
    builder = CrpsFigureBuilder(target_dir=TARGET_ROOT)
    output_path = builder.run()
    print(f"[fig9] Saved: {output_path}")


if __name__ == "__main__":
    main()


# python reports/fig9.py
