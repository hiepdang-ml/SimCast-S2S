from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap


DDPM_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/finetune/diffusion_v23_cosine_eta100_rank64/")
ECMWF_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/ecmwfs2s/")
TARGET_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/")


class AccMapFigureBuilder:

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
    DATA_VMIN: float = -1.
    DATA_VMAX: float = 1.
    DIFF_VMIN: float = -0.6
    DIFF_VMAX: float = 0.6

    def __init__(
        self,
        ddpm_root: Path,
        ecmwf_root: Path,
        target_dir: Path,
        postprocess: str,
    ) -> None:
        if not ddpm_root.exists():
            raise FileNotFoundError(f"DDPM root directory does not exist: {ddpm_root}")
        if not ecmwf_root.exists():
            raise FileNotFoundError(f"ECMWF root directory does not exist: {ecmwf_root}")
        if postprocess not in ("none", "bias", "bias_var"):
            raise ValueError(f"Unsupported postprocess mode: {postprocess}")
        self.dataset: str = "era5"
        self.ddpm_root: Path = ddpm_root
        self.ecmwf_root: Path = ecmwf_root
        self.target_dir: Path = target_dir
        self.output_path: Path = self.target_dir.joinpath("fig10b.png")
        self.postprocess: str = postprocess

    @staticmethod
    def _sample_key(payload: dict[str, Any]) -> tuple[str, str]:
        return (
            str(payload["out_startdate"]),
            str(payload["out_enddate"]),
        )

    def _load_aggregate_payloads(self, root: Path) -> list[dict[str, Any]]:
        tensor_dir: Path = root.joinpath(self.dataset, "tensors")
        if not tensor_dir.exists():
            raise FileNotFoundError(f"Missing tensor directory: {tensor_dir}")
        payloads: list[dict[str, Any]] = []
        for path in sorted(tensor_dir.glob("*_ens_aggregate.pt")):
            payloads.append(torch.load(path, map_location="cpu"))
        if not payloads:
            raise FileNotFoundError(f"No aggregate tensor files found in: {tensor_dir}")
        return payloads

    def _load_aggregate_samples(self, root: Path) -> tuple[torch.Tensor, torch.Tensor]:
        payloads = self._load_aggregate_payloads(root=root)
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
    def _acc_map(predictions: torch.Tensor, groundtruths: torch.Tensor) -> torch.Tensor:
        assert predictions.shape == groundtruths.shape
        numerator = torch.sum(predictions * groundtruths, dim=0)
        denominator = torch.sqrt(
            torch.sum(predictions ** 2, dim=0) * torch.sum(groundtruths ** 2, dim=0)
        )
        acc = numerator / denominator
        print(f"numerator.mean(): {numerator.mean()}")
        print(f"denominator.mean(): {denominator.mean()}")
        print(f"numerator.mean() / denominator.mean(): {numerator.mean() / denominator.mean()}")
        print(f"acc.mean(): {acc.mean()}")
        return torch.where(denominator > 0, acc, torch.full_like(acc, torch.nan))

    def _apply_postprocess(self, predictions: torch.Tensor, groundtruths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.postprocess == "none":
            return predictions, groundtruths

        calibration_predictions = predictions
        calibration_groundtruths = groundtruths

        pred_mean = calibration_predictions.mean(dim=0)
        truth_mean = calibration_groundtruths.mean(dim=0)
        adjusted_predictions = predictions + (truth_mean - pred_mean)

        if self.postprocess == "bias_var":
            pred_std = calibration_predictions.std(dim=0, unbiased=False)
            truth_std = calibration_groundtruths.std(dim=0, unbiased=False)
            scale = torch.ones_like(pred_std)
            valid = pred_std > 0
            scale[valid] = truth_std[valid] / pred_std[valid]
            adjusted_predictions = (adjusted_predictions - truth_mean) * scale + truth_mean

        print(f"[fig10b] postprocess={self.postprocess} samples={predictions.shape[0]}")
        return adjusted_predictions, groundtruths

    def collect(self) -> dict[str, torch.Tensor]:
        ddpm_predictions, ddpm_groundtruths = self._load_aggregate_samples(root=self.ddpm_root)
        ecmwf_predictions, ecmwf_groundtruths = self._load_aggregate_samples(root=self.ecmwf_root)
        ddpm_predictions, ddpm_groundtruths = self._apply_postprocess(ddpm_predictions, ddpm_groundtruths)
        ecmwf_predictions, ecmwf_groundtruths = self._apply_postprocess(ecmwf_predictions, ecmwf_groundtruths)
        print(f"ddpm_predictions.shape: {tuple(ddpm_predictions.shape)}")
        print("DDPM")
        ddpm_acc = self._acc_map(predictions=ddpm_predictions, groundtruths=ddpm_groundtruths)
        print("ECMWFS2S")
        ecmwf_acc = self._acc_map(predictions=ecmwf_predictions, groundtruths=ecmwf_groundtruths)
        return {
            "acc_ecmwf": ecmwf_acc,
            "acc_ddpm": ddpm_acc,
            "acc_diff": ddpm_acc - ecmwf_acc,
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
        keys = ("acc_ecmwf", "acc_ddpm", "acc_diff")
        data_levels = np.linspace(self.DATA_VMIN, self.DATA_VMAX, 17)
        diff_levels = np.linspace(self.DIFF_VMIN, self.DIFF_VMAX, 17)

        data_mappable = None
        diff_mappable = None
        for ax, title, key in zip(axs, titles, keys):
            frame = maps[key].cpu().numpy()
            if key == "acc_diff":
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
    parser = ArgumentParser()
    parser.add_argument("--postprocess", choices=["none", "bias", "bias_var"], default="none")
    args: Namespace = parser.parse_args()
    builder = AccMapFigureBuilder(
        ddpm_root=DDPM_ROOT,
        ecmwf_root=ECMWF_ROOT,
        target_dir=TARGET_ROOT,
        postprocess=args.postprocess,
    )
    output_path = builder.run()
    print(f"[fig10b] Saved: {output_path}")


if __name__ == "__main__":
    main()


# python reports/fig10b.py
