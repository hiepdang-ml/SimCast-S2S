from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


DDPM_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/finetune/diffusion_v23_cosine_eta100_rank64/")
TARGET_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/")


class GroundtruthDistributionFigureBuilder:

    def __init__(self, source_root: Path, target_dir: Path) -> None:
        if not source_root.exists():
            raise FileNotFoundError(f"Source root directory does not exist: {source_root}")
        self.dataset: str = "era5"
        self.source_root: Path = source_root
        self.target_dir: Path = target_dir
        self.output_path: Path = self.target_dir.joinpath("fig7c.png")

    def _load_unique_groundtruths(self) -> list[torch.Tensor]:
        tensor_dir = self.source_root.joinpath(self.dataset, "tensors")
        if not tensor_dir.exists():
            raise FileNotFoundError(f"Missing tensor directory: {tensor_dir}")

        grouped: dict[tuple[str, str, str, str], torch.Tensor] = {}
        for path in sorted(tensor_dir.glob("*_ens_*.pt")):
            if path.name.endswith("_ens_aggregate.pt"):
                continue
            obj: dict[str, Any] = torch.load(path, map_location="cpu")
            sample_key = (
                str(obj["in_startdate"]),
                str(obj["in_enddate"]),
                str(obj["out_startdate"]),
                str(obj["out_enddate"]),
            )
            if sample_key in grouped:
                continue
            grouped[sample_key] = torch.as_tensor(obj["groundtruth"], dtype=torch.float32)

        if not grouped:
            raise FileNotFoundError(f"No member tensor files found in: {tensor_dir}")
        return list(grouped.values())

    def collect(self) -> np.ndarray:
        groundtruths = self._load_unique_groundtruths()
        values = torch.cat([groundtruth.reshape(-1) for groundtruth in groundtruths], dim=0)
        return values.cpu().numpy().astype(np.float64)

    def plot(self, values: np.ndarray) -> Path:
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.hist(
            values,
            bins=120,
            range=(-0.1, 0.1),
            density=True,
            color="#4C78A8",
            edgecolor="white",
            linewidth=0.3,
            alpha=0.9,
        )
        ax.set_xlim(-0.1, 0.1)
        ax.set_xlabel("Groundtruth Value", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.set_title("Groundtruth Distribution", fontsize=14, pad=8.0)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="both", labelsize=10)
        fig.tight_layout()

        self.target_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.output_path, dpi=500, bbox_inches="tight")
        plt.close(fig)
        return self.output_path

    def run(self) -> Path:
        return self.plot(values=self.collect())


def main() -> None:
    builder = GroundtruthDistributionFigureBuilder(source_root=DDPM_ROOT, target_dir=TARGET_ROOT)
    output_path = builder.run()
    print(f"[fig7c] Saved: {output_path}")


if __name__ == "__main__":
    main()


# python reports/fig7c.py
