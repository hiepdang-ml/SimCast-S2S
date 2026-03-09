import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import torch

from common.configs import DiffusionConfig


class DiffusionEnsembleFigureBuilder:

    _MEMBER_PATTERN = re.compile(r"^(?P<prefix>.+)_ens_(?P<member>\d{4})\.pt$")

    def __init__(
        self,
        source_dir: Path,
        target_dir: Path,
        output_name: str | None,
    ) -> None:
        self.source_dir: Path = source_dir
        self.target_dir: Path = target_dir
        self.output_name: str | None = output_name
        self.q: float = 0.99
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def extract_groups(self) -> dict[str, list[Path]]:
        groups: dict[str, list[Path]] = {}
        for path in sorted(self.source_dir.glob("*_ens_*.pt")):
            if path.name.endswith("_ens_aggregate.pt"):
                continue
            match = self._MEMBER_PATTERN.match(path.name)
            if match is None:
                continue
            prefix: str = match.group("prefix")
            if self.output_name is not None and f"_{self.output_name}_" not in prefix:
                continue
            groups.setdefault(prefix, []).append(path)

        for prefix in groups:
            groups[prefix].sort(key=self.member_index_from_path)
        return groups

    def member_index_from_path(self, path: Path) -> int:
        match = self._MEMBER_PATTERN.match(path.name)
        if match is None:
            raise ValueError(f"Invalid member filename: {path.name}")
        return int(match.group("member"))

    def build_and_save(self, prefix: str, member_paths: list[Path], max_members: int = 20) -> Path:
        if max_members <= 0:
            raise ValueError(f"max_members must be > 0, got {max_members}")

        selected_member_paths: list[Path] = member_paths[:max_members]
        gt: torch.Tensor | None = None
        member_tensors: list[tuple[int, torch.Tensor]] = []
        for path in selected_member_paths:
            obj: dict[str, Any] = torch.load(path, map_location="cpu")
            if gt is None:
                gt = obj["groundtruth"]
            member_idx: int = self.member_index_from_path(path) + 1
            member_tensors.append((member_idx, obj["prediction"]))

        assert gt is not None
        all_frames: list[torch.Tensor] = [gt] + [pred for _, pred in member_tensors]
        vmax: float = max(float(frame.abs().quantile(q=self.q).item()) for frame in all_frames)
        vmin: float = -vmax

        # 5x5 grid: keep center for groundtruth, place members in remaining 24 slots.
        fig, axs = plt.subplots(5, 5, figsize=(12, 10))
        fig.suptitle(prefix.replace("DIFFUSION_", ""), fontsize=10)

        member_slots: list[tuple[int, int]] = []
        for r in range(5):
            for c in range(5):
                if (r, c) != (2, 2):
                    member_slots.append((r, c))

        for ax in axs.ravel():
            ax.axis("off")

        for (member_idx, pred), (r, c) in zip(member_tensors, member_slots):
            im = axs[r, c].imshow(pred, cmap="RdBu", vmin=vmin, vmax=vmax, origin="lower")
            axs[r, c].set_title(f"M{member_idx}", fontsize=8, pad=1.5)
            axs[r, c].axis("off")

        im = axs[2, 2].imshow(gt, cmap="RdBu", vmin=vmin, vmax=vmax, origin="lower")
        axs[2, 2].set_title("Groundtruth", fontsize=9, fontweight="bold", pad=1.5)
        axs[2, 2].axis("off")

        cbar = fig.colorbar(im, ax=axs.ravel().tolist(), orientation="vertical", fraction=0.02, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        fig.subplots_adjust(left=0.01, right=0.95, bottom=0.01, top=0.95, wspace=0.01, hspace=0.01)

        output_filename: str = f"{prefix}_ensemble_grid.png"
        output_path: Path = self.target_dir.joinpath(output_filename)
        fig.savefig(output_path, dpi=350, bbox_inches="tight")
        plt.close(fig)
        return output_path


def main(
    dataset: Literal["cesm2", "era5"],
    output_name: str | None,
) -> None:
    config: DiffusionConfig = DiffusionConfig()
    src: Path = config.target_path.joinpath(f"{dataset}/tensors")
    dst: Path = config.target_path.joinpath(f"{dataset}/plots")

    builder = DiffusionEnsembleFigureBuilder(
        source_dir=src,
        target_dir=dst,
        output_name=output_name,
    )
    groups: dict[str, list[Path]] = builder.extract_groups()
    if not groups:
        raise FileNotFoundError(f"No diffusion ensemble member files found in: {src}")

    for idx, prefix in enumerate(sorted(groups.keys()), start=1):
        output_path: Path = builder.build_and_save(prefix=prefix, member_paths=groups[prefix], max_members=20)
        print(f"[{idx}/{len(groups)}] Saved: {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["cesm2", "era5"], required=True)
    parser.add_argument("--output-name", type=str, default=None, required=False)
    args: Namespace = parser.parse_args()

    main(
        dataset=args.dataset,
        output_name=args.output_name,
    )


# Example: python reports/fig3.py --dataset cesm2
