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
        q: float,
    ) -> None:
        if not (0.0 < q <= 1.0):
            raise ValueError(f"q must be in (0, 1], got {q}")
        self.source_dir: Path = source_dir
        self.target_dir: Path = target_dir
        self.output_name: str | None = output_name
        self.q: float = q
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

    @staticmethod
    def select_member_paths(member_paths: list[Path], n_members: int) -> list[Path]:
        if n_members <= 0:
            raise ValueError(f"n_members must be > 0, got {n_members}")
        if len(member_paths) <= n_members:
            return member_paths
        if n_members == 1:
            return [member_paths[len(member_paths) // 2]]

        last_idx: int = len(member_paths) - 1
        selected_indices: list[int] = []
        for i in range(n_members):
            idx: int = round(i * last_idx / (n_members - 1))
            if idx not in selected_indices:
                selected_indices.append(idx)
        return [member_paths[i] for i in selected_indices]

    def build_and_save(self, prefix: str, member_paths: list[Path]) -> Path:
        gt: torch.Tensor | None = None
        member_tensors: list[tuple[int, torch.Tensor]] = []
        for path in member_paths:
            obj: dict[str, Any] = torch.load(path, map_location="cpu")
            if gt is None:
                gt = obj["groundtruth"]
            member_idx: int = self.member_index_from_path(path) + 1
            member_tensors.append((member_idx, obj["prediction"]))

        assert gt is not None
        all_frames: list[torch.Tensor] = [gt] + [pred for _, pred in member_tensors]
        vmax: float = max(float(frame.abs().quantile(q=self.q).item()) for frame in all_frames)
        vmin: float = -vmax

        fig, axs = plt.subplots(3, 3, figsize=(10, 6))
        fig.suptitle(prefix.replace("DIFFUSION_", ""), fontsize=11)

        member_slots: list[tuple[int, int]] = [
            (0, 0), (0, 1), (0, 2),
            (1, 0),         (1, 2),
            (2, 0), (2, 1), (2, 2),
        ]
        for ax in axs.ravel():
            ax.axis("off")

        for (member_idx, pred), (r, c) in zip(member_tensors, member_slots):
            im = axs[r, c].imshow(pred, cmap="RdBu", vmin=vmin, vmax=vmax, origin="lower")
            axs[r, c].set_title(f"Member {member_idx}", fontsize=9)
            axs[r, c].axis("off")

        im = axs[1, 1].imshow(gt, cmap="RdBu", vmin=vmin, vmax=vmax, origin="lower")
        axs[1, 1].set_title("Groundtruth", fontsize=10, fontweight="bold")
        axs[1, 1].axis("off")

        cbar = fig.colorbar(im, ax=axs.ravel().tolist(), orientation="vertical", fraction=0.02, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        fig.tight_layout(rect=(0, 0, 0.95, 0.95))

        output_filename: str = f"{prefix}_ensemble_grid.png"
        output_path: Path = self.target_dir.joinpath(output_filename)
        fig.savefig(output_path, dpi=350, bbox_inches="tight")
        plt.close(fig)
        return output_path


def main(
    dataset: Literal["cesm2", "era5"],
    output_name: str | None,
    group_index: int,
    n_members: int,
    source_dir: str | None,
    target_dir: str | None,
    q: float,
) -> None:
    config: DiffusionConfig = DiffusionConfig()
    src: Path = Path(source_dir) if source_dir else config.target_path.joinpath(f"{dataset}/tensors")
    dst: Path = Path(target_dir) if target_dir else config.target_path.joinpath(f"{dataset}/plots")

    builder = DiffusionEnsembleFigureBuilder(
        source_dir=src,
        target_dir=dst,
        output_name=output_name,
        q=q,
    )
    groups: dict[str, list[Path]] = builder.extract_groups()
    if not groups:
        raise FileNotFoundError(f"No diffusion ensemble member files found in: {src}")

    prefixes: list[str] = sorted(groups.keys())
    if group_index < 0 or group_index >= len(prefixes):
        raise IndexError(f"group_index={group_index} is out of range [0, {len(prefixes) - 1}]")

    prefix: str = prefixes[group_index]
    selected_paths: list[Path] = builder.select_member_paths(
        member_paths=groups[prefix],
        n_members=min(8, n_members),
    )
    output_path: Path = builder.build_and_save(prefix=prefix, member_paths=selected_paths)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["cesm2", "era5"], required=True)
    parser.add_argument("--output-name", type=str, default=None, required=False)
    parser.add_argument("--group-index", type=int, default=0, required=False)
    parser.add_argument("--members", type=int, default=8, required=False)
    parser.add_argument("--source-dir", type=str, default=None, required=False)
    parser.add_argument("--target-dir", type=str, default=None, required=False)
    parser.add_argument("--quantile", type=float, default=0.99, required=False)
    args: Namespace = parser.parse_args()

    main(
        dataset=args.dataset,
        output_name=args.output_name,
        group_index=args.group_index,
        n_members=args.members,
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        q=args.quantile,
    )
