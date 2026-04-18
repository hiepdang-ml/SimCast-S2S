import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.configs import DiffusionConfig, ECMWFS2SConfig


class DiffusionEnsembleFigureBuilder:

    _MEMBER_PATTERN = re.compile(r"^(?P<prefix>.+)_ens_(?P<member>\d{4})\.pt$")

    def __init__(
        self,
        source_dir: Path,
        target_dir: Path,
        output_name: str | None,
        vlimit: float,
    ) -> None:
        if vlimit <= 0:
            raise ValueError(f"vlimit must be > 0, got {vlimit}")
        self.source_dir: Path = source_dir
        self.target_dir: Path = target_dir
        self.output_name: str | None = output_name
        self.vlimit: float = vlimit
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
    def _style_panel(ax: Any, title: str, title_size: int, is_bold: bool = False) -> None:
        ax.set_axis_on()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=title_size, pad=1.0, fontweight=("bold" if is_bold else "normal"))
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("black")

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
        vmax: float = self.vlimit
        vmin: float = -vmax
        levels = np.linspace(vmin, vmax, 17)

        # Equal-size layout:
        # left block (5x4) = 20 members, spacer column, right side = groundtruth.
        fig, axs = plt.subplots(
            5,
            6,
            figsize=(20, 15),
            gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.15, 1], "wspace": 0.05, "hspace": 0.15},
        )

        all_axes: list[Any] = axs.ravel().tolist()
        for ax in all_axes:
            ax.axis("off")

        member_slots: list[tuple[int, int]] = [(r, c) for r in range(5) for c in range(4)]
        for (member_idx, pred), (r, c) in zip(member_tensors, member_slots):
            y = np.arange(pred.shape[0], dtype=np.float64)
            x = np.arange(pred.shape[1], dtype=np.float64)
            xx, yy = np.meshgrid(x, y)
            im = axs[r, c].contourf(xx, yy, pred, levels=levels, cmap="RdBu", extend="both")
            self._style_panel(ax=axs[r, c], title=f"Member {member_idx}", title_size=18, is_bold=True)

        gt_row: int = 2
        gt_col: int = 5
        y = np.arange(gt.shape[0], dtype=np.float64)
        x = np.arange(gt.shape[1], dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        im = axs[gt_row, gt_col].contourf(xx, yy, gt, levels=levels, cmap="RdBu", extend="both")
        self._style_panel(ax=axs[gt_row, gt_col], title="Truth", title_size=18, is_bold=True)

        # Reserve bottom space and place a dedicated horizontal colorbar axis.
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.08, top=0.99)
        cbar_ax = fig.add_axes((0., 0.03, 1., 0.020))  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="both", extendfrac=0.025)
        cbar.ax.tick_params(labelsize=18)

        output_filename: str = f"{prefix}_ensemble_grid.png"
        output_path: Path = self.target_dir.joinpath(output_filename)
        fig.savefig(output_path, dpi=1000, bbox_inches="tight")
        plt.close(fig)
        return output_path


def main(
    model: Literal["diffusion", "ecmwf-s2s"],
    dataset: Literal["cesm2", "era5"],
    output_name: str | None,
    vlimit: float,
) -> None:
    if model == "diffusion":
        target_path: Path = DiffusionConfig().target_path
    elif model == "ecmwf-s2s":
        target_path = ECMWFS2SConfig().target_path
    else:
        raise ValueError(f"Unsupported model: {model}")
    src: Path = target_path.joinpath(f"{dataset}/tensors")
    dst: Path = target_path.joinpath(f"{dataset}/grids")

    builder = DiffusionEnsembleFigureBuilder(
        source_dir=src,
        target_dir=dst,
        output_name=output_name,
        vlimit=vlimit,
    )
    groups: dict[str, list[Path]] = builder.extract_groups()
    if not groups:
        raise FileNotFoundError(f"No ensemble member files found in: {src}")

    for idx, prefix in enumerate(sorted(groups.keys()), start=1):
        output_path: Path = builder.build_and_save(prefix=prefix, member_paths=groups[prefix], max_members=20)
        print(f"[{idx}/{len(groups)}] Saved: {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, choices=["diffusion", "ecmwf-s2s"], required=True)
    parser.add_argument("--dataset", type=str, choices=["cesm2", "era5"], required=True)
    parser.add_argument("--output-name", type=str, default=None, required=False)
    parser.add_argument("--vlimit", type=float, required=True)
    args: Namespace = parser.parse_args()

    main(
        model=args.model,
        dataset=args.dataset,
        output_name=args.output_name,
        vlimit=args.vlimit,
    )

# Example: python reports/fig5.py --model ecmwf-s2s --dataset era5 --vlimit 0.04
