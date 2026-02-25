from typing import Any
from pathlib import Path

import json
import numpy as np
from common.utils import TorchDictIO

def main(root_string: str) -> str:
    """
    Produce table 1 in the paper as JSON (saved as file and print as string)
    """
    root_path: Path = Path(root_string)
    report: dict[str, Any] = {}

    # Group files by parent directory to avoid rescanning/recomputing per file.
    pt_files: list[Path] = sorted(root_path.rglob("*.pt"))
    parents: set[Path] = {p.parent for p in pt_files}

    for parent in sorted(parents):
        relative_parent: Path = parent.relative_to(root_path)
        directories: tuple[str, ...] = relative_parent.parts

        if "diffusion" in parent.as_posix():
            filepaths: list[Path] = sorted(parent.glob("*aggregate.pt"))
        else:
            filepaths = sorted(parent.glob("*.pt"))

        if not filepaths:
            continue

        global_mae_values: list[float] = []
        tropical_mae_values: list[float] = []
        extratropical_mae_values: list[float] = []

        torchio = TorchDictIO(dirpath=parent.as_posix())
        for f in filepaths:
            data = torchio.load(filename=f.name)
            global_mae: float = data["global_mae"]
            tropical_mae: float = data["tropical_mae"]
            extratropical_mae: float = data["extratropical_mae"]
            global_mae_values.append(global_mae)
            tropical_mae_values.append(tropical_mae)
            extratropical_mae_values.append(extratropical_mae)

        node: dict[str, Any] = report
        for d in directories:
            node = node.setdefault(d, {})

        node["global_mean_mae"] = np.mean(global_mae_values).item()
        node["global_std_mae"] = np.std(global_mae_values).item()
        node["tropical_mean_mae"] = np.mean(tropical_mae_values).item()
        node["tropical_std_mae"] = np.std(tropical_mae_values).item()
        node["extratropical_mean_mae"] = np.mean(extratropical_mae_values).item()
        node["extratropical_std_mae"] = np.std(extratropical_mae_values).item()

    report_string: str = json.dumps(report)
    with open(root_path.joinpath("report.json"), mode="w") as report_path:
        report_path.write(report_string)

    return report_string
