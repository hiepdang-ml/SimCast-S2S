from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import torch


DEFAULT_ROOT: Path = Path("/scratch/zgp2ps/s2s_results/train_cesm2/diffusion_eta100/")


def _sample_key(payload: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(payload["sim_id"]),
        str(payload["in_startdate"]),
        str(payload["in_enddate"]),
        str(payload["out_startdate"]),
        str(payload["out_enddate"]),
    )


def _load_diagnostics(root: Path, dataset: str) -> list[dict[str, float]]:
    tensor_dir: Path = root.joinpath(dataset, "tensors")
    if not tensor_dir.exists():
        raise FileNotFoundError(f"Missing tensor directory: {tensor_dir}")

    diagnostics_by_sample: dict[tuple[str, str, str, str, str], dict[str, float]] = {}
    missing_count: int = 0
    for path in sorted(tensor_dir.glob("*_ens_aggregate.pt")):
        payload = torch.load(path, map_location="cpu")
        required_keys = (
            "latent_ensemble_std_mean",
            "latent_ensemble_var_mean",
            "decoded_ensemble_std_mean",
            "decoded_ensemble_var_mean",
        )
        if not all(key in payload for key in required_keys):
            missing_count += 1
            continue
        diagnostics_by_sample.setdefault(
            _sample_key(payload),
            {key: float(payload[key]) for key in required_keys},
        )

    if missing_count > 0:
        print(f"[latent-vs-decoded] skipped {missing_count} aggregate files without spread diagnostics")
    diagnostics = list(diagnostics_by_sample.values())
    if not diagnostics:
        raise FileNotFoundError(
            "No latent-vs-decoded spread diagnostics found. "
            "Rerun diffusion prediction after the predictor diagnostic change."
        )
    return diagnostics


def _summarize(values: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "p10": float(values.quantile(0.10).item()),
        "p90": float(values.quantile(0.90).item()),
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", choices=["cesm2", "era5"], default="cesm2")
    args: Namespace = parser.parse_args()

    diagnostics = _load_diagnostics(root=args.root, dataset=args.dataset)
    latent_std = torch.tensor([item["latent_ensemble_std_mean"] for item in diagnostics], dtype=torch.float64)
    decoded_std = torch.tensor([item["decoded_ensemble_std_mean"] for item in diagnostics], dtype=torch.float64)
    latent_var = torch.tensor([item["latent_ensemble_var_mean"] for item in diagnostics], dtype=torch.float64)
    decoded_var = torch.tensor([item["decoded_ensemble_var_mean"] for item in diagnostics], dtype=torch.float64)
    ratio = decoded_std / latent_std.clamp_min(1e-12)

    print(f"[latent-vs-decoded] root={args.root}")
    print(f"[latent-vs-decoded] dataset={args.dataset}")
    print(f"[latent-vs-decoded] samples={len(diagnostics)}")
    print(f"[latent-vs-decoded] latent std mean summary: {_summarize(latent_std)}")
    print(f"[latent-vs-decoded] decoded std mean summary: {_summarize(decoded_std)}")
    print(f"[latent-vs-decoded] latent var mean summary: {_summarize(latent_var)}")
    print(f"[latent-vs-decoded] decoded var mean summary: {_summarize(decoded_var)}")
    print(f"[latent-vs-decoded] decoded/latent std ratio summary: {_summarize(ratio)}")
    print("[latent-vs-decoded] ratio is a unit-change diagnostic, not a calibrated physical target.")


if __name__ == "__main__":
    main()


# python reports/latent_vs_decoded_spread.py
