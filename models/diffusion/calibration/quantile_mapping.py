from __future__ import annotations

import copy
import re
from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch


class QuantileMapper:
    """Gridpoint-wise empirical quantile mapping for saved ensemble tensors."""

    _MEMBER_PATTERN = re.compile(r"^(?P<prefix>.+)_ens_(?P<member>\d{4})\.pt$")

    def __init__(
        self,
        probabilities: torch.Tensor,
        forecast_quantiles: torch.Tensor,
        observed_quantiles: torch.Tensor,
        output_name: str,
    ) -> None:
        assert probabilities.ndim == 1
        assert forecast_quantiles.shape == observed_quantiles.shape
        assert forecast_quantiles.ndim == 3
        assert forecast_quantiles.shape[0] == probabilities.shape[0]
        self.probabilities: torch.Tensor = probabilities.to(dtype=torch.float32, device="cpu")
        self.forecast_quantiles: torch.Tensor = forecast_quantiles.to(dtype=torch.float32, device="cpu")
        self.observed_quantiles: torch.Tensor = observed_quantiles.to(dtype=torch.float32, device="cpu")
        self.output_name: str = output_name

    @classmethod
    def fit(
        cls,
        forecast_members: torch.Tensor,
        observations: torch.Tensor,
        output_name: str,
        n_quantiles: int,
    ) -> "QuantileMapper":
        if n_quantiles < 2:
            raise ValueError(f"n_quantiles must be at least 2, got {n_quantiles}")
        assert forecast_members.ndim == 4
        assert observations.ndim == 3
        n_samples, n_members, height, width = forecast_members.shape
        assert observations.shape == (n_samples, height, width)

        probabilities = torch.linspace(0.0, 1.0, steps=n_quantiles, dtype=torch.float32)
        forecast_values = forecast_members.reshape(n_samples * n_members, height, width)
        forecast_quantiles = torch.quantile(forecast_values, q=probabilities, dim=0)
        observed_quantiles = torch.quantile(observations, q=probabilities, dim=0)
        return cls(
            probabilities=probabilities,
            forecast_quantiles=forecast_quantiles,
            observed_quantiles=observed_quantiles,
            output_name=output_name,
        )

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        original_shape = values.shape
        if values.ndim == 2:
            values_3d = values.unsqueeze(dim=0)
        elif values.ndim == 3:
            values_3d = values
        else:
            raise ValueError(f"values must have shape (H, W) or (N, H, W), got {tuple(values.shape)}")

        n_items, height, width = values_3d.shape
        n_quantiles, q_height, q_width = self.forecast_quantiles.shape
        assert (height, width) == (q_height, q_width)

        flat_values = values_3d.reshape(n_items, height * width).transpose(0, 1).contiguous()
        flat_forecast = self.forecast_quantiles.reshape(n_quantiles, height * width).transpose(0, 1).contiguous()
        flat_observed = self.observed_quantiles.reshape(n_quantiles, height * width).transpose(0, 1).contiguous()

        indices = torch.searchsorted(flat_forecast, flat_values, right=False)
        right_idx = indices.clamp(min=1, max=n_quantiles - 1)
        left_idx = right_idx - 1

        left_forecast = torch.gather(flat_forecast, dim=1, index=left_idx)
        right_forecast = torch.gather(flat_forecast, dim=1, index=right_idx)
        left_observed = torch.gather(flat_observed, dim=1, index=left_idx)
        right_observed = torch.gather(flat_observed, dim=1, index=right_idx)

        denominator = right_forecast - left_forecast
        fraction = torch.zeros_like(flat_values)
        valid = denominator.abs() > 1e-12
        fraction[valid] = ((flat_values[valid] - left_forecast[valid]) / denominator[valid]).clamp(0.0, 1.0)
        calibrated = left_observed + fraction * (right_observed - left_observed)
        calibrated = torch.where(indices <= 0, flat_observed[:, :1].expand_as(calibrated), calibrated)
        calibrated = torch.where(indices >= n_quantiles, flat_observed[:, -1:].expand_as(calibrated), calibrated)

        output = calibrated.transpose(0, 1).reshape(n_items, height, width)
        if len(original_shape) == 2:
            return output.squeeze(dim=0)
        return output

    def state_dict(self) -> dict[str, Any]:
        return {
            "probabilities": self.probabilities,
            "forecast_quantiles": self.forecast_quantiles,
            "observed_quantiles": self.observed_quantiles,
            "output_name": self.output_name,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "QuantileMapper":
        return cls(
            probabilities=torch.as_tensor(state["probabilities"], dtype=torch.float32),
            forecast_quantiles=torch.as_tensor(state["forecast_quantiles"], dtype=torch.float32),
            observed_quantiles=torch.as_tensor(state["observed_quantiles"], dtype=torch.float32),
            output_name=str(state["output_name"]),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path) -> "QuantileMapper":
        state: dict[str, Any] = torch.load(path, map_location="cpu")
        return cls.from_state_dict(state)

    @classmethod
    def _member_paths(cls, tensor_dir: Path, output_name: str) -> list[Path]:
        paths: list[Path] = []
        for path in sorted(tensor_dir.glob("*_ens_*.pt")):
            if path.name.endswith("_ens_aggregate.pt"):
                continue
            match = cls._MEMBER_PATTERN.match(path.name)
            if match is None:
                continue
            if f"_{output_name}_" not in match.group("prefix"):
                continue
            paths.append(path)
        if not paths:
            raise FileNotFoundError(f"No member tensor files found for output_name={output_name} in {tensor_dir}")
        return paths

    @classmethod
    def load_validation_tensors(
        cls,
        tensor_dir: Path,
        output_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        for path in cls._member_paths(tensor_dir=tensor_dir, output_name=output_name):
            payload: dict[str, Any] = torch.load(path, map_location="cpu")
            key = (
                str(payload["in_startdate"]),
                str(payload["in_enddate"]),
                str(payload["out_startdate"]),
                str(payload["out_enddate"]),
            )
            grouped.setdefault(key, []).append(payload)

        group_sizes = Counter(len(payloads) for payloads in grouped.values())
        expected_members = group_sizes.most_common(1)[0][0]
        skipped_groups = 0

        forecast_samples: list[torch.Tensor] = []
        observation_samples: list[torch.Tensor] = []
        for payloads in grouped.values():
            if len(payloads) != expected_members:
                skipped_groups += 1
                continue
            payloads = sorted(payloads, key=lambda item: int(item["ensemble_member"]))
            members = [torch.as_tensor(item["prediction"], dtype=torch.float32) for item in payloads]
            groundtruth = torch.as_tensor(payloads[0]["groundtruth"], dtype=torch.float32)
            member_stack = torch.stack(members, dim=0)
            assert member_stack.ndim == 3
            assert groundtruth.shape == member_stack.shape[1:]
            forecast_samples.append(member_stack)
            observation_samples.append(groundtruth)

        if not forecast_samples:
            raise ValueError(f"No validation samples found for output_name={output_name} in {tensor_dir}")
        if skipped_groups > 0:
            print(
                f"[quantile-mapping] skipped {skipped_groups} incomplete validation groups "
                f"for {output_name}; expected_members={expected_members}"
            )
        return torch.stack(forecast_samples, dim=0), torch.stack(observation_samples, dim=0)


def _aggregate_payload(member_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    members = sorted(member_payloads, key=lambda item: int(item["ensemble_member"]))
    stack = torch.stack([torch.as_tensor(item["prediction"], dtype=torch.float32) for item in members], dim=0)
    groundtruth = torch.as_tensor(members[0]["groundtruth"], dtype=torch.float32)
    spread_unbiased = stack.shape[0] > 1
    prefix = str(members[0]["prefix"])
    result = copy.deepcopy(members[0])
    result.pop("prediction", None)
    result.pop("error_map", None)
    result["ensemble_mean"] = stack.mean(dim=0)
    result["ensemble_std"] = stack.std(dim=0, unbiased=spread_unbiased)
    result["ensemble_var"] = stack.var(dim=0, unbiased=spread_unbiased)
    for quantile, tag in (
        (0.01, "q001"),
        (0.05, "q005"),
        (0.10, "q010"),
        (0.25, "q025"),
        (0.50, "q050"),
        (0.75, "q075"),
        (0.90, "q090"),
        (0.95, "q095"),
        (0.99, "q099"),
    ):
        result[f"ensemble_{tag}"] = stack.quantile(q=quantile, dim=0)
    result["groundtruth"] = groundtruth
    result["ensemble_size"] = int(stack.shape[0])
    result["ensemble_member"] = -1
    result["ensemble_stat"] = "aggregate"
    result["prefix"] = prefix
    result["suffix"] = "ens_aggregate.pt"
    result["quantile_mapping_applied"] = True
    return result


def fit_from_validation(
    validation_tensor_dir: Path,
    output_dir: Path,
    output_names: list[str],
    n_quantiles: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "validation_tensor_dir": validation_tensor_dir.as_posix(),
        "n_quantiles": n_quantiles,
        "output_names": output_names,
    }
    for output_name in output_names:
        forecasts, observations = QuantileMapper.load_validation_tensors(
            tensor_dir=validation_tensor_dir,
            output_name=output_name,
        )
        mapper = QuantileMapper.fit(
            forecast_members=forecasts,
            observations=observations,
            output_name=output_name,
            n_quantiles=n_quantiles,
        )
        mapper.save(output_dir.joinpath(f"{output_name}_quantile_mapper.pt"))
        print(
            f"[quantile-mapping] fitted {output_name}: "
            f"forecasts={tuple(forecasts.shape)} observations={tuple(observations.shape)}"
        )
    torch.save(manifest, output_dir.joinpath("manifest.pt"))


def apply_to_tensor_dir(
    source_tensor_dir: Path,
    target_tensor_dir: Path,
    mapper_dir: Path,
    output_names: list[str],
) -> None:
    target_tensor_dir.mkdir(parents=True, exist_ok=True)
    for output_name in output_names:
        mapper = QuantileMapper.load(mapper_dir.joinpath(f"{output_name}_quantile_mapper.pt"))
        source_groups: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
        for path in QuantileMapper._member_paths(tensor_dir=source_tensor_dir, output_name=output_name):
            payload: dict[str, Any] = torch.load(path, map_location="cpu")
            source_groups.setdefault(str(payload["prefix"]), []).append((path, payload))

        group_sizes = Counter(len(items) for items in source_groups.values())
        expected_members = group_sizes.most_common(1)[0][0]
        grouped: dict[str, list[dict[str, Any]]] = {}
        skipped_groups = 0
        for prefix, items in source_groups.items():
            if len(items) != expected_members:
                skipped_groups += 1
                continue
            for path, payload in sorted(items, key=lambda item: item[0].name):
                calibrated_payload = copy.deepcopy(payload)
                prediction = torch.as_tensor(payload["prediction"], dtype=torch.float32)
                calibrated_payload["prediction"] = mapper.apply(prediction)
                calibrated_payload["quantile_mapping_applied"] = True
                calibrated_payload["quantile_mapper_dir"] = mapper_dir.as_posix()
                torch.save(calibrated_payload, target_tensor_dir.joinpath(path.name))
                grouped.setdefault(prefix, []).append(calibrated_payload)

        for prefix, member_payloads in grouped.items():
            aggregate = _aggregate_payload(member_payloads=member_payloads)
            torch.save(aggregate, target_tensor_dir.joinpath(f"{prefix}_ens_aggregate.pt"))
        if skipped_groups > 0:
            print(
                f"[quantile-mapping] skipped {skipped_groups} incomplete apply groups "
                f"for {output_name}; expected_members={expected_members}"
            )
        print(f"[quantile-mapping] applied {output_name}: groups={len(grouped)}")


def _complete_member_groups(tensor_dir: Path, output_name: str) -> tuple[dict[str, list[dict[str, Any]]], int, int]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for path in QuantileMapper._member_paths(tensor_dir=tensor_dir, output_name=output_name):
        payload: dict[str, Any] = torch.load(path, map_location="cpu")
        groups.setdefault(str(payload["prefix"]), []).append(payload)

    group_sizes = Counter(len(items) for items in groups.values())
    expected_members = group_sizes.most_common(1)[0][0]
    complete_groups: dict[str, list[dict[str, Any]]] = {}
    skipped_groups = 0
    for prefix, payloads in groups.items():
        if len(payloads) != expected_members:
            skipped_groups += 1
            continue
        complete_groups[prefix] = sorted(payloads, key=lambda item: int(item["ensemble_member"]))
    return complete_groups, expected_members, skipped_groups


def _pit_map(member_predictions: torch.Tensor, groundtruth: torch.Tensor) -> torch.Tensor:
    assert member_predictions.ndim == 3
    assert groundtruth.ndim == 2
    assert groundtruth.shape == member_predictions.shape[1:]
    n_members, height, width = member_predictions.shape
    if n_members == 1:
        return torch.full_like(groundtruth, fill_value=0.5, dtype=torch.float32)

    sorted_members = member_predictions.sort(dim=0).values
    sorted_flat = sorted_members.permute(1, 2, 0).reshape(height * width, n_members).contiguous()
    truth_flat = groundtruth.reshape(height * width, 1).contiguous()
    lower_idx = torch.searchsorted(sorted_flat, truth_flat, right=True).squeeze(dim=-1).to(dtype=torch.long)
    below_mask = lower_idx == 0
    above_mask = lower_idx == n_members
    lower_idx = lower_idx.clamp(min=1, max=n_members - 1)
    left_idx = lower_idx - 1
    right_idx = lower_idx

    left_value = torch.gather(sorted_flat, dim=1, index=left_idx.unsqueeze(dim=1)).squeeze(dim=1)
    right_value = torch.gather(sorted_flat, dim=1, index=right_idx.unsqueeze(dim=1)).squeeze(dim=1)
    denominator = right_value - left_value
    truth_values = truth_flat.squeeze(dim=1)
    same_value_mask = denominator == 0
    fraction = torch.zeros_like(truth_values, dtype=torch.float32)
    fraction[~same_value_mask] = (
        (truth_values[~same_value_mask] - left_value[~same_value_mask]) / denominator[~same_value_mask]
    ).to(dtype=torch.float32)
    fraction = fraction.clamp(min=0.0, max=1.0)

    rank = left_idx.to(dtype=torch.float32) + fraction
    if same_value_mask.any():
        equal_to_truth = sorted_flat == truth_flat
        first_equal = equal_to_truth.to(dtype=torch.int64).argmax(dim=1)
        last_equal = (n_members - 1) - equal_to_truth.flip(dims=(1,)).to(dtype=torch.int64).argmax(dim=1)
        rank[same_value_mask] = (
            first_equal[same_value_mask] + last_equal[same_value_mask]
        ).to(dtype=torch.float32) / 2.0

    rank[below_mask] = 0.0
    rank[above_mask] = float(n_members - 1)
    return (rank / float(n_members - 1)).reshape(height, width)


def plot_pit_for_tensor_dir(
    tensor_dir: Path,
    output_dir: Path,
    output_names: list[str],
    n_bins: int,
) -> None:
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for output_name in output_names:
        groups, expected_members, skipped_groups = _complete_member_groups(
            tensor_dir=tensor_dir,
            output_name=output_name,
        )
        pit_maps: list[torch.Tensor] = []
        for payloads in groups.values():
            members = [torch.as_tensor(item["prediction"], dtype=torch.float32) for item in payloads]
            groundtruth = torch.as_tensor(payloads[0]["groundtruth"], dtype=torch.float32)
            pit_maps.append(_pit_map(member_predictions=torch.stack(members, dim=0), groundtruth=groundtruth))
        if not pit_maps:
            raise ValueError(f"No complete groups found for output_name={output_name} in {tensor_dir}")

        pit_values = torch.stack(pit_maps, dim=0).reshape(-1).clamp(min=0.0, max=1.0)
        stats = {
            "tensor_dir": tensor_dir.as_posix(),
            "output_name": output_name,
            "n_groups": len(groups),
            "expected_members": expected_members,
            "skipped_groups": skipped_groups,
            "n_values": int(pit_values.numel()),
            "mean": float(pit_values.mean().item()),
            "std": float(pit_values.std(unbiased=False).item()),
            "p05": float(pit_values.quantile(0.05).item()),
            "p50": float(pit_values.quantile(0.50).item()),
            "p95": float(pit_values.quantile(0.95).item()),
            "pit_values": pit_values,
        }
        stats_path = output_dir.joinpath(f"{output_name}_pit_stats.pt")
        torch.save(stats, stats_path)

        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.hist(
            pit_values.cpu().numpy(),
            bins=n_bins,
            range=(0.0, 1.0),
            density=True,
            color="#4C78A8",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85,
        )
        ax.axhline(y=1.0, color="#D62728", linestyle="--", linewidth=1.4, label="Uniform")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("PIT")
        ax.set_ylabel("Density")
        ax.set_title(f"PIT Histogram: {output_name}")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(frameon=False)
        fig.tight_layout()
        figure_path = output_dir.joinpath(f"{output_name}_pit_histogram.png")
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        if skipped_groups > 0:
            print(
                f"[quantile-mapping] skipped {skipped_groups} incomplete PIT groups "
                f"for {output_name}; expected_members={expected_members}"
            )
        print(f"[quantile-mapping] saved PIT stats: {stats_path}")
        print(f"[quantile-mapping] saved PIT histogram: {figure_path}")


def _parse_output_names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--validation-tensor-dir", type=Path, required=True)
    fit_parser.add_argument("--output-dir", type=Path, required=True)
    fit_parser.add_argument("--output-names", type=_parse_output_names, default=["PRECT"])
    fit_parser.add_argument("--n-quantiles", type=int, default=101)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--source-tensor-dir", type=Path, required=True)
    apply_parser.add_argument("--target-tensor-dir", type=Path, required=True)
    apply_parser.add_argument("--mapper-dir", type=Path, required=True)
    apply_parser.add_argument("--output-names", type=_parse_output_names, default=["PRECT"])

    pit_parser = subparsers.add_parser("pit")
    pit_parser.add_argument("--tensor-dir", type=Path, required=True)
    pit_parser.add_argument("--output-dir", type=Path, required=True)
    pit_parser.add_argument("--output-names", type=_parse_output_names, default=["PRECT"])
    pit_parser.add_argument("--n-bins", type=int, default=20)

    args: Namespace = parser.parse_args()
    if args.command == "fit":
        fit_from_validation(
            validation_tensor_dir=args.validation_tensor_dir,
            output_dir=args.output_dir,
            output_names=args.output_names,
            n_quantiles=args.n_quantiles,
        )
    elif args.command == "apply":
        apply_to_tensor_dir(
            source_tensor_dir=args.source_tensor_dir,
            target_tensor_dir=args.target_tensor_dir,
            mapper_dir=args.mapper_dir,
            output_names=args.output_names,
        )
    elif args.command == "pit":
        plot_pit_for_tensor_dir(
            tensor_dir=args.tensor_dir,
            output_dir=args.output_dir,
            output_names=args.output_names,
            n_bins=args.n_bins,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
