"""Command-line entry point for reproducible policy experiments."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .policies import Model, Replica, build_policies
from .simulator import SimulationResult, run_simulation
from .workloads import generate_workload


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    required = {"seed", "request_count", "duration_seconds", "scenario", "models", "replicas"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"configuration missing keys: {', '.join(sorted(missing))}")
    return config


def _models(config: dict[str, Any]) -> list[Model]:
    return [
        Model(
            model_id=item["id"], quality=float(item["quality"]),
            cost_per_million_tokens=float(item["cost_per_million_tokens"]),
            prefill_tokens_per_second=float(item["prefill_tokens_per_second"]),
            decode_tokens_per_second=float(item["decode_tokens_per_second"]),
            max_context_tokens=int(item["max_context_tokens"]),
            privacy_levels=frozenset(item["privacy_levels"]),
            reliability=float(item["reliability"]),
        )
        for item in config["models"]
    ]


def _replicas(config: dict[str, Any]) -> list[Replica]:
    return [
        Replica(
            replica_id=item["id"], model_id=item["model_id"],
            speed_factor=float(item["speed_factor"]), capacity=int(item["capacity"]),
            reliability=float(item["reliability"]),
        )
        for item in config["replicas"]
    ]


def run_experiment(config: dict[str, Any]) -> list[SimulationResult]:
    requests = generate_workload(
        int(config["request_count"]), float(config["duration_seconds"]),
        int(config["seed"]), str(config["scenario"]),
    )
    models = _models(config)
    replicas = _replicas(config)
    return [
        run_simulation(
            requests, models, replicas, policy, int(config["seed"]),
            float(config["duration_seconds"]),
            bool(config.get("faults_enabled", False)) or config["scenario"] == "fault_injected",
        )
        for policy in build_policies(config.get("adaptive_weights"))
    ]


def _write_results(results: list[SimulationResult], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summaries = [result.summary for result in results]
    (output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    request_fields = list(asdict(results[0].requests[0]).keys()) if results and results[0].requests else []
    if request_fields:
        with (output / "requests.csv").open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=request_fields)
            writer.writeheader()
            for result in results:
                writer.writerows(asdict(item) for item in result.requests)
    _write_svg(summaries, output / "comparison.svg")


def _write_svg(summaries: list[dict[str, Any]], path: Path) -> None:
    width, height = 920, 430
    left, top, plot_width, plot_height = 74, 46, 800, 300
    max_latency = max((float(row["p95_latency_seconds"]) for row in summaries), default=1.0) or 1.0
    max_cost = max((float(row["estimated_cost"]) for row in summaries), default=1.0) or 1.0
    bars: list[str] = []
    labels: list[str] = []
    group_width = plot_width / max(len(summaries), 1)
    for index, row in enumerate(summaries):
        x = left + index * group_width
        latency_height = float(row["p95_latency_seconds"]) / max_latency * plot_height
        cost_height = float(row["estimated_cost"]) / max_cost * plot_height
        bars.append(f'<rect x="{x + 7:.1f}" y="{top + plot_height - latency_height:.1f}" width="{group_width * .30:.1f}" height="{latency_height:.1f}" fill="#147d72"><title>p95 latency: {row["p95_latency_seconds"]:.3f}s</title></rect>')
        bars.append(f'<rect x="{x + group_width * .38:.1f}" y="{top + plot_height - cost_height:.1f}" width="{group_width * .30:.1f}" height="{cost_height:.1f}" fill="#dd7a35"><title>estimated cost: {row["estimated_cost"]:.5f}</title></rect>')
        labels.append(f'<text x="{x + group_width / 2:.1f}" y="{top + plot_height + 22}" text-anchor="middle" font-size="12">{row["policy"]}</text>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#f5f3ed"/><text x="74" y="25" font-size="19" font-family="sans-serif" fill="#202a28">Policy comparison</text>'
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#53615d"/>'
        f'{"".join(bars)}{"".join(labels)}'
        '<rect x="690" y="14" width="12" height="12" fill="#147d72"/><text x="708" y="25" font-size="12">p95 latency (scaled)</text>'
        '<rect x="824" y="14" width="12" height="12" fill="#dd7a35"/><text x="842" y="25" font-size="12">cost (scaled)</text>'
        '</svg>'
    )
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(prog="llm-serving-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run all policy comparisons")
    run_parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    run_parser.add_argument("--output", type=Path, default=Path("results"))
    run_parser.add_argument("--scenario", choices=["bursty", "heterogeneous", "fault_injected"])
    run_parser.add_argument("--seed", type=int)
    run_parser.add_argument("--requests", type=int)
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.scenario:
        config["scenario"] = args.scenario
    if args.seed is not None:
        config["seed"] = args.seed
    if args.requests is not None:
        config["request_count"] = args.requests
    results = run_experiment(config)
    _write_results(results, args.output)
    print(f"Wrote {len(results)} policy summaries to {args.output}")


if __name__ == "__main__":
    main()