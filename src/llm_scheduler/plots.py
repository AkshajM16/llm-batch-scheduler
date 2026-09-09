from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load_runs(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        for run in payload["runs"]:
            rows.append(
                {
                    "source": path.name,
                    "benchmark_type": payload["benchmark_type"],
                    "model": payload["model"],
                    **run["config"],
                    **run["summary"],
                }
            )
    if not rows:
        raise ValueError("no benchmark runs found")
    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "benchmark_type",
        "model",
        "label",
        "mode",
        "concurrency",
        "max_batch_size",
        "max_wait_ms",
        "length_buckets",
    ]
    numeric = [column for column in frame.select_dtypes("number").columns if column != "trial"]
    return frame.groupby(keys, as_index=False)[numeric].mean()


def pareto_rows(frame: pd.DataFrame) -> pd.DataFrame:
    mask: list[bool] = []
    for _index, row in frame.iterrows():
        dominated = (
            (frame["requests_per_second"] >= row["requests_per_second"])
            & (frame["p95_latency_ms"] <= row["p95_latency_ms"])
            & (
                (frame["requests_per_second"] > row["requests_per_second"])
                | (frame["p95_latency_ms"] < row["p95_latency_ms"])
            )
        ).any()
        mask.append(not bool(dominated))
    return frame.loc[mask].sort_values("requests_per_second")


def create_plots(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    written: list[Path] = []

    sequential = frame[frame["mode"] == "sequential"].sort_values("concurrency")
    if not sequential.empty:
        fig, axis = plt.subplots(figsize=(7, 4.5))
        axis.plot(sequential["concurrency"], sequential["requests_per_second"], marker="o")
        axis.set(
            title="Sequential throughput vs client concurrency",
            xlabel="Concurrency",
            ylabel="Requests / second",
        )
        fig.tight_layout()
        path = output_dir / "throughput-vs-concurrency.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    for label, group in frame.groupby("label"):
        axis.scatter(group["requests_per_second"], group["p95_latency_ms"], label=label, s=55)
    frontier = pareto_rows(frame)
    axis.plot(
        frontier["requests_per_second"],
        frontier["p95_latency_ms"],
        "k--",
        alpha=0.7,
        label="Pareto frontier",
    )
    axis.set(
        title="p95 latency / throughput trade-off",
        xlabel="Requests / second",
        ylabel="p95 latency (ms)",
    )
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "p95-vs-throughput.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    policies = frame[frame["label"].isin(["dynamic_bucketed", "dynamic_unbucketed"])]
    if not policies.empty:
        matched = policies[
            (policies["concurrency"] == 8)
            & (policies["max_batch_size"] == 4)
            & (policies["max_wait_ms"] == 5)
        ]
        if set(matched["label"]) == {"dynamic_bucketed", "dynamic_unbucketed"}:
            policies = matched
        values = policies.groupby("label")["padding_overhead_ratio"].apply(mean)
        fig, axis = plt.subplots(figsize=(7, 4.5))
        values.mul(100).plot.bar(ax=axis, color=["#4472C4", "#ED7D31"])
        axis.set(
            title="Padding overhead by batching policy",
            xlabel="Policy",
            ylabel="Padding overhead (%)",
        )
        axis.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        path = output_dir / "padding-overhead.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)
    return written


def write_summary(frame: pd.DataFrame, path: Path) -> None:
    frontier = pareto_rows(frame)
    lines = [
        "# Measured benchmark summary",
        "",
        frame.to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Measured Pareto frontier",
        "",
        frontier.to_markdown(index=False, floatfmt=".2f"),
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot scheduler benchmark results")
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "plots")
    args = parser.parse_args()
    inputs = args.inputs or sorted((ROOT / "artifacts" / "raw").glob("benchmark-*.json"))
    frame = aggregate(load_runs(inputs))
    paths = create_plots(frame, args.output_dir)
    write_summary(frame, args.output_dir / "RESULTS.md")
    frame.to_csv(args.output_dir / "summary.csv", index=False)
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
