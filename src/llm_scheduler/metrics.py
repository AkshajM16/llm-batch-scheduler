from __future__ import annotations

from dataclasses import asdict
from statistics import mean

import numpy as np

from llm_scheduler.types import GenerationResult


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, quantile))


def summarize_records(records: list[dict[str, object]]) -> dict[str, float | int]:
    successful = [record for record in records if record.get("status") == "ok"]
    total = len(records)
    elapsed_values = [float(record["total_latency_ms"]) for record in successful]
    queue_values = [float(record["queue_ms"]) for record in successful]
    inference_values = [float(record["inference_ms"]) for record in successful]
    duration_s = max((float(record.get("completed_offset_s", 0)) for record in records), default=0)
    generated_tokens = sum(int(record.get("generated_tokens", 0)) for record in successful)
    useful = sum(int(record.get("useful_prompt_tokens", 0)) for record in successful)
    padded = sum(int(record.get("padded_prompt_tokens", 0)) for record in successful)
    return {
        "requests": total,
        "successful_requests": len(successful),
        "requests_per_second": len(successful) / duration_s if duration_s else 0.0,
        "generated_tokens_per_second": generated_tokens / duration_s if duration_s else 0.0,
        "p50_latency_ms": percentile(elapsed_values, 50),
        "p95_latency_ms": percentile(elapsed_values, 95),
        "p99_latency_ms": percentile(elapsed_values, 99),
        "mean_queue_ms": mean(queue_values) if queue_values else 0.0,
        "mean_inference_ms": mean(inference_values) if inference_values else 0.0,
        "padding_overhead_ratio": (padded - useful) / useful if useful else 0.0,
        "error_rate": (total - len(successful)) / total if total else 0.0,
    }


class MetricsRegistry:
    def __init__(self) -> None:
        self.results: list[dict[str, object]] = []
        self.errors: dict[str, int] = {}

    def record_result(self, result: GenerationResult) -> None:
        self.results.append(asdict(result))

    def record_error(self, error_type: str) -> None:
        self.errors[error_type] = self.errors.get(error_type, 0) + 1

    def snapshot(self) -> dict[str, object]:
        latencies = [float(item["total_latency_ms"]) for item in self.results]
        queue_times = [float(item["queue_ms"]) for item in self.results]
        inference_times = [float(item["inference_ms"]) for item in self.results]
        return {
            "completed": len(self.results),
            "errors": dict(self.errors),
            "latency_ms": {
                "p50": percentile(latencies, 50),
                "p95": percentile(latencies, 95),
                "p99": percentile(latencies, 99),
            },
            "queue_ms": {"mean": mean(queue_times) if queue_times else 0.0},
            "inference_ms": {"mean": mean(inference_times) if inference_times else 0.0},
        }
