from llm_scheduler.metrics import percentile, summarize_records


def test_percentile_empty_is_zero() -> None:
    assert percentile([], 95) == 0


def test_metric_calculations() -> None:
    records = [
        {
            "status": "ok",
            "total_latency_ms": 10,
            "queue_ms": 2,
            "inference_ms": 8,
            "generated_tokens": 4,
            "useful_prompt_tokens": 5,
            "padded_prompt_tokens": 8,
            "completed_offset_s": 1,
        },
        {
            "status": "ok",
            "total_latency_ms": 30,
            "queue_ms": 6,
            "inference_ms": 24,
            "generated_tokens": 4,
            "useful_prompt_tokens": 7,
            "padded_prompt_tokens": 8,
            "completed_offset_s": 2,
        },
        {"status": "error", "completed_offset_s": 2},
    ]
    summary = summarize_records(records)

    assert summary["requests"] == 3
    assert summary["successful_requests"] == 2
    assert summary["requests_per_second"] == 1
    assert summary["generated_tokens_per_second"] == 4
    assert summary["p50_latency_ms"] == 20
    assert summary["mean_queue_ms"] == 4
    assert summary["mean_inference_ms"] == 16
    assert summary["padding_overhead_ratio"] == 4 / 12
    assert summary["error_rate"] == 1 / 3
