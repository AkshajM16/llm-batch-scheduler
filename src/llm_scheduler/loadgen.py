from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

from llm_scheduler.metrics import summarize_records

SOURCE_ROOT = Path(__file__).resolve().parents[2]
ROOT = SOURCE_ROOT if (SOURCE_ROOT / "pyproject.toml").exists() else Path.cwd()
PACKAGED_CORPUS = Path(__file__).resolve().parent / "data" / "prompts.jsonl"
DEFAULT_CORPUS = ROOT / "data" / "prompts.jsonl"
if not DEFAULT_CORPUS.exists():
    DEFAULT_CORPUS = PACKAGED_CORPUS


def load_corpus(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        corpus = [json.loads(line) for line in handle if line.strip()]
    if not corpus:
        raise ValueError(f"empty corpus: {path}")
    return corpus


async def run_load(
    base_url: str,
    corpus: list[dict[str, str]],
    concurrency: int,
    request_count: int,
    warmup: int = 2,
    timeout_s: float = 60,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    limits = httpx.Limits(
        max_connections=max(concurrency, 1), max_keepalive_connections=concurrency
    )
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s, limits=limits) as client:
        for index in range(warmup):
            item = corpus[index % len(corpus)]
            response = await client.post(
                "/v1/generate",
                json={"prompt": item["prompt"]},
                headers={"x-request-id": f"warmup-{index}"},
            )
            response.raise_for_status()

        started = perf_counter()
        next_index = 0
        lock = asyncio.Lock()
        records: list[dict[str, Any]] = []

        async def worker(worker_id: int) -> None:
            nonlocal next_index
            while True:
                async with lock:
                    if next_index >= request_count:
                        return
                    index = next_index
                    next_index += 1
                item = corpus[index % len(corpus)]
                request_id = f"load-{worker_id}-{index}-{item['id']}"
                sent = perf_counter()
                try:
                    response = await client.post(
                        "/v1/generate",
                        json={"prompt": item["prompt"]},
                        headers={"x-request-id": request_id},
                    )
                    completed = perf_counter()
                    if response.status_code != 200:
                        records.append(
                            {
                                "request_id": request_id,
                                "prompt_id": item["id"],
                                "status": "error",
                                "http_status": response.status_code,
                                "response_body": response.text[:500],
                                "client_latency_ms": (completed - sent) * 1_000,
                                "completed_offset_s": completed - started,
                            }
                        )
                        continue
                    payload = response.json()
                    if payload["request_id"] != request_id:
                        raise RuntimeError(
                            f"response correlation failure: {request_id} != {payload['request_id']}"
                        )
                    payload.update(
                        {
                            "prompt_id": item["id"],
                            "status": "ok",
                            "http_status": response.status_code,
                            "client_latency_ms": (completed - sent) * 1_000,
                            "completed_offset_s": completed - started,
                        }
                    )
                    records.append(payload)
                except (httpx.HTTPError, RuntimeError) as exc:
                    completed = perf_counter()
                    records.append(
                        {
                            "request_id": request_id,
                            "prompt_id": item["id"],
                            "status": "error",
                            "error": type(exc).__name__,
                            "client_latency_ms": (completed - sent) * 1_000,
                            "completed_offset_s": completed - started,
                        }
                    )

        await asyncio.gather(*(worker(worker_id) for worker_id in range(concurrency)))
    records.sort(key=lambda record: str(record["request_id"]))
    return records, summarize_records(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a deterministic concurrent workload")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=60)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records, summary = asyncio.run(
        run_load(
            args.url,
            load_corpus(args.corpus),
            args.concurrency,
            args.requests,
            args.warmup,
            args.timeout_s,
        )
    )
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "config": {"concurrency": args.concurrency, "requests": args.requests},
        "summary": summary,
        "records": records,
    }
    output = args.output or ROOT / "artifacts" / "raw" / f"load-{datetime.now():%Y%m%d-%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
