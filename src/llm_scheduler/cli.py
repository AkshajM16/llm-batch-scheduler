from __future__ import annotations

import argparse

import uvicorn

from llm_scheduler.app import create_app
from llm_scheduler.config import Settings


def build_parser() -> argparse.ArgumentParser:
    defaults = Settings()
    parser = argparse.ArgumentParser(description="Serve the LLM inference scheduler")
    parser.add_argument(
        "--mode", choices=["sequential", "dynamic"], default=defaults.scheduler_mode
    )
    parser.add_argument("--backend", choices=["transformers", "fake"], default=defaults.backend)
    parser.add_argument("--model", default=defaults.model_id)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=defaults.device)
    parser.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default=defaults.dtype
    )
    parser.add_argument("--max-batch-size", type=int, default=defaults.max_batch_size)
    parser.add_argument("--max-wait-ms", type=float, default=defaults.max_wait_ms)
    parser.add_argument("--max-queue-size", type=int, default=defaults.max_queue_size)
    parser.add_argument("--request-timeout-s", type=float, default=defaults.request_timeout_s)
    parser.add_argument("--length-buckets", default=",".join(map(str, defaults.length_buckets)))
    parser.add_argument("--max-input-tokens", type=int, default=defaults.max_input_tokens)
    parser.add_argument("--min-new-tokens", type=int, default=defaults.min_new_tokens)
    parser.add_argument("--max-new-tokens", type=int, default=defaults.max_new_tokens)
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--host", default=defaults.host)
    parser.add_argument("--port", type=int, default=defaults.port)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings(
        scheduler_mode=args.mode,
        backend=args.backend,
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        max_batch_size=args.max_batch_size,
        max_wait_ms=args.max_wait_ms,
        max_queue_size=args.max_queue_size,
        request_timeout_s=args.request_timeout_s,
        length_buckets=args.length_buckets,
        max_input_tokens=args.max_input_tokens,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        host=args.host,
        port=args.port,
    )
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
