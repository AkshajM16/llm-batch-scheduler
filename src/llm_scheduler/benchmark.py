from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import torch

from llm_scheduler.loadgen import DEFAULT_CORPUS, ROOT, load_corpus, run_load


@dataclass(frozen=True)
class RunConfig:
    label: str
    mode: str
    concurrency: int
    max_batch_size: int
    max_wait_ms: float
    length_buckets: str


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return path.name


def available_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def machine_metadata() -> dict[str, Any]:
    packages = ["fastapi", "httpx", "numpy", "torch", "transformers", "uvicorn"]
    memory_bytes: int | None = None
    try:
        memory_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    except (OSError, ValueError, subprocess.CalledProcessError):
        if hasattr(os, "sysconf"):
            with suppress(OSError, ValueError):
                memory_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "memory_bytes": memory_bytes,
        "torch_mps_available": torch.backends.mps.is_available(),
        "torch_cuda_available": torch.cuda.is_available(),
        "dependency_versions": {
            package: importlib.metadata.version(package) for package in packages
        },
    }


def configs_for_preset(preset: str) -> list[RunConfig]:
    if preset == "full":
        configs = [
            RunConfig("sequential", "sequential", concurrency, 1, 0, "32,128,512")
            for concurrency in (1, 4, 8, 16, 32)
        ]
        configs.extend(
            RunConfig(
                "dynamic_bucketed",
                "dynamic",
                16,
                batch_size,
                wait_ms,
                "32,128,512",
            )
            for batch_size in (1, 2, 4, 8)
            for wait_ms in (0, 2, 5, 10, 20)
        )
        configs.append(RunConfig("dynamic_unbucketed", "dynamic", 16, 4, 5, "1000000"))
        return configs
    return [
        *[
            RunConfig("sequential", "sequential", concurrency, 1, 0, "32,128,512")
            for concurrency in (1, 4, 8, 16, 32)
        ],
        RunConfig("dynamic_bucketed", "dynamic", 8, 2, 2, "32,128,512"),
        RunConfig("dynamic_bucketed", "dynamic", 8, 4, 5, "32,128,512"),
        RunConfig("dynamic_bucketed", "dynamic", 8, 8, 10, "32,128,512"),
        RunConfig("dynamic_unbucketed", "dynamic", 8, 4, 5, "1000000"),
    ]


async def wait_until_ready(
    url: str, process: asyncio.subprocess.Process, timeout_s: float = 180
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    async with httpx.AsyncClient(timeout=2) as client:
        while asyncio.get_running_loop().time() < deadline:
            if process.returncode is not None:
                stdout, _ = await process.communicate()
                message = stdout.decode(errors="replace")[-4000:] if stdout else ""
                raise RuntimeError(f"server exited during startup:\n{message}")
            try:
                response = await client.get(f"{url}/health")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.25)
    raise TimeoutError("server did not become healthy")


async def benchmark_one(
    config: RunConfig,
    args: argparse.Namespace,
    trial: int,
) -> dict[str, Any]:
    port = available_port()
    url = f"http://127.0.0.1:{port}"
    command = [
        sys.executable,
        "-m",
        "llm_scheduler.cli",
        "--mode",
        config.mode,
        "--backend",
        args.backend,
        "--model",
        args.model,
        "--device",
        args.device,
        "--max-batch-size",
        str(config.max_batch_size),
        "--max-wait-ms",
        str(config.max_wait_ms),
        "--length-buckets",
        config.length_buckets,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--min-new-tokens",
        str(args.min_new_tokens),
        "--port",
        str(port),
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        await wait_until_ready(url, process)
        records, summary = await run_load(
            url,
            load_corpus(args.corpus),
            config.concurrency,
            args.requests,
            args.warmup,
            args.timeout_s,
        )
        return {
            "config": {**asdict(config), "trial": trial},
            "summary": summary,
            "records": records,
        }
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=20)
            except TimeoutError:
                process.kill()
                await asyncio.wait_for(process.wait(), timeout=5)


async def run_benchmark(args: argparse.Namespace) -> Path:
    configs = configs_for_preset(args.preset)
    payload: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark_type": "real-model" if args.backend == "transformers" else "scheduler-only-fake",
        "preset": args.preset,
        "model": args.model,
        "backend": args.backend,
        "device": args.device,
        "generation": {
            "min_new_tokens": args.min_new_tokens,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.0,
            "seed": 7,
        },
        "corpus": portable_path(args.corpus),
        "requests_per_trial": args.requests,
        "warmup_requests": args.warmup,
        "trials": args.trials,
        "machine": machine_metadata(),
        "runs": [],
    }
    total = len(configs) * args.trials
    completed = 0
    for config in configs:
        for trial in range(1, args.trials + 1):
            completed += 1
            print(
                f"[{completed}/{total}] {config.label} c={config.concurrency} "
                f"b={config.max_batch_size} wait={config.max_wait_ms}ms trial={trial}",
                flush=True,
            )
            payload["runs"].append(await benchmark_one(config, args, trial))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = args.output or ROOT / "artifacts" / "raw" / f"benchmark-{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible scheduler benchmark sweeps")
    parser.add_argument("--preset", choices=["representative", "full"], default="representative")
    parser.add_argument("--backend", choices=["transformers", "fake"], default="transformers")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        help="defaults to --max-new-tokens so benchmark work is fixed",
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--requests", type=int, default=36)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=120)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.min_new_tokens is None:
        args.min_new_tokens = args.max_new_tokens
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
