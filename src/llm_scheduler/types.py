from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from time import perf_counter, time


@dataclass(slots=True)
class BackendOutput:
    generated_text: str
    prompt_tokens: int
    generated_tokens: int


@dataclass(slots=True)
class BatchOutput:
    outputs: list[BackendOutput]
    padded_prompt_tokens: int
    useful_prompt_tokens: int


@dataclass(slots=True)
class GenerationResult:
    request_id: str
    arrival_time_unix_s: float
    completed_time_unix_s: float
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    queue_ms: float
    inference_ms: float
    total_latency_ms: float
    batch_size: int
    padded_prompt_tokens: int
    useful_prompt_tokens: int


@dataclass(slots=True)
class PendingRequest:
    request_id: str
    prompt: str
    future: asyncio.Future[GenerationResult]
    arrived_at: float = field(default_factory=perf_counter)
    arrival_time_unix_s: float = field(default_factory=time)


class QueueOverloadedError(RuntimeError):
    """Raised when bounded scheduler capacity has been exhausted."""


class RequestExpiredError(TimeoutError):
    """Raised when a request exceeds its end-to-end timeout."""
