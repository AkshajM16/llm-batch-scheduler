from __future__ import annotations

import asyncio
from collections import deque
from contextlib import suppress
from time import perf_counter, time

from llm_scheduler.backend import GenerationBackend
from llm_scheduler.config import Settings
from llm_scheduler.metrics import MetricsRegistry
from llm_scheduler.types import (
    GenerationResult,
    PendingRequest,
    QueueOverloadedError,
    RequestExpiredError,
)


class Scheduler:
    """A bounded, correlation-safe scheduler supporting sequential and dynamic modes."""

    def __init__(
        self,
        backend: GenerationBackend,
        settings: Settings,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self.backend = backend
        self.settings = settings
        self.metrics = metrics or MetricsRegistry()
        self._condition = asyncio.Condition()
        self._buckets: dict[int, deque[PendingRequest]] = {
            boundary: deque() for boundary in (*settings.length_buckets, 2**31 - 1)
        }
        self._queued = 0
        self._closing = False
        self._worker: asyncio.Task[None] | None = None

    @property
    def queued(self) -> int:
        return self._queued

    @property
    def bucket_sizes(self) -> dict[int, int]:
        return {boundary: len(queue) for boundary, queue in self._buckets.items()}

    def bucket_for_length(self, token_count: int) -> int:
        return next(boundary for boundary in self._buckets if token_count <= boundary)

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="batch-scheduler")

    async def close(self) -> None:
        async with self._condition:
            self._closing = True
            self._condition.notify_all()
        if self._worker is not None:
            with suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

    async def submit(self, request_id: str, prompt: str) -> GenerationResult:
        if self._worker is None:
            raise RuntimeError("scheduler has not been started")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[GenerationResult] = loop.create_future()
        pending = PendingRequest(request_id=request_id, prompt=prompt, future=future)
        boundary = self.bucket_for_length(self.backend.estimate_tokens(prompt))
        async with self._condition:
            if self._closing:
                raise RuntimeError("scheduler is closing")
            if self._queued >= self.settings.max_queue_size:
                raise QueueOverloadedError("inference queue is full")
            self._buckets[boundary].append(pending)
            self._queued += 1
            self._condition.notify_all()
        try:
            return await asyncio.wait_for(
                asyncio.shield(future), timeout=self.settings.request_timeout_s
            )
        except TimeoutError as exc:
            await self._remove_if_queued(pending)
            future.cancel()
            raise RequestExpiredError("request exceeded its end-to-end timeout") from exc
        except asyncio.CancelledError:
            await self._remove_if_queued(pending)
            future.cancel()
            raise

    async def _remove_if_queued(self, target: PendingRequest) -> None:
        async with self._condition:
            for queue in self._buckets.values():
                with suppress(ValueError):
                    queue.remove(target)
                    self._queued -= 1
                    break
            self._condition.notify_all()

    def _oldest_bucket(self) -> int | None:
        candidates = [
            (queue[0].arrived_at, boundary) for boundary, queue in self._buckets.items() if queue
        ]
        return min(candidates)[1] if candidates else None

    def _full_bucket(self) -> int | None:
        candidates = [
            (queue[0].arrived_at, boundary)
            for boundary, queue in self._buckets.items()
            if len(queue) >= self.settings.max_batch_size
        ]
        return min(candidates)[1] if candidates else None

    def _pop_batch(self, boundary: int, size: int) -> list[PendingRequest]:
        queue = self._buckets[boundary]
        batch: list[PendingRequest] = []
        while queue and len(batch) < size:
            item = queue.popleft()
            self._queued -= 1
            if not item.future.done():
                batch.append(item)
        return batch

    async def _next_batch(self) -> list[PendingRequest] | None:
        async with self._condition:
            while True:
                if self._closing and self._queued == 0:
                    return None
                boundary = self._oldest_bucket()
                if boundary is None:
                    await self._condition.wait()
                    continue
                if self.settings.scheduler_mode == "sequential":
                    return self._pop_batch(boundary, 1)
                oldest = self._buckets[boundary][0]
                deadline = oldest.arrived_at + self.settings.max_wait_ms / 1_000
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    return self._pop_batch(boundary, self.settings.max_batch_size)
                full_boundary = self._full_bucket()
                if full_boundary is not None:
                    return self._pop_batch(full_boundary, self.settings.max_batch_size)
                if self._closing:
                    return self._pop_batch(boundary, self.settings.max_batch_size)
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except TimeoutError:
                    return self._pop_batch(boundary, self.settings.max_batch_size)

    async def _run(self) -> None:
        while True:
            batch = await self._next_batch()
            if batch is None:
                return
            if not batch:
                continue
            started = perf_counter()
            queue_times = [(started - item.arrived_at) * 1_000 for item in batch]
            try:
                batch_output = await self.backend.generate([item.prompt for item in batch])
                finished = perf_counter()
                completed_time_unix_s = time()
                if len(batch_output.outputs) != len(batch):
                    raise RuntimeError("backend returned a different number of outputs")
                inference_ms = (finished - started) * 1_000
                for item, output, queue_ms in zip(
                    batch, batch_output.outputs, queue_times, strict=True
                ):
                    result = GenerationResult(
                        request_id=item.request_id,
                        arrival_time_unix_s=item.arrival_time_unix_s,
                        completed_time_unix_s=completed_time_unix_s,
                        generated_text=output.generated_text,
                        prompt_tokens=output.prompt_tokens,
                        generated_tokens=output.generated_tokens,
                        queue_ms=queue_ms,
                        inference_ms=inference_ms,
                        total_latency_ms=(finished - item.arrived_at) * 1_000,
                        batch_size=len(batch),
                        padded_prompt_tokens=batch_output.padded_prompt_tokens // len(batch),
                        useful_prompt_tokens=output.prompt_tokens,
                    )
                    if not item.future.done():
                        item.future.set_result(result)
                        self.metrics.record_result(result)
            except Exception as exc:
                self.metrics.record_error("backend")
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)
