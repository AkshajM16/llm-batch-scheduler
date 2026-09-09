from __future__ import annotations

import asyncio
from time import perf_counter

import pytest

from llm_scheduler.backend import FakeBackend
from llm_scheduler.config import Settings
from llm_scheduler.scheduler import Scheduler
from llm_scheduler.types import (
    BatchOutput,
    PendingRequest,
    QueueOverloadedError,
    RequestExpiredError,
)


class RecordingBackend(FakeBackend):
    def __init__(self, delay_ms: float = 1) -> None:
        super().__init__(delay_ms)
        self.batches: list[list[str]] = []

    async def generate(self, prompts: list[str]) -> BatchOutput:
        self.batches.append(prompts)
        return await super().generate(prompts)


class BlockingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__(0)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, prompts: list[str]) -> BatchOutput:
        self.started.set()
        await self.release.wait()
        return await super().generate(prompts)


def settings(**overrides: object) -> Settings:
    values = {
        "backend": "fake",
        "scheduler_mode": "dynamic",
        "max_batch_size": 4,
        "max_wait_ms": 25,
        "max_queue_size": 8,
        "request_timeout_s": 1,
        "length_buckets": (4, 20),
    }
    values.update(overrides)
    return Settings(**values)


@pytest.mark.asyncio
async def test_dispatches_immediately_when_batch_is_full() -> None:
    backend = RecordingBackend()
    scheduler = Scheduler(backend, settings(max_batch_size=2, max_wait_ms=1_000))
    await scheduler.start()
    first, second = await asyncio.gather(
        scheduler.submit("a", "one two"),
        scheduler.submit("b", "three four"),
    )
    await scheduler.close()

    assert len(backend.batches) == 1
    assert backend.batches[0] == ["one two", "three four"]
    assert first.batch_size == second.batch_size == 2


@pytest.mark.asyncio
async def test_dispatches_partial_batch_at_wait_deadline() -> None:
    backend = RecordingBackend()
    scheduler = Scheduler(backend, settings(max_batch_size=4, max_wait_ms=12))
    await scheduler.start()
    result = await scheduler.submit("a", "one")
    await scheduler.close()

    assert len(backend.batches) == 1
    assert result.batch_size == 1
    assert result.queue_ms >= 8


@pytest.mark.asyncio
async def test_expired_partial_bucket_precedes_another_full_bucket() -> None:
    backend = RecordingBackend()
    scheduler = Scheduler(backend, settings(max_batch_size=2, max_wait_ms=10))
    loop = asyncio.get_running_loop()
    expired = PendingRequest("old", "one", loop.create_future())
    expired.arrived_at = perf_counter() - 1
    full = [
        PendingRequest(
            f"new-{index}", "this prompt belongs in the longer bucket", loop.create_future()
        )
        for index in range(2)
    ]
    scheduler._buckets[4].append(expired)
    scheduler._buckets[20].extend(full)
    scheduler._queued = 3

    batch = await scheduler._next_batch()

    assert batch is not None
    assert [item.request_id for item in batch] == ["old"]


@pytest.mark.asyncio
async def test_length_buckets_do_not_mix_dissimilar_prompts() -> None:
    backend = RecordingBackend()
    scheduler = Scheduler(backend, settings(max_batch_size=2, max_wait_ms=5))
    await scheduler.start()
    results = await asyncio.gather(
        scheduler.submit("short", "tiny prompt"),
        scheduler.submit("long", "this prompt has enough words to land in another bucket"),
    )
    await scheduler.close()

    assert scheduler.bucket_for_length(2) == 4
    assert scheduler.bucket_for_length(10) == 20
    assert len(backend.batches) == 2
    assert all(result.batch_size == 1 for result in results)


@pytest.mark.asyncio
async def test_response_correlation_under_concurrency() -> None:
    backend = RecordingBackend()
    scheduler = Scheduler(backend, settings(max_batch_size=4, max_wait_ms=2, max_queue_size=32))
    await scheduler.start()
    prompts = [f"prompt number {index}" for index in range(24)]
    results = await asyncio.gather(
        *(scheduler.submit(f"id-{index}", prompt) for index, prompt in enumerate(prompts))
    )
    await scheduler.close()

    assert [result.request_id for result in results] == [f"id-{index}" for index in range(24)]
    assert [result.generated_text for result in results] == [
        f"response:{prompt}" for prompt in prompts
    ]


@pytest.mark.asyncio
async def test_timeout_does_not_deliver_late_result() -> None:
    backend = RecordingBackend(delay_ms=40)
    scheduler = Scheduler(
        backend,
        settings(max_batch_size=1, max_wait_ms=0, request_timeout_s=0.005),
    )
    await scheduler.start()
    with pytest.raises(RequestExpiredError):
        await scheduler.submit("expired", "too slow")
    await scheduler.close()

    assert scheduler.queued == 0
    assert scheduler.metrics.results == []


@pytest.mark.asyncio
async def test_cancelled_queued_request_is_removed() -> None:
    backend = BlockingBackend()
    scheduler = Scheduler(backend, settings(max_batch_size=1, max_wait_ms=0))
    await scheduler.start()
    first = asyncio.create_task(scheduler.submit("first", "first"))
    await backend.started.wait()
    second = asyncio.create_task(scheduler.submit("second", "second"))
    await asyncio.sleep(0)
    assert scheduler.queued == 1
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    assert scheduler.queued == 0
    backend.release.set()
    await first
    await scheduler.close()


@pytest.mark.asyncio
async def test_bounded_queue_rejects_overload() -> None:
    backend = BlockingBackend()
    scheduler = Scheduler(
        backend,
        settings(max_batch_size=1, max_wait_ms=0, max_queue_size=1),
    )
    await scheduler.start()
    first = asyncio.create_task(scheduler.submit("first", "first"))
    await backend.started.wait()
    second = asyncio.create_task(scheduler.submit("second", "second"))
    await asyncio.sleep(0)
    with pytest.raises(QueueOverloadedError):
        await scheduler.submit("third", "third")
    backend.release.set()
    await asyncio.gather(first, second)
    await scheduler.close()


@pytest.mark.asyncio
async def test_sequential_mode_never_batches() -> None:
    backend = RecordingBackend()
    scheduler = Scheduler(
        backend,
        settings(scheduler_mode="sequential", max_batch_size=8, max_wait_ms=100),
    )
    await scheduler.start()
    await asyncio.gather(*(scheduler.submit(str(i), f"request {i}") for i in range(5)))
    await scheduler.close()

    assert [len(batch) for batch in backend.batches] == [1, 1, 1, 1, 1]
