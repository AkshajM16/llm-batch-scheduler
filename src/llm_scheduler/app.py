from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from llm_scheduler import __version__
from llm_scheduler.backend import create_backend
from llm_scheduler.config import Settings, get_settings
from llm_scheduler.metrics import MetricsRegistry
from llm_scheduler.scheduler import Scheduler
from llm_scheduler.types import QueueOverloadedError, RequestExpiredError

LOGGER = logging.getLogger(__name__)


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)


class GenerationResponse(BaseModel):
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


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        metrics = MetricsRegistry()
        scheduler = Scheduler(create_backend(resolved), resolved, metrics)
        await scheduler.start()
        app.state.scheduler = scheduler
        app.state.metrics = metrics
        yield
        await scheduler.close()

    app = FastAPI(
        title="Dynamic-Batching LLM Inference Scheduler", version=__version__, lifespan=lifespan
    )

    @app.get("/health")
    async def health(request: Request) -> dict[str, object]:
        scheduler: Scheduler = request.app.state.scheduler
        return {
            "status": "ok",
            "mode": resolved.scheduler_mode,
            "backend": resolved.backend,
            "model": resolved.model_id,
            "device": str(getattr(scheduler.backend, "device", "synthetic")),
            "queued": scheduler.queued,
        }

    @app.get("/metrics")
    async def metrics(request: Request) -> dict[str, object]:
        return request.app.state.metrics.snapshot()

    @app.post("/v1/generate", response_model=GenerationResponse)
    async def generate(
        body: GenerationRequest,
        request: Request,
        x_request_id: Annotated[str | None, Header()] = None,
    ) -> dict[str, object]:
        scheduler: Scheduler = request.app.state.scheduler
        request_id = x_request_id or str(uuid4())
        try:
            result = await scheduler.submit(request_id, body.prompt)
            return asdict(result)
        except QueueOverloadedError as exc:
            request.app.state.metrics.record_error("overloaded")
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except RequestExpiredError as exc:
            request.app.state.metrics.record_error("timeout")
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except Exception as exc:
            request.app.state.metrics.record_error("internal")
            LOGGER.exception("generation failed for request %s", request_id)
            raise HTTPException(status_code=503, detail="generation failed") from exc

    return app


app = create_app()
