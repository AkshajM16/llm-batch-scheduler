from __future__ import annotations

from fastapi.testclient import TestClient

from llm_scheduler.app import create_app
from llm_scheduler.config import Settings


def test_generate_and_health_api() -> None:
    settings = Settings(
        backend="fake",
        scheduler_mode="dynamic",
        max_batch_size=1,
        max_wait_ms=0,
    )
    with TestClient(create_app(settings)) as client:
        health = client.get("/health")
        response = client.post(
            "/v1/generate",
            json={"prompt": "hello scheduler"},
            headers={"x-request-id": "known-id"},
        )
        metrics = client.get("/metrics")

    assert health.status_code == 200
    assert health.json()["mode"] == "dynamic"
    assert response.status_code == 200
    assert response.json()["request_id"] == "known-id"
    assert response.json()["completed_time_unix_s"] >= response.json()["arrival_time_unix_s"]
    assert response.json()["generated_text"] == "response:hello scheduler"
    assert metrics.json()["completed"] == 1
