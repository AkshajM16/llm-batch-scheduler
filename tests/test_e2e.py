from __future__ import annotations

import asyncio
import socket
import sys

import httpx
import pytest


def available_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_live_server_handles_correlated_concurrent_requests() -> None:
    port = available_port()
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "llm_scheduler.cli",
        "--backend",
        "fake",
        "--mode",
        "dynamic",
        "--max-batch-size",
        "4",
        "--max-wait-ms",
        "5",
        "--port",
        str(port),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        async with httpx.AsyncClient(base_url=url, timeout=5) as client:
            for _ in range(80):
                if process.returncode is not None:
                    output, _ = await process.communicate()
                    pytest.fail(f"server exited early: {output.decode(errors='replace')}")
                try:
                    if (await client.get("/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.05)
            else:
                pytest.fail("server never became ready")

            async def send(index: int) -> dict[str, object]:
                request_id = f"e2e-{index}"
                response = await client.post(
                    "/v1/generate",
                    json={"prompt": f"prompt {index}"},
                    headers={"x-request-id": request_id},
                )
                assert response.status_code == 200
                return response.json()

            results = await asyncio.gather(*(send(index) for index in range(12)))

        assert [result["request_id"] for result in results] == [f"e2e-{i}" for i in range(12)]
        assert [result["generated_text"] for result in results] == [
            f"response:prompt {i}" for i in range(12)
        ]
        assert any(result["batch_size"] > 1 for result in results)
    finally:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except TimeoutError:
            process.kill()
            await asyncio.wait_for(process.wait(), timeout=5)
