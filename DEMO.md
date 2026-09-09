# 2–3 minute demo

## 0:00–0:30 — Frame the systems problem

Open the README architecture diagram. Explain that concurrent HTTP is not concurrent model
execution: V1 serializes every request, while V2 waits at most a small deadline to form a
bounded batch. Point out explicit 429 overload and 504 timeout behavior.

## 0:30–1:15 — Run the service

```bash
make serve
```

In a second terminal:

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
curl -s http://127.0.0.1:8000/v1/generate \
  -H 'content-type: application/json' -H 'x-request-id: live-demo' \
  -d '{"prompt":"Why does dynamic batching trade latency for throughput?"}' \
  | python -m json.tool
```

Call out `request_id`, `queue_ms`, `inference_ms`, `batch_size`, and prompt padding fields.
Then run `.venv/bin/llm-loadgen --concurrency 8 --requests 24` and show `/metrics`.

## 1:15–2:15 — Show measured trade-offs

Open `artifacts/plots/p95-vs-throughput.png`. At concurrency 8, V2 unbucketed reached 10.15
requests/s versus 3.09 for V1 and reduced p95 from 3,023 ms to 984 ms. All requests generated
exactly eight tokens.

Open `artifacts/plots/padding-overhead.png`. Bucketing cut matched padding overhead from 123%
to 33%, but fragmentation lowered throughput. Explain that the best scheduler changes with
arrival rate and prompt-length distribution.

## 2:15–2:45 — Prove reproducibility

Show `artifacts/raw/benchmark-m3-real.json` for per-request records and machine/dependency
metadata. Finish with `make verify`: 15 unit/API tests plus a live-server concurrent test.
