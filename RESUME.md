# Resume bullet templates

Choose one and adapt the wording to your resume style. Every number below comes from the
checked-in Apple M3 real-model benchmark.

- Built a bounded dynamic-batching LLM inference gateway in Python, FastAPI, PyTorch, and
  Transformers; improved fixed-output throughput **3.28×** (3.09 → 10.15 requests/s) and cut
  p95 latency **67.4%** (3,023 → 984 ms) versus sequential inference at concurrency 8.

- Designed deadline- and size-triggered batching with async backpressure, cancellation-safe
  response correlation, and prompt-length buckets; reduced mixed-length padding overhead
  **73.2%** (123% → 33%) in a matched batch-size-4 experiment.

- Built a reproducible load/benchmark pipeline with raw per-request JSON, hardware and
  dependency metadata, repeated trials, Pareto-frontier plots, and **16 passing tests**
  covering dispatch, overload, timeouts, correlation, metrics, and live concurrent HTTP.
