.PHONY: setup format lint test test-unit test-e2e serve serve-v1 load benchmark benchmark-full plots verify

setup:
	uv sync --extra dev

format:
	.venv/bin/ruff format .
	.venv/bin/ruff check --fix .

lint:
	.venv/bin/ruff format --check .
	.venv/bin/ruff check .

test: test-unit test-e2e

test-unit:
	.venv/bin/pytest -m "not e2e"

test-e2e:
	.venv/bin/pytest -m e2e

serve:
	.venv/bin/llm-scheduler

serve-v1:
	.venv/bin/llm-scheduler --mode sequential

load:
	.venv/bin/llm-loadgen --concurrency 8 --requests 40

benchmark:
	.venv/bin/llm-benchmark --preset representative

benchmark-full:
	.venv/bin/llm-benchmark --preset full

plots:
	.venv/bin/llm-plots

verify: lint test
