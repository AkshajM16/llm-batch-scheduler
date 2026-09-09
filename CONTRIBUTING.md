# Contributing

Contributions are welcome through focused issues and pull requests.

## Development setup

```bash
uv sync --extra dev
make verify
```

Use `make format` before opening a pull request. Tests that exercise scheduling should use the
deterministic fake backend; performance claims must come from the Transformers backend and
include raw benchmark metadata.

## Pull requests

- Keep the sequential and dynamic modes compatible with the same HTTP API.
- Add regression tests for scheduler, cancellation, correlation, or metric changes.
- Do not commit downloaded model weights, virtual environments, credentials, or caches.
- Label fake-backend measurements as scheduler-only; never mix them with model-inference results.
- Update the README when configuration, benchmark methodology, or measured claims change.

By contributing, you agree that your work is licensed under the repository's MIT License.
