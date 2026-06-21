# Testing

Diffulex uses `pytest` for Python tests. The test suite includes lightweight
unit tests, engine behavior tests, server tests, model tests, and GPU-heavy
kernel or smoke tests.

## pytest

Run the full Python test suite from the repository root:

```bash
pytest
```

Run a focused test file while developing:

```bash
pytest test/python/engine/test_generation_outputs.py
```

Prefer focused commands while iterating. Full-suite runs are useful before
integration, but they can hide the signal from a single failing area.

## Test Layout

Common test areas include:

- `test/python/engine/` for scheduler, request, config, cache, and generation
  behavior;
- `test/python/server/` for HTTP server and frontend/backend coordination;
- `test/python/model/` for model-family behavior;
- `test/python/layer/` for reusable layers;
- `test/python/kernel/` for kernel correctness and performance checks;
- `test/python/moe/` for MoE dispatcher and routing behavior.

Use the nearest existing test as the template for new coverage.

## GPU Tests

Many engine and kernel tests require CUDA devices. Keep GPU tests small and make
their resource assumptions explicit through markers, filenames, or test docs.

When diagnosing a GPU-only failure:

1. reproduce with the smallest prompt or tensor shape;
2. reduce tensor and data parallelism to `1`;
3. disable eager/CUDA graph toggles only after correctness is established;
4. compare against a reference implementation when available.

## Adding Regression Coverage

For bug fixes, add a test that fails on the original behavior when practical.
If a full reproduction requires large models or private checkpoints, add the
smallest unit-level check that guards the broken contract.

For docs-only changes, a Sphinx build is the relevant verification command:

```bash
./.venv/bin/python -m sphinx -E -b html docs docs/_build/html
```
