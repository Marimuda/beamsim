# Development guide

## Environment setup

```bash
git clone https://github.com/jakupsv/beamsim.git
cd beamsim
python -m venv .venv && source .venv/bin/activate
make install        # editable install with [dev] extras
make hooks          # pre-commit + pre-push git hooks
```

The Makefile (`make help`) is the single source of truth for developer
commands. CI runs the same targets.

## Quality gate

```bash
make check     # ruff format-check + ruff lint + mypy + fast tests
```

The pre-push hook runs the same fast subset.

## Full test suite

```bash
make test      # all 185 tests, including @pytest.mark.slow
make cov       # full suite with coverage report
```

## Style and conventions

| Tool          | Configured in        | Run via                  |
| ------------- | -------------------- | ------------------------ |
| `ruff format` | `[tool.ruff.format]` | `make format`            |
| `ruff check`  | `[tool.ruff.lint]`   | `make lint`              |
| `mypy`        | `[tool.mypy]`        | `make type`              |
| `pytest`      | `[tool.pytest.ini_options]` | `make test`       |

- **Line length**: 100.
- **Quotes**: double.
- **Imports**: absolute (`from beamsim.x import y`); ruff isort handles ordering.
- **Typing**: `disallow_untyped_defs = true`. Public functions must be typed.
- **Logging**: `logger = logging.getLogger(__name__)` per module; never `print`.
  See [Logging strategy](#logging-strategy) below.

## Logging strategy

- Library modules call `logging.getLogger(__name__)` and emit at INFO for
  progress, WARNING for recoverable issues, and ERROR for unrecoverable ones.
- The CLI (`beamsim-run`) lets Hydra configure the root handler. Pass
  `--info` for verbose tracing or override `hydra.verbose=true`.
- No module configures the root logger at import time. Tests should not
  rely on log output.

## Determinism

The simulator uses common random numbers. Every algorithm receives a per-trial
`np.random.Generator` from `runner.run_experiment`. Do not:

- call the global `numpy.random` API,
- read the system clock for randomness,
- introduce non-deterministic data structures (e.g. `set` ordering for
  reduce-style aggregations).

If you must add stochastic behaviour, accept a `Generator` argument and
default it to `np.random.default_rng(seed)` only at the public boundary.

## Adding tests

- Unit / integration / regression tests live flat under `tests/`.
- File-touching tests use `tmp_path`.
- Slow Monte-Carlo tests (over many trials) use `@pytest.mark.slow`.
- Property-based tests use Hypothesis; `tests/test_property.py` is the model.
- The Hydra integration test (`TestHydraConfig`) is the template for new
  end-to-end smoke tests.

## Benchmarks

`pytest-benchmark` is installed; benchmark files live under `tests/` (or a
future `benchmarks/`). To run them:

```bash
pytest --benchmark-only
```

Benchmark numbers reported in commit messages or papers must specify the
machine and Python version — there is no benchmark-environment auto-capture
yet.

## Commit and PR conventions

- Imperative-mood commit subjects, ≤ 72 chars.
- Phase-prefixed commits (`Phase 4D: …`) are accepted for bulk milestone work.
- Update `CHANGELOG.md` under **Unreleased** for any user-visible change.
- One logical change per PR. CI must be green before merge.

See [`CONTRIBUTING.md`](https://github.com/jakupsv/beamsim/blob/main/CONTRIBUTING.md)
for the contributor flow.

## Releasing (maintainers)

1. Move **Unreleased** entries in `CHANGELOG.md` under a dated version heading.
2. Bump `version` in `pyproject.toml`, `__init__.py`, and `CITATION.cff`.
3. `make build` and verify `dist/` contents.
4. Tag: `git tag -a vX.Y.Z -m "Release X.Y.Z"` and push the tag.
5. Create a GitHub release pointing at the tag; the release triggers Zenodo
   archival and DOI minting (after the repo is configured on Zenodo).
