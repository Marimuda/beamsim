# Contributing to beamsim

Thanks for your interest. This is a research-reproduction repository for an
MSc-paper reformulation, so the contribution surface is intentionally small,
but external pull requests are welcome.

## Quick start

```bash
git clone https://github.com/jakupsv/beamsim.git
cd beamsim
python -m venv .venv && source .venv/bin/activate
make install      # installs the package + dev extras (test, docs, lint, type)
make hooks        # installs pre-commit and pre-push git hooks
make check        # format + lint + type + fast tests
```

`make help` lists every target.

## Development workflow

1. Open or claim an issue before non-trivial work.
2. Branch from `main`. Keep one logical change per PR.
3. Run `make check` locally (the pre-push hook does the same).
4. Add tests for any behavioural change; preserve existing seeds and CRN
   pairing in the simulation runners.
5. Update `CHANGELOG.md` under the **Unreleased** section.
6. Write commit messages in the existing style (imperative mood, short
   subject, optional body explaining the *why*).

## Coding standards

- **Formatting**: `ruff format` (line length 100, double quotes).
- **Linting**: `ruff check` with the rule set defined in `pyproject.toml`.
- **Typing**: `mypy` with `disallow_untyped_defs`. Public functions must be
  typed; `Any` is acceptable only at the Hydra/OmegaConf boundary.
- **Logging**: library code uses `logging.getLogger(__name__)`. Never `print`.
- **Determinism**: every algorithm receives an `np.random.Generator` from
  the runner; never call the global RNG.
- **No hidden I/O**: only `runner.save_experiment` writes to disk; only
  `runner.run_experiment` may parallelise via `ProcessPoolExecutor`.
- **Imports**: absolute imports inside the package (`from beamsim.x import y`).

## Tests

```bash
make test         # full suite (slow tests included)
make test-fast    # unit + property + smoke (no slow Monte-Carlo runs)
make cov          # full suite with coverage report
```

Test categories live under `tests/`:

- `test_*` — unit, integration, regression, property-based (Hypothesis).
- Slow Monte-Carlo tests are marked `@pytest.mark.slow`.
- Filesystem tests use `tmp_path`; never write to the working directory.

## Pull requests

- Reference the issue your PR addresses.
- Keep PRs focused; split unrelated changes.
- CI must be green before merge: `format-check`, `lint-check`, `type`, `test`,
  `build` across the supported Python matrix.
- Update `docs/` and `CHANGELOG.md` for user-visible changes.
- For experimental algorithms or scenarios, add at least one regression
  test that locks in the expected behaviour at a fixed seed.

## Reporting bugs and security issues

- **Bugs** — use the issue templates under `.github/ISSUE_TEMPLATE/`.
- **Security** — see [`SECURITY.md`](SECURITY.md).

## License

By contributing you agree your contribution is released under the
[MIT license](LICENSE).
