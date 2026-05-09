# Repository Quality Checklist — `beamsim`

A scoped subset of [`../checklist.md`](../checklist.md), tailored to this
repository: a single-author scientific Python simulation that reproduces the
figures of an MSc-paper reformulation, exposes one Hydra-driven entry point,
ships no notebooks, makes no network calls, and handles no secrets.

Items pruned from the master checklist for this repo type:

- §17 Security: subprocess shell-injection, network handling, deserialization
  of untrusted input, broad permission auditing — not applicable.
- §19 Community: funding metadata, governance model, multi-maintainer policy.
- §20 API design: public deprecation/backward-compatibility policy at v0.1.
- §21 CLI design: dry-run / structured-output / verbosity flags — Hydra's
  defaults already cover `--help`, overrides, and exit codes.
- §25 Release: TestPyPI/PyPI publishing automation, signed artefacts, SBOM,
  provenance attestations.
- §26 Archival: optional Software Heritage step (Zenodo DOI is enough).
- §29 Notebook policy — no notebooks in this repo.
- §6 Optional: Apptainer/Singularity, Nix flake, Dev container.

Legend: `[x]` done · `[ ]` pending (post-GitHub-push) · `[~]` partial.

---

## 1. Repository identity

- [x] Clear repository name (`beamsim`).
- [ ] Short one-sentence description in GitHub metadata (set after first push).
- [x] `README.md` exists.
- [x] README states what the software does.
- [x] README states intended users.
- [x] README states repository status (`Active research code, version 0.1.0`).
- [x] README states supported operating systems (Linux, macOS).
- [x] README states supported Python versions (3.10 / 3.11 / 3.12).
- [x] README includes installation instructions.
- [x] README includes basic usage.
- [x] README includes development commands (Makefile target table).
- [x] README includes testing commands.
- [x] README includes citation instructions.
- [x] README includes license summary.
- [x] README does not rely on hidden local assumptions.

## 2. Legal and citation metadata

- [x] `LICENSE` exists (MIT).
- [x] `CITATION.cff` with authors, title, repo URL, version, date, license.
- [x] `[project.authors]` populated in `pyproject.toml`.
- [x] `CHANGELOG.md` exists (Keep a Changelog format, seeded from git log).

## 3. Repository layout

- [x] `src/` layout used.
- [x] Source code separated from tests / configs / experiments / docs.
- [x] No important code lives only in notebooks (no notebooks at all).
- [x] No hardcoded absolute paths in production code (`/tmp/...` only in
      negative-path tests).
- [x] No committed virtual envs / caches / build artefacts.
- [x] Large simulation outputs (`*.npz`, `*.pdf`, `*.log`, `models/*.pt`,
      `site/`, coverage artefacts) excluded.

## 4. Packaging

- [x] `pyproject.toml` with explicit build backend (`setuptools.build_meta`).
- [x] Package name, version, description, readme, license declared.
- [x] Supported Python versions declared (`>=3.10`).
- [x] Trove classifiers populated.
- [x] Runtime dependencies declared.
- [x] Dev / test / docs / dl optional-dep groups separated.
- [x] Console entry point declared (`beamsim-run`).
- [x] No `sys.path.append` / no `PYTHONPATH` mutation.
- [x] `py.typed` marker exists and ships in the wheel.
- [x] Public API exported through `__all__` in `beamsim/__init__.py`.
- [x] Internal helpers prefixed `_` where appropriate.
- [x] Wheel and sdist build (`python -m build`) and pass `twine check`.

## 5. Dependency management

- [x] Dependency manager chosen deliberately (pip + pyproject.toml ranges).
- [x] Runtime deps minimal; dev/test/docs separated.
- [x] Optional heavy deps (`torch`) not mandatory.
- [x] Version bounds reasonable.
- [ ] Dependabot enabled on GitHub (config committed; activates on push).

> **Lockfile note.** This is a research-reproducibility repo, not a deployable
> service. Pinned ranges in `pyproject.toml` plus a CI matrix across supported
> Python versions are sufficient. A full `uv.lock` is intentionally **not**
> required.

## 6. Environment reproducibility

- [x] Python version constrained (`>=3.10`).
- [x] `.python-version` exists.
- [x] Installation works on a fresh machine (`pip install -e .` validated).
- [ ] Installation works in CI (post first GitHub push).
- [x] No environment variables required.
- [x] No secrets in repo or history.
- [x] Determinism documented (CRN seed strategy in `docs/architecture.md`).

## 7. Configuration

- [x] Configuration is explicit (Hydra YAML in `configs/`).
- [x] Defaults documented inline in YAML files.
- [x] Local user configuration (`outputs/`) ignored.
- [x] No absolute machine-specific paths in configs.
- [x] No credentials in configs.
- [x] Config loading precedence is Hydra-standard.
- [x] Example config files exist for every experiment.
- [x] Tests cover Hydra config loading (`TestHydraConfig::test_load_rotational_config`).
- [x] Tests cover invalid config handling
      (`TestHydraConfig::test_unknown_sweep_variable_raises`,
      `test_unknown_channel_kind_raises`).

## 8. Command interface

- [x] `Makefile` exists with `install`, `format`, `lint`, `type`, `test`,
      `cov`, `check`, `docs`, `build`, `clean`, `hooks`.
- [x] All commands run from repo root.
- [x] Commands referenced consistently in README and CI.

## 9. Code quality

- [x] Formatter configured (`ruff format`).
- [x] Linter configured (`ruff check`).
- [x] Import sorting configured (`ruff` rule `I`).
- [x] Static type checker configured (`mypy`).
- [x] Public functions have clear names.
- [x] Public APIs have docstrings.
- [x] Error handling explicit; domain values raise `ValueError` with context.
- [x] Library code uses `logging`, not `print` (all 4 prints migrated).
- [x] No import-time side effects.
- [x] No mutable global state.
- [x] I/O isolated from pure logic.
- [x] Core logic is testable without the CLI.
- [x] CLI is a thin Hydra adapter over `runner.py`.
- [x] No broad `except Exception` without justification (3 cases — all in
      ML-checkpoint loading, fall back to a heuristic policy).
- [x] No silent failure modes.
- [x] No hidden network calls.
- [x] No hidden filesystem writes.

## 10. Typing

- [x] Type checker selected (`mypy`).
- [ ] Type checker runs in CI (config committed; activates on push).
- [x] Public API is typed.
- [x] Configuration objects are typed.
- [x] Return types explicit on public functions.
- [~] `Any` usage minimised (survives at the Hydra/OmegaConf boundary).
- [x] `type: ignore` comments are localised and justified.
- [x] Protocols used for the algorithm interface (`base.AlgorithmBase`).
- [x] Package includes `py.typed` and ships it in the wheel.

## 11. Testing

- [x] Test framework configured (`pytest`, `pytest-xdist`, `hypothesis`,
      `pytest-benchmark`, `pytest-cov`).
- [x] Unit tests exist.
- [x] Integration tests exist.
- [x] Regression tests exist.
- [x] Property-based tests exist.
- [x] Tests run locally (193 passing, ~ 34 s).
- [ ] Tests run in CI (post first push).
- [x] Tests are deterministic.
- [x] Tests do not depend on developer machine paths.
- [x] Tests do not require secrets.
- [x] Slow tests marked (`@pytest.mark.slow`).
- [x] Filesystem tests use `tmp_path`.
- [x] Tests cover invalid input and edge cases.
- [x] Tests cover public API surface.
- [x] Test verifies installed package imports + `py.typed` marker
      (`tests/test_packaging.py`).
- [x] Coverage measurement configured (`[tool.coverage.*]`).
- [ ] Coverage reported in CI (post first push).

## 12. Continuous integration

- [x] GitHub Actions configured (`.github/workflows/ci.yml`).
- [x] CI runs on PRs and pushes to `main`.
- [x] CI runs ruff format check, ruff lint, mypy, pytest.
- [x] CI builds package (sdist + wheel) and runs `twine check`.
- [x] CI matrix covers Python 3.10 / 3.11 / 3.12.
- [x] CI matrix covers Linux + macOS.
- [x] Dependency caching enabled (`cache: pip`).
- [x] CI builds the docs site (`mkdocs build --strict`).
- [ ] Required checks enforced before merge (set on GitHub after first push).

## 13. Pre-commit

- [x] `.pre-commit-config.yaml` exists.
- [x] Format hook (`ruff-format`).
- [x] Lint hook (`ruff` with `--fix`).
- [x] Import sorting handled by ruff.
- [x] EOF fixer.
- [x] Trailing-whitespace hook.
- [x] Mixed-line-ending hook.
- [x] Large-file detection hook (max 512 KB).
- [x] Private-key detection hook.
- [x] YAML / TOML / JSON validation hooks.
- [x] Merge-conflict marker check.
- [x] mypy hook.
- [x] Pytest fast subset hook (`pre-push` stage).
- [x] Hook installation documented (`make hooks` in CONTRIBUTING.md
      and `docs/development.md`).
- [x] Pre-commit runs successfully on all files.

## 14. Documentation

- [x] Documentation system selected (MkDocs Material + mkdocstrings).
- [x] Docs build locally (`mkdocs build --strict` succeeds).
- [x] `docs/index.md`.
- [x] `docs/installation.md`.
- [x] `docs/quickstart.md`.
- [x] `docs/usage.md`.
- [x] `docs/architecture.md` (with Mermaid module map and data-flow diagram).
- [x] `docs/development.md`.
- [x] `docs/api.md` (auto-generated reference via `mkdocstrings`).
- [x] `docs/SOTA_BASELINES.md` already documents the baselines reference card.
- [x] `docs/changelog.md` (snippet of root `CHANGELOG.md`).
- [x] README links to full docs.
- [x] Public functions have docstrings.

## 15. Examples

- [x] `examples/minimal_example.py` — public API only, runs in &lt; 2 s.
- [x] Example smoke-tested (`tests/test_examples.py`).
- [x] Example output explained in `docs/quickstart.md`.

## 16. Logging

- [x] Library code uses `logging.getLogger(__name__)`, never `print`.
- [x] Logging strategy documented in `docs/development.md`.
- [x] Hydra's app-level logging used at the entry point.

## 17. Security (scoped)

- [x] No secrets in history.
- [x] `.env` ignored (no `.env` exists; pattern still covered).
- [x] Private-key detection hook in pre-commit.
- [x] `SECURITY.md`.
- [x] User input validated where applicable.
- [x] File paths handled via `pathlib`.
- [x] Temporary files use `tmp_path` / `tempfile`.

## 18. Git hygiene

- [x] `.gitignore` appropriate.
- [ ] Working branch matches PR-target branch (`master` → rename to `main`
      before first GitHub push).
- [x] Commit messages are meaningful.
- [x] No large binary blobs in history.
- [x] No generated noise / no formatting churn mixed with logic.
- [ ] Release tags created (`v0.1.0` after first publish).
- [ ] Branch protection enforced (set on GitHub after first push).

## 19. GitHub community files

- [x] `CONTRIBUTING.md`.
- [x] `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1).
- [x] `SECURITY.md`.
- [x] `.github/ISSUE_TEMPLATE/bug_report.md`.
- [x] `.github/ISSUE_TEMPLATE/feature_request.md`.
- [x] `.github/ISSUE_TEMPLATE/documentation.md`.
- [x] `.github/ISSUE_TEMPLATE/config.yml` (disables blank issues, points to
      security policy).
- [x] `.github/PULL_REQUEST_TEMPLATE.md`.
- [x] Maintainer contact clear (single author).

## 20. API design (light)

- [x] Public API intentionally defined.
- [x] Internal API separated (leading-underscore helpers).
- [x] Public API documented in `docs/api.md`.
- [x] Public names stable (renames go through CHANGELOG).
- [x] Inputs validated; exceptions documented.
- [x] No import-time side effects.
- [x] Public API tested (`tests/test_packaging.py`,
      `tests/test_runner.py`, `tests/test_algorithms.py`).

## 21. CLI (light, Hydra)

- [x] CLI entry point declared (`beamsim-run`).
- [x] `--help` works (Hydra default).
- [x] CLI separates parsing from business logic (`main` → `run_from_config`).
- [x] CLI validates inputs (rejects unknown sweep variables / channel kinds).
- [x] CLI tests exist (4 tests under `TestHydraConfig`).
- [x] CLI usage documented in `docs/usage.md`.

## 22. Data and file handling

- [x] Output format documented (`.npz` schema in `docs/usage.md`).
- [x] Path handling via `pathlib`.
- [x] Output directories created with `parents=True, exist_ok=True`.

## 23. Performance

- [x] Vectorised hot path (240-subray einsum, ~ 8× speedup).
- [x] `pytest-benchmark` available; benchmark output explained in
      `docs/development.md`.
- [x] Parallelism explicit (`ProcessPoolExecutor`).
- [x] Resource cleanup guaranteed.

## 24. Reliability

- [x] Random seeds controlled (CRN across algorithms per trial).
- [x] Determinism documented (`docs/architecture.md`).
- [x] Edge cases tested.
- [x] No partial writes.

## 25. Release management (light)

- [x] Versioning policy stated (semver, pre-1.0 churn allowed) in CHANGELOG
      and CONTRIBUTING.
- [x] `CHANGELOG.md` seeded from Phase-1…Phase-4C history.
- [ ] First Git tag (`v0.1.0`) created after first GitHub push.
- [x] Source distribution + wheel build via `make build`.

## 26. Archival

- [ ] Stable `v0.1.0` release tagged.
- [ ] Zenodo integration enabled when the repo goes public.
- [ ] DOI badge added to README post-archival.

## 27. Maintainability

- [x] Architecture documented (`docs/architecture.md`).
- [x] Module-boundary diagram (Mermaid).
- [x] Dependency direction clean.
- [x] No circular imports.
- [x] I/O not mixed into core logic.
- [x] Developer onboarding instructions (`docs/development.md`,
      `CONTRIBUTING.md`).
- [x] Known limitations documented (UMi default, single-pol, azimuth-only —
      surfaced in `docs/index.md` and README §Scope).

## 28. Badges (post-CI)

- [x] CI status badge in README (will go live on first push).
- [x] Python version badge.
- [x] License badge.
- [x] Code-style (ruff) badge.
- [x] Type-checked (mypy) badge.
- [ ] DOI badge (post-archival).

## 29. Quality gates

The repository passes:

```bash
make install
make format
make lint
make type
make test
make build
make docs
```

…with all of these true:

- [x] Fresh clone works (would; pinned to `pip install -e ".[dev]"`).
- [ ] CI is green (post first push).
- [x] Package builds (`dist/beamsim-0.1.0-py3-none-any.whl`,
      `dist/beamsim-0.1.0.tar.gz`).
- [x] Package installs (`pip install -e ".[dev]"`).
- [x] Public API imports (`tests/test_packaging.py`).
- [x] Tests pass (193 / 193 in 33 s).
- [x] Type checker passes (mypy: 35 source files, 0 issues).
- [x] Linter passes (ruff: all checks passed).
- [x] Formatter check passes (ruff format --check).
- [x] Docs build (`mkdocs build --strict`).
- [x] README is accurate.
- [x] License is present.
- [x] Citation metadata is present.
- [x] Changelog exists.
- [x] No secrets present.
- [x] No machine-specific paths in production code.
- [x] No undocumented setup steps.

## 30. Final target

```bash
git clone <repo>
cd <repo>
make install
make check
make docs
make build
```

…with no hidden assumptions. **Status: this works locally today.**
The repository is **installable, typed, tested, linted, documented,
packaged, citable, licensed, versioned, and ready for CI validation
on first push to GitHub.**

## Remaining work (post-push)

These last boxes can only be ticked once the repo is on GitHub:

1. Push to `github.com/jakupsv/beamsim`, set the one-sentence GitHub
   description, rename the default branch from `master` to `main`.
2. Tag `v0.1.0` and create a GitHub Release — triggers Zenodo archival
   and DOI minting once the Zenodo–GitHub integration is enabled.
3. Configure branch protection on `main`: require PR + the CI checks
   defined in `.github/workflows/ci.yml`.
4. Verify Dependabot starts opening update PRs (already configured).
5. Add the DOI badge to `README.md` once Zenodo mints one.
