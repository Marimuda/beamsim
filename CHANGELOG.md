# Changelog

All notable changes to `beamsim` are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The
public surface is still pre-1.0 — minor versions may break API.

## [Unreleased]

### Added

- Repository hygiene pass: `LICENSE` (MIT), `CITATION.cff`, `CHANGELOG.md`,
  `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`.
- `Makefile` with `install / format / lint / type / test / cov / check /
  docs / build / clean / hooks` targets.
- `py.typed` marker; curated public surface in `beamsim/__init__.py`.
- Hardened pre-commit (EOF / trailing-whitespace / large-file / private-key /
  YAML-TOML-JSON validation).
- GitHub Actions CI matrix (Python 3.10/3.11/3.12 on Linux + macOS).
- Dependabot, issue templates, PR template, MkDocs documentation scaffold,
  `examples/minimal_example.py` smoke test, coverage configuration.

### Changed

- All `print` calls in library code replaced with `logging.getLogger(__name__)`.
- `pyproject.toml`: populated authors/license/classifiers/URLs; split
  optional dependencies into `[test]`, `[docs]`, `[dl]`, and `[dev]`.

## [0.1.0] — 2026-05-09 *(unreleased on GitHub)*

The full Phase-1…Phase-4C reproduction of the predecessor MSc evaluation,
extended with SOTA baselines for the journal-paper reformulation.

### Phase 4C — LSTM predictor, 3GPP TR 38.843 metrics, baselines reference card

- LSTM-based beam predictor (`algorithms/dl_lstm_predictor.py`).
- Beam-management metrics aligned with 3GPP TR 38.843.
- `docs/SOTA_BASELINES.md` reference card.

### Phase 4B — Additional SOTA baselines

- MAMBA neighbourhood-explore policy.
- EKF tracker, position-aware MAB, BAI pure-exploration baselines.

### Phase 4A — Fidelity audit fixes

- Bug fixes uncovered by audit against the SOTA reference implementations.

### Phase 3 — SOTA baselines

- Compressive-sensing OMP, hierarchical beam-management (HBM), Thompson
  sampling, UCB1, and DL predictor baselines.

### Phase 2 — Tooling

- Hydra config layer (`configs/scenario/`, `configs/sweep/`, `configs/algo/`).
- SciPy-based bootstrap CI in `metrics.py`.
- pre-commit (ruff, ruff-format, mypy, pytest-fast).

### Phase 1 — Replicate report Figs 6.5–6.9 + calibration

- Faithful predecessor fidelity (NNS LIFO, Tabu `s=20`, AngPred gradient-sum,
  MCMD binary `C_nns`).
- Channel: full TR 38.901 LSP / sub-ray procedure with Laplacian rays,
  LOS-LSPs always, geometric powers, Model A blockage.
- 240-subray sum vectorised into a single `einsum` (~8× speedup).
- Common-random-numbers paired Monte Carlo runner with parallel execution.
- Bootstrap-CI plotting for journal-style figures.
- Match in sin-space to handle ULA front/back-lobe ambiguity.

### Initial skeleton

- `pyproject.toml`, `src/` layout, README, `.gitignore`.
- Cosine-spaced ULA codebook, mobility tracks (rotation / straight line).
- Simplified TR 38.901 GSCM, BPLM bookkeeping, six initial MBP policies.

[Unreleased]: https://github.com/jakupsv/beamsim/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jakupsv/beamsim/releases/tag/v0.1.0
