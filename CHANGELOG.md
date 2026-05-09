# Changelog

All notable changes to `beamsim` are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The
public surface is still pre-1.0 — minor versions may break API.

## [Unreleased]

## [0.3.0] — 2026-05-09

Minor release: closes the largest gaps from the MATLAB-parity audit
(`docs/MATLAB_PARITY.md`) by porting four standalone algorithms and
adding Uniform Planar Array support — the original MATLAB simulator
exposed all of these but the Python re-implementation did not.

### Added

- `algorithms/agemx.py` (`AgeMx`) — standalone least-recently-measured
  beam-pair scan, formerly only available as MCMD's age criterion.
- `algorithms/random_search.py` (`RandomSearch`) — randperm-then-mark
  scan baseline, RNG seeded from `context["trial_seed"]`.
- `algorithms/nns_tabu.py` (`NNSTabu`) — NNS-with-tabu (Ascent_Tabu),
  the algorithm MCMD's slot-7 weight (`W_High[7] = 0.8742` in the
  MATLAB simulator) actually points at; distinct from plain NNS by
  using the GLOBAL argmax of `|Y_obs|` for relocation when the
  five-cell list is exhausted.
- `algorithms/ci_mbs.py` (`ContextInformationMBS`) — multi-BS CI,
  picks the closest BS by L2 distance then runs the standard
  sin-space CI match in that BS's frame.
- `codebook.PlanarCodebook` + `codebook.planar_steering_vector` —
  Uniform Planar Array codebook (n_x × n_y elements in xy-plane at
  half-wavelength spacing, azimuth-only steering uniform over
  `[0, 2π)`). Mirrors MATLAB `placodebook.m`.
- `channel.PlanarFreeSpaceLosChannel` — companion LOS channel for
  end-to-end UPA experiments; works with `BPLMState` via the same
  `w.conj() @ H @ f` contract as the ULA channel.
- `make_default_planar_ue_codebook` (2×2 UPA, 6 beams) and
  `make_default_planar_bs_codebook` (4×4 UPA, 12 beams) factories.
- 28 new tests (`tests/test_matlab_ports.py`,
  `tests/test_planar_codebook.py`).

### Changed

- `BPLMState.ue_codebook` and `BPLMState.bs_codebook` type annotations
  widened to `Codebook | PlanarCodebook` so mypy recognises the
  polymorphism (no runtime change — both types already exposed the
  required `n_beams` and `codeword(k)` interface).
- `beamsim.__all__` extended with `PlanarCodebook` and
  `PlanarFreeSpaceLosChannel`; `tests/test_packaging.py`'s
  expected-public-API set updated to match.
- `docs/MATLAB_PARITY.md` updated to mark the resolved entries; the
  full TR 38.901 cluster channel (`ChannelRealisation`) still uses
  ULA steering internally and is tracked under `docs/ROADMAP.md`
  for a future UPA extension.

### Note on library choice

The planar steering implementation is hand-rolled numpy. The mature
candidates (NVIDIA Sionna, DeepMIMO) are at least an order of
magnitude heavier than the codebook layer alone needs, and DeepMIMO
is the natural integration only at the channel layer when ray-traced
channels are added. Both are kept on the roadmap rather than imported
as required dependencies.

### Test count

216 → 248 passing.

## [0.2.1] — 2026-05-09

Patch release: codebook-oracle SNR is now computed for single-BS experiments,
not just multi-BS, so the regret diagnostics shipped in 0.2.0 apply uniformly
across all `runner.py` paths.

### Added

- `runner._run_trial` now computes per-step codebook-oracle SNR for
  **single-BS** experiments by collecting the `(n_steps, N_UE, N_BS)` channel
  matrix stack and delegating to `metrics.oracle_snr_db` once per trial
  (vectorised, one einsum call).
- `runner.run_experiment` allocates `snr_oracle_agg` unconditionally (was
  gated on `multi_bs`); the result dict now carries `snr_oracle` with shape
  `(n_trials, n_steps)` for both experiment topologies.
- `TrialResult.snr_oracle` is now populated for both single-BS and multi-BS
  runs; the docstring comment updated accordingly.
- Three new tests in `tests/test_runner.py`:
  - `test_single_bs_oracle_populated` — shape and finite-value guard.
  - `test_oracle_dominates_achieved_single_bs` — Perfect algorithm oracle ≥
    achieved up to floating-point floor (1 × 10⁻³ dB).
  - `test_oracle_matches_metrics_function` — cross-checks the runner value
    against an independent call to `metrics.oracle_snr_db` on the same
    channel sequence.

### Changed

- `runner.py`: import `beamsim.metrics.oracle_snr_db` directly (no circular
  dependency — `metrics.py` has no beamsim internal imports by design).
- Multi-BS inline oracle computation unchanged; pre-conjugated `_W` matrix
  now derived from the shared `_UE_W` to avoid redundant transposes.

## [0.2.0] — 2026-05-09

Public-API addition release: four codebook-oracle / regret / outage /
beam-switch metrics land in `beamsim.metrics`, and the repository
adopts the modern beam-management evaluation vocabulary in its docs
without changing the simulator's scientific object.

### Added

- `metrics.oracle_snr_db(channel_matrices, ue_weights, bs_weights,
  noise_amplitude, tx_amp)` — vectorised exhaustive scan over the
  finite UE × BS codebook for the same channel realisation. Documented
  as the *codebook* oracle; explicitly not Shannon capacity and not a
  deployable policy.
- `metrics.snr_regret_db(achieved, oracle)` — additive dB gap with
  sign convention `oracle − achieved` (lower is better, zero is
  optimal under the simulated codebook). Documented as a dB-domain
  gap, not a linear-power regret functional.
- `metrics.outage_probability(snr_db, threshold_db)` — pooled scalar
  `Pr(SNR_dB < threshold_db)` with strict inequality and explicit NaN
  propagation.
- `metrics.beam_switch_rate(obp_history)` — fraction of consecutive
  step pairs at which (k, l) changes; explicit `n_steps − 1`
  denominator; returns `0.0` for traces of length < 2.
- `docs/SOTA_BASELINES.md`: new "Algorithms by measurement budget"
  taxonomy table classifying every shipped algorithm by probe budget
  per decision (oracle-like → coarse-to-fine → compressive →
  local/temporal → uncertainty-aware adaptive → predictive → genie).
- `docs/related_work.md`: new "Initial access vs. tracking vs.
  recovery" subsection that explicitly marks 3GPP-style beam-failure
  recovery (BFR) as out of scope.
- `docs/ROADMAP.md`: deferred-work register covering per-algorithm
  measurement budgets and reacquisition time after blockage, each
  with the architectural change required to land it.
- README "Methodological commitments" section elevating
  common-random-numbers paired evaluation, codebook-oracle regret,
  and the overhead/switching/outage metric set as first-class
  contributions.
- 25 new tests covering shape handling, NaN behaviour, threshold
  boundary, switch-rate denominators, oracle-dominates-specific-probe,
  and BPLM-convention agreement on a single-pair codebook.

### Changed

- "Oracle" terminology disambiguated across `runner.py`,
  `algorithms/perfect.py`, and `docs/SOTA_BASELINES.md` so every
  reference now reads as the *codebook* oracle (not Shannon capacity).
- `metrics.snr_regret_db` docstring spells out the dB-vs-linear-power
  semantics with the formula
  `regret_dB(t) = 10·log10(SNR_oracle(t) / SNR_achieved(t))`.
- `metrics.outage_probability` docstring leads with the defining
  formula and the strict-inequality justification (deterministic
  tests can equal the threshold by construction).
- `metrics.beam_switch_rate` docstring leads with the explicit
  `n_steps − 1` denominator and reasons out the `0.0` return on
  single-step traces.

## [0.1.0] — 2026-05-09 *(unreleased on GitHub; superseded by 0.2.0 the same day)*

The 0.1.0 line covered the scholarly-hygiene pass: `LICENSE`,
`CITATION.cff`, `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
`SECURITY.md`, `Makefile`, `py.typed`, hardened pre-commit, GitHub
Actions CI matrix (Py 3.10/3.11/3.12 × Linux/macOS), Dependabot, issue
and PR templates, MkDocs documentation scaffold,
`examples/minimal_example.py`, coverage configuration. Library code's
remaining `print` calls were migrated to `logging`. `pyproject.toml`
gained authors/license/classifiers/URLs and split optional dependencies
into `[test]`, `[docs]`, `[dl]`, and `[dev]`. No GitHub release was cut
for 0.1.0 in isolation.

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

[Unreleased]: https://github.com/Marimuda/beamsim/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Marimuda/beamsim/releases/tag/v0.3.0
[0.2.1]: https://github.com/Marimuda/beamsim/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Marimuda/beamsim/releases/tag/v0.2.0
[0.1.0]: https://github.com/Marimuda/beamsim/releases/tag/v0.1.0
