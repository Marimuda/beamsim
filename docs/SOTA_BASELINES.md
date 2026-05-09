# SOTA baselines reference card

This document is the source of truth for what each algorithm in
`src/beamsim/algorithms/` is, what it cites, what it is *not* a faithful
reproduction of, and what behaviour to expect.  It is intentionally honest
about the gaps between our implementations and the literature so that
results from this simulator can be reported without overclaiming.

The simulator's interface contract for every algorithm:

```python
algo.reset(state: BPLMState, context: dict) -> None
algo.select_next_mbp(state, m, context) -> tuple[int, int]
```

`context` provides `ue_pose_at(m)`, `bs_xy`, `bs_yaw`, optional
`true_H` (for oracles), and `trial_seed` (used to seed any internal RNG
so multi-trial Monte Carlo is reproducible).

---

## Roster

| Algorithm | File | Citation |
|---|---|---|
| `Exhaustive` | `exhaustive.py` | Row-major sweep — reference upper bound on overhead. |
| `NNS` | `nns.py` | Predecessor MSc thesis (steepest-ascent 4-connected). |
| `NNSBSSequential` | `nns_bs_sequential.py` | Same NNS but per-BS round-robin (Fig 6.7). |
| `Tabu` | `tabu.py` | Glover (1989) Tabu search. |
| `AngularPrediction` | `angular_prediction.py` | Predecessor thesis Algorithm 3 (gradient-sum). |
| `ContextInformation` | `ci.py` | Predecessor thesis context-aided beam selection. |
| `MCMD` | `mcmd.py` | The thesis's main contribution. |
| `Perfect` | `perfect.py` | Reads `context["true_H"]`; oracle upper bound. |
| `UCB1` | `ucb1.py` | Auer, Cesa-Bianchi, Fischer 2002 — **stationary, sanity-check only**. |
| `ThompsonGaussian` | `thompson.py` | Chapelle & Li 2011 — **stationary, sanity-check only**. |
| `HBM` | `hbm.py` | Giordani et al. COMST 2019 (3GPP NR P1/P2). |
| `OMPCompressive` | `omp_compressive.py` | Alkhateeb et al. JSAC 2014 §III-A. |
| `DLPredictor` | `dl_predictor.py` | Kim et al. JSAC 2023 — simpler MLP variant. |
| `MAMBA` | `mamba.py` | Aykin et al. INFOCOM 2020 / Krunz et al. TMC 2024. |
| `EKFTracker` | `ekf_tracker.py` | Jayaprakasam et al. Commun. Lett. 2017 / Burghal arXiv:1911.01638. |
| `PositionMAB` | `position_mab.py` | Va et al. IEEE Access 2019. |
| `BAIPureExploration` | `bai.py` | Chiu et al. TWC 2022 / Even-Dar et al. COLT 2002. |
| `DLLSTMPredictor` | `dl_lstm_predictor.py` | Kim et al. JSAC 2023 (LSTM variant). |

---

## Algorithms by measurement budget

A more useful classification than "exhaustive vs. heuristic vs. ML" is
**how many beam pairs the algorithm probes per decision** and **what it
relies on to keep that count down**.  The lower the budget, the more
the algorithm depends on temporal continuity, side information, or
learned priors — and the worse it tends to fail when those assumptions
break.

| Tier | Budget per decision | Reliance | Algorithms in this repo |
|---|---|---|---|
| **High — oracle-like** | $K\\cdot L$ (full sweep every step) | None: a brute-force baseline. | `Exhaustive` |
| **Reduced — coarse-to-fine** | $O(\\log K + \\log L)$ when the hierarchy holds; falls back to $K\\cdot L$ when it does not. | Codebook hierarchy validity. | `HBM` |
| **Compressive** | $O(\\log K \\cdot \\log L)$ random or structured probes. | Channel sparsity in the angle domain. | `OMPCompressive` |
| **Low — local / temporal** | One probe + a few neighbours; reuses the previous OBP between probes. | Mobility coherence between steps. | `NNS`, `NNSBSSequential`, `Tabu`, `AngularPrediction` |
| **Adaptive — uncertainty-aware** | One probe per step, but where it probes is controlled by belief state, regret estimates, or freshness. | Stationarity assumptions of the policy class. | `UCB1`, `ThompsonGaussian`, `BAIPureExploration`, `MAMBA`, `MCMD`, `EKFTracker`, `PositionMAB` |
| **Predictive — context-aided** | One probe (the predicted beam), no exploration unless prediction confidence collapses. | Trained model + matching deployment statistics. | `DLPredictor`, `DLLSTMPredictor`, `ContextInformation` |
| **Genie** | Zero probes — reads the true channel. | None physical: oracle for diagnosis only. | `Perfect` |

Practical reading order when comparing on a new scenario: start with
`Exhaustive` and `Perfect` to bracket what's achievable; then run one
representative from each lower tier; then probe the failure modes
(mobility regime, blockage rate, SNR) that should hurt each tier.

> Note that `beamsim` currently issues **one BPLM measurement per step**
> for every algorithm — the `Algorithm` interface does not yet let
> policies declare a per-step probe budget greater than one. The tiers
> above describe the *target* measurement budget each policy is meant
> to operate under in the literature; the simulator collapses that to
> a one-step-one-probe schedule for like-for-like comparison. Native
> per-algorithm probe budgets are tracked under
> [`ROADMAP.md`](ROADMAP.md).

---

## Per-algorithm notes

### `UCB1` — stationary multi-armed bandit (Auer 2002)

**What it is.** Standard UCB1 with bonus `sqrt(2 ln t / N)` over the
flattened K×L beam-pair index space.

**Caveats.**
- *Stationary-arm assumption.*  Regret bound does not hold under UE
  mobility.  Compare against `MAMBA` for a non-stationary variant.
- *Cold-start cost.*  K·L = 256 forced pulls before UCB takes over.
  At trial lengths below ~300 steps UCB1 essentially never leaves
  cold-start; this is a property of the algorithm, not a bug.
- *Reward bound.*  The exploration constant 2 in the bonus is derived
  under [0, 1]-bounded rewards (Hoeffding).  We use raw |y(k,l)|; this
  is dimensionally inconsistent with the bonus but rescaling broke
  ranking on a synthetic stationary test.  Treat the regret guarantee
  as informal.
- *Not contextual.*  Hashemi 2018 / Va 2019 (previously cited) are
  contextual bandits — see `PositionMAB` for the contextual variant.

**Use as.** Lower-bound sanity check.  Drop from primary figures.

---

### `ThompsonGaussian` — stationary Bayesian bandit

**What it is.** Conjugate Gaussian-Gaussian Thompson sampling with a
Welford-online estimate of the cross-arm reward standard deviation
(`sigma`), floored to `noise_amplitude` during cold-start.

**Caveats.**
- *Stationary-arm assumption.*  Same as UCB1; under mobility the
  posterior never forgets stale measurements.
- *Likelihood misspecification.*  `|y(k,l)|` follows a Rician
  distribution, not Gaussian.  At high SNR the Gaussian approximation
  is acceptable; at low SNR it is increasingly biased.
- *Cold-start.*  K·L = 256 forced pulls.
- The previous implementation used `sigma = noise_amplitude`, which
  collapsed the posterior to a delta after a single pull.  The
  Welford-estimated variant tracks the actual reward volatility and
  remains exploratory.

**Use as.** Lower-bound sanity check.  Drop from primary figures.

---

### `HBM` — coarse-then-fine beam search

**What it is.** Sub-sampled DFT coarse sweep at stride `coarse_factor`
followed by NNS-style steepest-ascent fine refinement around the winning
sector.  Refresh into coarse mode every `refresh_every` steps.

**Cited reference.** Giordani et al. COMST 2019 §III-A (P1) and §III-B
(P2) — the 3GPP NR beam-management procedure.

**NOT.** Alkhateeb et al. JSAC 2014, despite earlier docstrings.
Alkhateeb's contribution is *purpose-designed wide-beam codewords* with
flat gain over each sector.  We just sub-sample the existing narrow DFT
codebook — different algorithm.

**Phase 4C upgrade path.** A learned hierarchical codebook (Yang TWC 2024
HBAN, Dreifuerst & Heath TMLCN 2024 X-BM) would replace the sub-sampled
DFT with end-to-end learned wide/narrow codewords.

---

### `OMPCompressive` — compressive channel-estimation OMP

**What it is.** Greedy Orthogonal Matching Pursuit on a sensing matrix
built from codebook outer-products `kron(conj(w_k), f_l)`, recovering
`vec(H)`.  Every `measurements_per_solve` steps the buffer of recent
measurements is solved; cached OBP exploits between solves.

**Cited reference.** Alkhateeb et al. JSAC 2014 §III-A — codebook
outer-product CS formulation; greedy recovery follows Tropp & Gilbert
2007.

**NOT.** Marzi et al. JSTSP 2016, despite earlier docstrings.  Marzi's
algorithm uses *pseudorandom-phase compressive beacons* (random ±1 RF
precoder phases) and runs Newtonised OMP (NOMP) in the continuous
spatial-frequency domain.  Our DFT codeword rows are highly mutually
coherent, so the formal compressive-recovery guarantees of Marzi 2016
do not apply.

**Phase 4C upgrade path.** Replace coherent DFT sensing with random-phase
beacons + NOMP, or upgrade to in-sector CS (Masoumi & Myers TCOM 2025).

---

### `DLPredictor` — MLP beam-prediction baseline

**What it is.** A 3-hidden-layer MLP that maps a 4-step OBP-index window
to a K·L-way classification over the next-step best beam.  Falls back to
`Exhaustive` when the checkpoint is missing; pass `require_checkpoint=True`
to make a missing checkpoint a hard error.

**Cited reference.** Kim et al. JSAC 2023 — the modern published
sequence-based DL baseline (LSTM on RSRP history).  Our MLP variant is
simpler than Kim 2023's architecture; the LSTM variant is in
`DLLSTMPredictor`.

**NOT.** Klautau et al. 2018, despite earlier docstrings.  Klautau uses
ray-tracing channel data and environment metadata as DL inputs, not OBP
history.

**Known limitation: train/inference distribution mismatch.**  Training
labels come from running `Exhaustive` (full sweep every step), so the
training OBP is the argmax of a fully-populated BPLM.  At inference time
the BPLM is partially-swept (one entry updated per step), so the OBP
distribution differs from training.  Documented but not fixed in this
baseline.

---

### `MAMBA` — non-stationary Thompson with neighbourhood explore

**What it is.** Discounted-update Gaussian Thompson sampling with a
neighbourhood-explore trigger.  When the current best arm's most-recent
reward drops by more than `explore_threshold * running_mean`, the
sampler restricts to the 4-connected neighbourhood of the best arm for
`explore_horizon` steps.

**Cited reference.** Aykin et al. INFOCOM 2020; journal version Krunz
et al. TMC 2024.  The Aykin 2020 paper jointly optimises beam +
modulation/coding scheme; we model only the beam component (our channel
returns a complex `y` and the runner reports SNR — there is no MCS).

**Use as.** The credible 2024-era MAB baseline.  Replaces stationary
Thompson as the primary bandit competitor to MCMD.

---

### `EKFTracker` — Kalman filter on AoA/AoD

**What it is.** Constant-angular-velocity EKF on the 4-D state
`[θ_AoA, θ_AoD, ω_AoA, ω_AoD]`.  Predicts the next-step beam pair as the
codebook indices closest to the predicted angles.  Latin-square diagonal
warmup ensures every UE row is sampled before the EKF takes over.

**Cited references.** Jayaprakasam et al. Commun. Lett. 2017; Burghal
et al. arXiv:1911.01638 (GlobalSIP 2019).

**Expected behaviour.**
- *Wins at low rotational speed.*  The constant-rate motion model is
  exact for steady rotation; the Kalman gain concentrates quickly.
- *Loses at high speed / abrupt motion.*  When angular rate exceeds one
  beam index per step, the model lags.  This is the classical
  limitation of model-based trackers and is documented in the cited
  references.
- *Static channel.*  Locks to a single OBP within `~80` steps after
  warmup (verified by `test_ekf_tracker_locks_onto_static_los`).

---

### `PositionMAB` — position-aided contextual MAB

**What it is.** Per-spatial-bin Thompson sampling.  Bin index is the
deterministic (x, y, yaw) grid cell of the UE pose at time `m`; each bin
maintains its own `(K × L)` posterior.

**Cited reference.** Va et al. IEEE Access 2019.  Va et al. use offline
KMeans on UE traces to define spatial clusters; we use a deterministic
grid for reproducibility.  An offline-clustered variant is left as a
Phase 4C upgrade.

**Expected behaviour.** On revisited spatial bins (rotational, periodic,
or repeated-handover scenarios), the bin's posterior is already
informative — no fresh cold-start.  Verified by
`test_position_mab_reuses_bin_posterior_on_revisit`.

---

### `BAIPureExploration` — successive-elimination best-arm identification

**What it is.** Naïve successive-elimination (Even-Dar et al. COLT 2002):
all arms start active; round-robin pulls; after each round, eliminate any
arm whose UCB falls below the LCB of the empirically best arm.  Once one
arm remains, exploit forever.

**Cited references.** Chiu et al. TWC 2022 (beam-correlation-aware BAI);
Even-Dar et al. COLT 2002 (PAC bounds).

**Difference from UCB1 / Thompson.** Different objective: minimise the
*probability of misidentification* under a fixed budget rather than
cumulative regret.  Closer to the operational goal of P1 acquisition.

**Caveats.** Hoeffding-based elimination on a `[0, R_max]` reward range
needs many pulls per arm before the confidence radius is below the gap.
Our test runs 32 sweeps; at shorter horizons elimination may be too
conservative.  An empirical-Bernstein variant would tighten this.

---

### `DLLSTMPredictor` — LSTM beam-sequence predictor

**What it is.** Single-layer LSTM (hidden=64) over a window of past OBP
pairs as a 2-D-per-step sequence; classifier head over the K·L flat
indices.

**Cited reference.** Kim et al. JSAC 2023 — the modern OTA-validated
sequence DL baseline.

**Training.**

```bash
python -m beamsim.algorithms._dl.train --model lstm \
    --output models/beam_predictor_lstm.pt
```

The trainer shares the same data-collection pipeline as the MLP variant
(`Exhaustive` on Case A UMi 10 m/s straight-line trajectories) — same
distribution caveat applies.

---

## Honest expectations vs. MCMD

| Baseline | MCMD vs baseline expectation |
|---|---|
| `Exhaustive` | MCMD wins on overhead-vs-SNR Pareto. |
| `Perfect` | Upper bound — MCMD must approach but cannot beat. |
| `UCB1` / `Thompson` (stationary) | MCMD wins comfortably (these are sanity bounds). |
| `HBM` (3GPP P1/P2) | MCMD must win — minimum bar.  If MCMD does not, there is a bug. |
| `MAMBA` | Approximate draw in steady state; MCMD wins by 1–2 dB in acquisition. |
| `EKFTracker` | MCMD loses at low rpm (constant-rate exact); wins at high rpm (model breaks). |
| `PositionMAB` | MCMD wins on first pass through a spatial bin; loses on repeated revisits. |
| `BAIPureExploration` | MCMD wins on cumulative SNR; BAI may win on time-to-stable-OBP. |
| `OMPCompressive` | MCMD wins at low SNR / mobility; loses at high-SNR static. |
| `DLPredictor` / `DLLSTMPredictor` | DL wins on trained scenario; loses under distribution shift. |

These are working hypotheses for the new paper.  Empirical confirmation
is part of the Phase 4C overhead-Pareto sweep.

---

## Reproducibility checklist

Every algorithm that uses an internal RNG must seed from
`context.get("trial_seed")`:

- ✅ `MAMBA`, `EKFTracker`, `PositionMAB`, `BAIPureExploration`,
  `Thompson`, `HBM`, `OMPCompressive`, `Perfect`, `Exhaustive`, `MCMD`,
  `Tabu`, `NNS`, `NNSBSSequential`, `AngularPrediction`,
  `ContextInformation`, `UCB1` (deterministic), `DLPredictor`,
  `DLLSTMPredictor`.

If a future algorithm is added, it must follow the same convention.

## New metrics for community release

`src/beamsim/metrics.py` ships:

| Function | Purpose |
|---|---|
| `output_snr_db` | Per-step SNR in dB. |
| `mean_snr_db` | Trial-mean SNR in dB. |
| `coverage_rate(γ_th)` | Fraction of steps with SNR ≥ γ_th (per trial). |
| `outage_fraction(γ_th)` | Complement: fraction below γ_th (per trial). |
| `outage_probability(γ_th)` | Pooled `Pr(SNR_dB < γ_th)` across trials and steps; NaN-propagating. |
| `oracle_snr_db` | Per-step **codebook** oracle: `max_{k,l} 10·log10(\|w_k^H H f_l\|² / σ_n²)`. The strongest comparator a measurement policy could achieve given the same codebook and channel realisation. *Not* Shannon capacity. |
| `snr_regret_db` | Per-step gap to the codebook oracle: `oracle - achieved`, lower is better, zero is optimal. |
| `beam_switch_rate` | Fraction of consecutive step pairs at which (k, l) changes — proxy for control-plane churn. |
| `bs_selection_loss` | L_BS in dB (handover quality). |
| `probing_overhead` | Distinct-arm probe count, normalised — for the 3GPP TR 38.843 overhead-vs-accuracy curve. |
| `top_k_accuracy` | Top-1 / top-k OBP-match against an oracle. |
| `time_to_realign` | Steps to recover SNR ≥ threshold after an explicit handover trigger. **Not** 3GPP BFR. |
| `bootstrap_ci` | BCa bootstrap CI for the mean (scipy.stats.bootstrap wrapper). |

These cover the headline metrics from 3GPP TR 38.843 §6.3 (beam
prediction accuracy) and §6.4 (overhead reduction), plus the handover
continuity metrics that few simulation papers report. The
codebook-oracle / regret pair gives a scenario-normalised diagnostic:
two algorithms that look similar on raw SNR can have very different
regret profiles once the per-step achievable ceiling is factored in.
