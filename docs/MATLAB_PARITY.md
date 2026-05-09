# MATLAB parity audit

This page records the cross-language audit of `beamsim` against the original
MATLAB simulator that produced the figures of the predecessor MSc thesis.
The MATLAB sources are not part of this repository --- they live in
`05_Clean_Simulator/` in the parent research directory and are referenced
by file path only. The audit was run on 2026-05-09 against `beamsim`
v0.2.1 by three parallel research subagents (channel model, codebook +
geometry, algorithms), and spot-verified by hand on the four highest-impact
findings.

The audit is a **reference document**: it tells you what is and is not
faithful, with the same level of detail at which a reviewer or external
re-implementer would ask. It is *not* a fix list. Where Python and MATLAB
diverge, the divergence is documented here and (where load-bearing) in the
paper's Sec.~4.4 Reproducibility-and-provenance section. Code-level
remediation is tracked separately in [`ROADMAP.md`](ROADMAP.md).

> **One-line summary.** `beamsim` is a faithful reproduction *in spirit* of
> the predecessor MATLAB simulator. The two implementations agree on all
> the headline scientific objects (codebook structure, BPLM contract,
> algorithm intent). They differ in concrete numerical conventions across
> half a dozen places --- some because the MATLAB original contains bugs
> that Python deliberately fixes, some because Python implements TR 38.901
> more strictly than the MATLAB shortcut, and some because the two were
> never going to be bit-identical (RNG, indexing, layout).

## Audit method

Each of three layers (channel model, codebook + geometry + arrays, MBP
algorithms + MCMD orchestration) was inspected by an independent agent
with read-only access. Each agent produced a categorised divergence
report; the synthesis below merges them and re-categorises every finding
as one of:

| Tag | Meaning |
|---|---|
| **Faithful** | Python and MATLAB compute the same scientific object up to language-level conventions (1-vs-0 indexing, row-vs-column layout). No remediation needed. |
| **Documented divergence** | Python and MATLAB differ; the difference is named in the paper (Sec.~4.4) at the time of writing. No additional disclosure needed. |
| **Undocumented divergence** | Python and MATLAB differ; the paper does not yet name the difference. Either the paper or this document must disclose it. |
| **Bug in MATLAB (Python correct)** | The MATLAB original deviates from the predecessor's stated algorithm; Python implements what the predecessor *intended*. Python is the more defensible reference. |
| **Bug in Python (MATLAB correct)** | Python diverges from the MATLAB original on a point where the MATLAB original is the source of truth. Worth fixing. |

## Findings

### Faithful (no action needed)

- **Geometric cluster powers.** MATLAB uses Friis on the extra path length;
  `beamsim/channel.py` uses the full TR 38.901 PL on `(d_ref + extra)`. Both
  encode "powers from extra path length"; the formulas differ only in the
  far-field tail.
- **Laplacian sub-ray AoA / AoD offsets** with scale $\sigma = \mathrm{cluster\,AS} / \sqrt{2}$.
- **Half-wavelength array spacing** at both UE and BS.
- **Element index runs $0\ldots N-1$** in both implementations.
- **OBP rule** `argmax_{k,l} |\tilde Y_{k,l}|` over the partially-stale BPLM.
- **Tabu tenure $s = 20$** in both implementations.
- **Tracking-priority scalar** $w_t = \mathrm{BQ}(\mathrm{BQ} + v) / 2$.
- **Exhaustive search** is structurally equivalent (MATLAB is implicit in
  the orchestrator; `beamsim/algorithms/exhaustive.py` is explicit).
- **Half-wavelength element index, and steering-sign convention** (Python's
  combined codebook-and-channel sign flips cancel out, producing the same
  beamformed magnitude as MATLAB).

### Documented divergences (already named in paper Sec.~4.4)

- **Channel cluster count.** MATLAB uses one large-scale tap (`reflnum=0`)
  for the standard `.mat` configurations; `beamsim` uses the TR 38.901
  Table 7.5-6 12 clusters / 20 sub-rays. Documented in Sec.~4.4
  (Channel model paragraph: "12 clusters / 20 rays per cluster").
- **LOS-only LSP table.** Paper specifies "LOS-LSPs always" regardless of
  LOS / NLOS state.
- **Sub-ray random initial phases omitted in Python.** Paper says
  "Per-sub-ray random initial phases are not drawn (Sec.~3.2 explicitly
  excludes them)". *Caveat:* the audit confirmed that the MATLAB code in
  fact applies `exp(1j * rand(c.M-1, 1) * 2π)` per timestep
  (`small_scaleFn.m:39`), so the *predecessor specification* and the
  *predecessor implementation* disagree; Python sides with the
  specification. Worth noting as a sub-bullet.
- **CI sin-space match.** Paper says "we match in sin(θ)-space to handle
  the ULA front/back half-plane ambiguity that the predecessor's prose
  treats implicitly". The audit confirms that MATLAB `updateCI.m` collapses
  the front/back half-plane via `abs(atan2(...))` and Python is the more
  faithful interpretation of the predecessor's *intent*.
- **MCMD endpoint weights from Fig.~5.26.** Paper says "reproduced from
  the predecessor's evolutionary-search output (its Fig.~5.26)". *Caveat:*
  the MATLAB simulator code committed `W_High = [0.9742, 0, 0.2733, 0, 0,
  0, 0.8742]` (normalised active-three: $(0.46, 0.13, 0.41)$) whereas the
  paper reports $(0.16, 0.36, 0.49)$. The two endpoints disagree even
  before Python's mapping enters the picture; either Fig.~5.26's
  pie-chart values were re-fitted between the figure and the simulator
  commit, or one of the two readings is wrong. The Python implementation
  uses the paper's published values, so changing it to match
  the MATLAB code would invalidate the existing MCMD figures. See
  Sec.~4.4 for the disclosure.

### Undocumented divergences (this audit's net-new findings)

#### Channel model

- **Cluster delay distribution.** MATLAB draws sub-ray excess delays from
  `Uniform(0, 2*sqrt(3) * DS)` then sorts (`small_scaleFn.m:37`). Python
  uses the TR 38.901 §7.5 Step 5 exponential `−r_τ · DS · ln(U)`. The
  Python choice is more standards-conformant; the MATLAB choice produces
  a bounded, symmetric delay spread. Magnitude of impact: tail of the
  delay profile differs, propagating into cluster Doppler and per-cluster
  power statistics in any non-noiseless figure.
- **Two-slope path-loss model.** MATLAB `PL_3gpp.m` deliberately omits the
  TR 38.901 breakpoint (its header comment: `BP calcs not included due to
  limited distances in model`); it always uses the near-field
  $32.4 + 21\log_{10}(d) + 20\log_{10}(f_c)$ formula. Python applies the
  full two-slope model with breakpoint $d_{BP} = 4(h_\mathrm{BS} - 1)(h_\mathrm{UT} - 1)f_c / c$,
  switching to the 40 dB/decade slope above $\sim$11.5 m at 28 GHz. At
  the IBS$/2 = 100$ m reference distance this produces an $\sim 8$--$12$ dB
  systematic SNR offset between the two implementations. The Python
  choice is the standards-conformant one; the MATLAB choice is the
  predecessor's working configuration.
- **Self-blocker attenuation.** MATLAB `blockage.m:31` keeps the
  self-blocker line as `attenuation(1, ...) = 0`; the 30 dB version is
  commented out one line above. The MATLAB simulator therefore models
  blockage with no self-shadowing component. Python applies the
  predecessor's specified 30 dB flat self-blocker over a 120° back-of-body
  cone (paper Sec.~4.4, "Model A: one self-blocker (120° wide centred at
  the back of the UE in body frame, 30 dB flat per Eq.~3.15)"). This
  divergence is *between the MATLAB code and the predecessor's prose*;
  Python sides with the prose.
- **KED $\lambda_\mathrm{eff}$ for non-self blockers.** MATLAB uses the
  carrier wavelength times the 10 m blocker radius
  (`blockage.m:48-51`), giving an effective scale near 28 at 28 GHz.
  Python hard-codes `lambda_eff = 0.4`, giving $\sim 6.85$. The KED
  attenuation depth differs by several dB at angles near the blocker
  edge. Both are inside the same general functional form but the
  numerical scale differs.

#### Codebook and geometry

- **Cosine-space sampling grid.** MATLAB `ulacodebook.m:26` uses
  `linspace(1, -1, N + 1)` with the last sample dropped, so the codebook
  *includes broadside* ($u = 1$). Python `codebook.py:40` uses the
  midpoint rule $(2k + 1) / N - 1$, which excludes both broadside and
  endfire. Misalignment magnitude: half a beamwidth ($1 / (2N)$ in
  cosine units). For $N = 32$ this is $0.031$, slightly less than 2°
  for an array steered near broadside. The figures' relative
  algorithm-vs-algorithm comparisons remain valid; absolute oracle SNRs
  are slightly lower in Python near broadside.
- **Codebook layout convention.** MATLAB `Codebook` matrix is
  `(n_beams, n_elements)` with rows as codewords; Python's
  `Codebook.matrix` is `(n_elements, n_beams)` with columns as codewords.
  The two conventions encode identical mathematical content; downstream
  code in both implementations applies the appropriate transpose.
- **Codebook normalisation asymmetry.** MATLAB normalises the BS codebook
  by $1 / N_\mathrm{tx}$ (`runTestcase.m:10`) and leaves the UE codebook
  un-normalised; Python normalises every codeword to unit norm
  (`codebook.steering_vector` divides by $\sqrt{N}$). Absolute received
  SNR differs by a constant scaling, but per-step achieved-vs-oracle
  comparisons are unaffected.
- **Tabu tie-breaking distance.** MATLAB precomputes circular distance
  matrices `rxdiff` and `txdiff` (`runTestcase.m:23-30`) and breaks ties
  by minimum circular Euclidean distance from the current OBP. Python
  `tabu.py` uses linear Euclidean distance on the beam grid. At the
  codebook boundary the difference can be up to $\lfloor N / 2 \rfloor - 1$
  beam slots; this matters for fast-rotating UEs whose oracle beam wraps
  around the codebook edge. The MATLAB convention is the load-bearing
  one for tabu's intent; the Python implementation is mildly biased
  away from edge transitions.

#### Algorithms

- **NNS internal structure.** MATLAB `updateAscentmx.m` maintains a
  fixed 5-probe explicit list and relocates the centre only to positions
  inside that list. Python `nns.py` implements a 4-connected stack with
  `obp()` global relocation. Both are valid local hill-climbers; they
  produce different probe sequences when the global OBP lies outside
  the recently-probed neighbourhood.
- **NNS-with-tabu (Ascent_Tabu) is missing in Python.** MATLAB
  distinguishes `updateAscentmx.m` (NNS without tabu) from
  `updateAscentmx_Tabu.m` (NNS with tabu, with global-search relocation
  when the probe list empties). Python collapses both into a single
  `nns.py`. MCMD's weight slot 7 (`C_Ascent_Tabu` in MATLAB) maps to
  plain NNS in the Python implementation.
- **Tabu cumulative penalty.** MATLAB does `T[k,l] -= s` (additive,
  cumulative); Python does `T[k,l] = -s` (absolute reset). Behaviour
  diverges only for cells that are revisited while still tabu, which is
  rare under aspiration-free defaults but non-zero under MCMD.
- **Tabu Python extensions.** Python `tabu.py` enables aspiration (a
  tabu cell whose magnitude exceeds the global best is admitted) and
  periodic diversification (random jump every 50 calls) by default;
  MATLAB has neither. For thesis parity the standalone Tabu baseline
  curves should be regenerated with these extensions disabled, or the
  divergence should be disclosed.
- **MCMD volatility / beam-quality scaling.** MATLAB uses a 51-step
  causal sum with `c_v = 6.1e3` in amplitude domain and a log-domain
  affine map for BQ. Python uses a 11-step rolling mean with `c_v = 30`
  in normalised domain and a linear-domain rescaling for BQ. Both
  produce a normalised $w_t \in [0, 1]$, but the time constants and
  saturation thresholds differ substantially.
- **MCMD tie-breaking.** MATLAB selects the tied cell minimising
  circular Euclidean distance to the current OBP and nudges its
  $R$-value by $+1$ before the final argmax. Python's
  `np.argmax(R)` returns the first row-major flat index with no
  tie-breaking. With all-zero criteria (trial start) MATLAB selects the
  cell nearest to OBP; Python always selects $(0, 0)$.

### Bugs in MATLAB (Python is the correct reference)

These are findings where the MATLAB original deviates from the
predecessor's *stated* algorithm and Python implements what the
predecessor intended.

- **Angular Prediction filter discards its own coefficients.** MATLAB
  `updateAngPredictionmx.m` applies `conv(angHist, [1, 0.5, 0.2])` and
  reads `UEpred(1)`, which by the convolution boundary returns just
  the oldest history entry $h_1$ — the filter coefficients $0.5$ and
  $0.2$ never enter the output. Python implements the
  gradient-sum predictor of Algorithm 3 with uniform $F(i) = 1$ as
  specified, which is the only way the predecessor's "AngPred"
  performance characterisation in Sec.~6.2 makes physical sense.
- **AgeMx tx / rx argument transposition.** MATLAB
  `updateAgemx(IN_agemx, rx_cb_I, tx_cb_I)` is called with
  `(C_agemx, tx_cb_I, rx_cb_I)` (`runTestcase.m:178`) — the second
  and third arguments are swapped relative to the function signature.
  The age criterion's row/column axes are therefore transposed
  relative to every other criterion. Python's `BPLMState` exposes
  `age_matrix` with consistent `(k_UE, l_BS)` axes and so produces the
  intended age criterion.
- **CI front/back half-plane fold.** MATLAB `updateCI.m` takes
  `abs(atan2(...))` of the AoA, collapsing both half-planes onto
  $[0, \pi/2]$. Python matches in `sin`-space, correctly resolving
  the ULA's front/back symmetry. (Already in paper Sec.~4.4.)
- **Rotation-track angular-unit mixing.** MATLAB `rotating_setup_Fn.m:7-8`
  adds `rand * 2π` (radians) to a degree-valued `linspace`. The
  intended uniform $[0°, 360°]$ initial-orientation scatter collapses
  to $[0°, \sim 6°]$. All Case~C iterations effectively share nearly
  the same starting orientation. Python's `rotation_track` accepts
  `initial_orientation` in radians without unit confusion.

### Bugs in Python (MATLAB is the correct reference)

- **MCMD `W_HIGH` numerical mismatch.** Python `mcmd.W_HIGH = (0.16, 0.36, 0.49)`
  (over the active criteria order age, tabu, NNS), while the MATLAB
  simulator commit-time value `W_High = [0.9742, 0, 0.2733, 0, 0, 0,
  0.8742]` normalises to $(0.46, 0.13, 0.41)$ over (age, tabu,
  ascent_tabu). The two are inconsistent in *both* the numerical values
  and the algorithm in slot 3 (Python's NNS vs MATLAB's
  Ascent_Tabu). Whichever is closer to the predecessor's Fig.~5.26
  intent is unclear without the lost training run; the Python value
  is what the paper currently reports, the MATLAB code is what
  *generated* the predecessor's MCMD figures. Code-level remediation
  is deferred to [`ROADMAP.md`](ROADMAP.md) because changing
  `W_HIGH` would shift the MCMD curves in every published figure.

## Severity ranking (top 5)

1. **MCMD `W_HIGH` mismatch** — different numerical weights *and* a
   different criterion in the third slot. Affects every MCMD curve at
   $w_t \to 1$ (high mobility, the regime where MCMD is supposed to
   shine).
2. **Two-slope vs single-slope path loss** — $\sim$10 dB systematic
   offset between MATLAB and Python at the IBS$/2 = 100$ m reference
   distance. Means the SNR-axis values in Python figures cannot be
   read off as if they were the MATLAB-equivalent SNR-axis values.
3. **Self-blocker 0 dB (MATLAB) vs 30 dB (Python)** — Python correctly
   implements the predecessor's specified Model A; MATLAB had it
   commented out. The two implementations therefore agree with the
   *prose* and disagree with the *figures*. Resolving requires either
   re-running MATLAB with the self-blocker re-enabled or documenting
   the discrepancy.
4. **Cluster delay distribution** — uniform vs exponential. Python is
   standards-conformant; MATLAB is the predecessor's working
   configuration. Magnitude of impact is the tail of the delay
   profile.
5. **Angular Prediction filter bug in MATLAB** — the predecessor's
   AngPred figures are produced by what is effectively a
   one-step-history null filter rather than the gradient-sum
   predictor described in Algorithm 3. Python implements the spec
   correctly. The qualitative paper claim ("AngPred works in clean
   LOS smooth motion, fails in multipath") still holds, but the
   quantitative MATLAB curves are what an *unfiltered* predictor
   produces.

## How to read the paper alongside this document

The paper's Sec.~4.4 (Reproducibility and provenance) lists the
*predecessor-to-Python* divergences --- treating the predecessor's
*prose* as ground truth. This document additionally records the
*MATLAB-code-to-Python* divergences --- treating the predecessor's
*code* as ground truth. The two views agree on most things; where
they disagree, the difference is usually a MATLAB bug or a
predecessor-prose-vs-predecessor-code inconsistency, and the
specific case is named in the corresponding subsection above.

Code-level remediation of the bugs is tracked in
[`ROADMAP.md`](ROADMAP.md). Most fixes are deferred because they
would invalidate existing figures in the paper or require re-running
the full simulation campaign at a moderate compute cost.
