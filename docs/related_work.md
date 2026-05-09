# Related work and field context

`beamsim` reproduces the algorithms of an MSc-thesis-era beam-alignment
problem. Since 2018, the literature has reframed the problem in several
important ways. This page situates the repository inside that broader
context: what `beamsim` does cover, what the modern field has moved on
to, and which directions are explicitly **out of scope** here.

If you are looking for the algorithms shipped today, see
[`SOTA_BASELINES.md`](SOTA_BASELINES.md) for the per-baseline reference card.

## From beam alignment to beam management

The 2018 framing — *select good TX/RX beam pairs from a codebook under
mobility and blockage* — is now treated as one component of the broader
**beam-management** lifecycle that 5G NR / 5G-Advanced standardised:

- beam sweeping / initial access,
- beam measurement and reporting,
- beam refinement,
- beam failure detection and recovery,
- beam prediction,
- multi-TRP and multi-cell coordination,
- positioning-assisted beam selection,
- AI/ML-assisted beam management.

3GPP Release 18 explicitly studied AI/ML for the NR air interface, with
beam management as one of three representative use cases (alongside CSI
feedback and positioning). See the 3GPP RAN1 Rel-18 summary
(<https://www.3gpp.org/technologies/ran1-rel18>) and the AI/ML for NR Air
Interface page (<https://www.3gpp.org/technologies/ai-ml-nr>).

`beamsim` covers **beam alignment within mobility-aware tracking**. It
does not cover the full lifecycle. Specifically, it does not model
beam-failure-recovery RACH procedures, multi-TRP coordination signalling,
or NR-style measurement-report quantisation.

### Initial access vs. tracking vs. recovery

Modern beam management is more usefully read as three distinct stages,
each with its own constraints, baselines, and performance criteria:

- **Initial access / acquisition** — find a viable beam pair from
  little or no prior state. Exhaustive sweep is the canonical baseline;
  hierarchical search and compressive sensing are the principled
  reductions. In `beamsim` this stage is `Exhaustive`, `HBM`,
  `OMPCompressive`, and the warm-up phase of every adaptive algorithm.
- **Tracking** — maintain a good beam pair under mobility and blockage
  using prior state. Steepest-ascent neighbour search, tabu search,
  position-aided MAB, EKF, MAMBA, MCMD, and the DL predictors all live
  here. This is the stage `beamsim` evaluates most carefully.
- **Recovery** — re-establish alignment after blockage failure or
  severe degradation. 3GPP NR specifies an explicit beam-failure
  detection and recovery (BFR) procedure with RACH-based reacquisition.

**Recovery is out of scope for `beamsim`.** Algorithms degrade
gracefully under Model A blockage (the channel pushes them back into
exploration), but the simulator does not model BFR signalling, RACH
latency, or beam-failure-instance counters. The
`metrics.time_to_realign` helper measures *handover-style* recovery —
the steps until SNR re-crosses a threshold after an explicit
handover_step trigger — which is a useful proxy but not 3GPP-conformant
BFR. Cross-trial reacquisition-time-after-blockage instrumentation is
on the roadmap (see [`ROADMAP.md`](ROADMAP.md)).

## The bottleneck shifted toward overhead and latency

With larger arrays, narrower beams, multi-panel devices, and FR2
deployment constraints, the dominant cost of beam management is now
**measurement overhead, latency, and signalling**, not just the SNR of
the chosen beam pair. Recent 5G-Advanced AI/ML work explicitly motivates
**beam prediction** as a way to replace or shrink the sequential beam
sweep (see e.g. the 5G-Advanced AI/ML Beam Management survey at
<https://arxiv.org/html/2404.15326v1>).

A modern evaluation framework should therefore report:

- beam-training overhead (number of probed pairs);
- top-K beam accuracy against the oracle codebook entry;
- regret to the oracle SNR;
- beam-switching rate;
- beam-failure rate and recovery latency;
- signalling load and energy cost;
- robustness under blockage transitions and mobility regimes;
- cross-scenario / sim-to-real generalisation;
- calibration sensitivity.

`beamsim` reports SNR, coverage rate, and `L_BS` per algorithm, plus
optional alignment with 3GPP TR 38.843 metrics (Phase 4C). Top-K
accuracy and oracle-regret instrumentation are natural near-term
extensions but not currently part of the simulator.

## Simulator-fidelity hierarchy

A useful way to read the field is along this fidelity ladder:

```text
toy geometry  →  stochastic geometry  →  ray-traced site  →  measured deployment
```

- **Toy geometry**: free-space LOS with one BS, used for validation.
- **Stochastic geometry**: 3GPP TR 38.901 cluster-delay-line channels,
  configured per scenario (UMi, UMa, RMa). This is `beamsim`'s working
  fidelity.
- **Ray-traced site**: site-specific deterministic propagation produced
  by tools like Sionna RT, NVIDIA Aerial, or Wireless InSite. The
  reference dataset family in this category is **DeepMIMO**
  (<https://www.deepmimo.net/>, paper at
  <https://arxiv.org/abs/1902.06435>), which formalises parameterised
  ray-traced wireless datasets specifically for ML on mmWave/massive
  MIMO. **Raymobtime** and the **LiDAR mmWave Beam Selection** dataset
  (<https://github.com/MatteoEURECOM/LIDAR-mmWave-Beam-Selection>) push
  toward multimodal beam selection.
- **Measured deployment**: real-world mmWave channel sounders. Outside
  the scope of any synthetic simulator.

`beamsim` lives in the stochastic-geometry tier. The simulator boundary
in `channel.ChannelRealisation` is intentionally narrow so it could be
swapped for a ray-traced source later, but no such adapter is shipped.

## Context-aided beam management

A large body of recent work uses **side information** to shrink the beam
search before any RF measurement:

- UE position / GNSS / IMU state;
- map geometry and ray-tracing priors;
- camera images, LiDAR, radar, and ISAC sensing;
- previous beam histories;
- sub-6 GHz channel fingerprints;
- neighbour-cell measurements.

The repo's `algorithms.PositionMAB` and `algorithms.AngularPrediction`
hint at this direction (position-conditioned arm selection, AoD-derived
candidate beams). Full multimodal context (imagery, LiDAR, radar) is
**out of scope** here.

## ML-based beam prediction

The current ML-for-beam-management literature spans:

- supervised beam prediction from position, prior RSRP/CSI, or
  environment features;
- recurrent / temporal models for beam tracking;
- reinforcement learning for adaptive search;
- graph-based or map-aware predictors;
- federated learning for multi-cell beam management;
- uncertainty-aware top-K beam prediction;
- transfer learning across deployment sites;
- AI/ML lifecycle topics: data collection, model update, inference
  placement, UE/network split.

`beamsim` ships `DLPredictor` (MLP) and `DLLSTMPredictor` (recurrent)
behind the optional `[dl]` extra so they remain reproducible without
forcing a torch dependency on every user. The deeper open problem —
**generalisation under deployment shift** (new streets, blockers, BS
geometry, UE distribution, panels, weather/foliage) — is the
benchmarking gap a future `beamsim` evolution could occupy.

## What stayed valid from the predecessor work

Several core insights of the 2018 thesis remain accurate:

- mmWave channels are sparse relative to sub-6 GHz.
- Blockage is a first-order problem.
- Beam alignment is tightly coupled to mobility.
- Codebook design matters.
- Exhaustive search does not scale.
- Naive hierarchical search can fail because broad beams have low gain.
- Geometry is essential for meaningful mobility simulations.
- Adaptive tracking is more realistic than one-shot selection.

What dates the predecessor framing is **scope and tooling**, not
physical intuition.

## Out of scope (for clarity)

`beamsim` does not currently model or simulate:

- reconfigurable intelligent surfaces (RIS);
- near-field / sub-THz beam *focusing* (beams as range-aware spatial
  foci rather than far-field angular directions);
- multi-TRP joint transmission and coordinated handover;
- multimodal sensing (camera, LiDAR, radar);
- a deployment-grade 3GPP conformance pipeline;
- ray-traced or measured channels;
- federated / distributed model training.

These are recognised modern directions; they are deliberately excluded
to keep the repository's scope honest. Pull requests adding any of them
should land behind clearly named optional extras and a corresponding
entry in [`SOTA_BASELINES.md`](SOTA_BASELINES.md).

## Selected entry points to the broader literature

- 3GPP RAN1 Release 18 summary —
  <https://www.3gpp.org/technologies/ran1-rel18>
- 3GPP AI/ML for NR Air Interface —
  <https://www.3gpp.org/technologies/ai-ml-nr>
- 5G-Advanced AI/ML Beam Management (survey) —
  <https://arxiv.org/html/2404.15326v1>
- DeepMIMO dataset and framework —
  <https://www.deepmimo.net/>, paper <https://arxiv.org/abs/1902.06435>
- LiDAR mmWave Beam Selection (multimodal) —
  <https://github.com/MatteoEURECOM/LIDAR-mmWave-Beam-Selection>
- *A Survey of Beam Management for mmWave and THz* (IEEE COMST 2024) —
  <https://dl.acm.org/doi/abs/10.1109/COMST.2024.3361991>
