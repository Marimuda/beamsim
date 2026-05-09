# Roadmap

This page tracks deferred work that is *intentionally* scoped out of the
current `0.1.x` line. Each item lists the motivation, the architectural
change required, and what would unblock the work. Open issues should
reference the heading slug as a label.

> The repository's `0.1.x` line is locked to **codebook-based beam
> alignment / tracking under common random numbers, with a fixed
> one-probe-per-step measurement schedule**.  Any item below that
> changes that contract should arrive behind a clearly named optional
> extra and a corresponding entry in
> [`SOTA_BASELINES.md`](SOTA_BASELINES.md).

## Per-algorithm measurement budgets

**Status:** deferred.

**What it is.** `beamsim` currently issues exactly one BPLM measurement
per simulator step for every algorithm. The
[measurement-budget taxonomy](SOTA_BASELINES.md#algorithms-by-measurement-budget)
documents the *target* probe budgets each policy is meant to operate
under in the literature, but the simulator collapses them all to a
one-step-one-probe schedule for like-for-like comparison.

**Why this matters.** Modern beam-management evaluation treats
measurement count as a first-class metric. Reporting "algorithm A
matched algorithm B's SNR while probing 5 % of the codebook" is the
kind of statement we should be able to support natively, not infer
post-hoc from the OBP history.

**Architectural change.** Extend the `Algorithm` interface
(`algorithms/base.py`) so a policy can return a list of probes per
step, not a single (k, l). The runner needs:

1. an iterator-shaped `select_next_mbps(state, m, context) -> Iterable[tuple[int, int]]`
   alongside the current `select_next_mbp` (keep both during the
   migration);
2. per-step probe-count accounting in `TrialResult`;
3. a fairness convention for the comparison: e.g. cap the global
   per-step probe count at `max_probe_budget`, share it among
   algorithms via a budget allocator, and record exhaustion events.

**What would unblock it.** A concrete use case where the regret /
overhead trade-off only becomes visible with non-unit budgets — for
example, a sub-6-fingerprint-aided initial-access experiment.

## Reacquisition time after blockage

**Status:** deferred.

**What it is.** `metrics.time_to_realign(snr_db, threshold_db,
handover_step)` already measures recovery time after an *explicit*
handover trigger, which is the right primitive for the BS-handover
experiments shipped today. What is missing is the *blockage-driven*
analogue: detect a blockage event in the channel state, then measure
the steps until the algorithm regains a beam pair above threshold.

**Why this matters.** Modern surveys treat reactive blockage robustness
as a baseline against which proactive (predictive) systems are
compared. We should be able to publish reactive-recovery latency for
every algorithm in the roster without instrumenting it by hand from
each `.npz`.

**Architectural change.** `channel.py`'s Model A blockage logic needs
to expose a per-step blockage state vector that the runner records
into `TrialResult`. Then a metric `reacquisition_time_after_blockage(
snr_db, blockage_state, threshold_db)` slices the SNR trace at every
falling edge of `blockage_state` and reuses `time_to_realign`'s search
loop. The metric should be censored when no recovery occurs within a
configurable horizon (consistent with the existing `max_search`).

**What would unblock it.** A representative blockage-heavy scenario
that the existing `experiments/exp_*.py` scripts do not yet cover, plus
a stable channel-state field name (the current `BlockageState` is
already in `channel.py` but is not threaded through `_run_trial`).

## Optional: ray-traced channel adapter

**Status:** speculative.

The `channel.ChannelRealisation` boundary is intentionally narrow so
that a DeepMIMO- or Sionna-RT-backed adapter could implement the same
`channel_matrix(ue_xy, ue_yaw)` contract. If you want to pilot this,
ship it behind a `[rt]` optional extra and document the adapter's
fidelity caveats in `docs/related_work.md` § *Simulator-fidelity
hierarchy*.
