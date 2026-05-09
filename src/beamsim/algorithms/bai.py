"""Best-Arm Identification (BAI) via successive elimination for fast beam
acquisition.

Reference:
    Chiu, S.-E., Zhang, R., Gu, Y. (2022). "Fast Beam Alignment via Pure
    Exploration in Multi-Armed Bandits." IEEE Transactions on Wireless
    Communications.  DOI: 10.1109/TWC.2022.3217131
    Foundational: Even-Dar, E., Mannor, S., Mansour, Y. (2002). "PAC
    bounds for multi-armed bandit and Markov decision processes." COLT.

Difference from UCB1 / Thompson
-------------------------------
UCB1 and Thompson minimise *cumulative regret* over an infinite horizon;
they are designed for "play forever, lose less while you learn."

Best-arm identification has a different objective: *minimise the
probability of misidentifying the best arm subject to a fixed
measurement budget*.  In beam-alignment terms: "given B beam probes,
return the most likely best (k, l)."

Successive elimination achieves this by maintaining a confidence
interval per arm and eliminating arms whose UCB falls below the LCB of
the empirically best arm.  Once an arm is eliminated, no further measure-
ments are spent on it; once one arm remains, exploitation kicks in for
the remaining horizon.

This baseline is conceptually closer to the operational goal of beam
acquisition in 3GPP NR P1 (find the best beam fast and then exploit)
than the cumulative-regret bandits, and exposes whether MCMD's advantage
is in *acquisition speed* or *steady-state tracking*.

Implementation
--------------
We use the *naive* successive-elimination algorithm (Even-Dar et al.):
  - All arms start active.
  - At each round, the active arms are pulled in cyclic order.
  - After each full round, eliminate any arm whose empirical mean is more
    than ``2 * confidence_radius(t)`` below the best active arm's mean.
  - Once one arm remains, return it forever.

We do not implement Chiu 2022's beam-correlation-aware variant (which
exploits spatial correlation between adjacent beams to eliminate
neighbours simultaneously); that is left as a Phase 4C upgrade.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState


class BAIPureExploration(Algorithm):
    """Successive-elimination best-arm identification.

    Parameters
    ----------
    delta:
        Confidence parameter (probability of misidentification).  Smaller
        ``delta`` means more conservative elimination.
    min_pulls_per_arm:
        Minimum number of pulls per arm before considering elimination.
        Prevents premature elimination on a single noisy measurement.
    """

    name = "bai_pure_explore"

    def __init__(self, delta: float = 0.1, min_pulls_per_arm: int = 2) -> None:
        self._delta = float(delta)
        self._min_pulls = max(int(min_pulls_per_arm), 1)

    def reset(self, state: BPLMState, context: dict) -> None:
        K, L = state.K, state.L
        n_arms = K * L
        self._n_arms = n_arms
        self._mean: NDArray[np.float64] = np.zeros(n_arms, dtype=np.float64)
        self._counts: NDArray[np.int_] = np.zeros(n_arms, dtype=np.int_)
        # Active set: starts as all arms; arms are removed by elimination.
        self._active: NDArray[np.bool_] = np.ones(n_arms, dtype=bool)
        self._round_robin_idx: int = 0
        self._t: int = 0
        self._last_arm: int | None = None
        self._winner: int | None = None
        self._L = L

    def _confidence_radius(self, n: int) -> float:
        # Hoeffding-based confidence radius for [0, R_max] rewards;
        # we use the running max as R_max (heuristic, see UCB1 module
        # docstring for the same caveat).  Falls back to 1 if no
        # observations yet.
        if n <= 0:
            return float("inf")
        # log(2 / delta) per arm; we use n_arms / delta as the union-bound
        # adjusted denominator.
        log_factor = float(np.log(2.0 * self._n_arms / max(self._delta, 1e-9)))
        return float(np.sqrt(log_factor / (2.0 * n)))

    def _eliminate(self) -> None:
        # Eliminate arms whose UCB falls below LCB of the best.  Only
        # consider arms with at least ``min_pulls`` measurements.
        active_idx = np.where(self._active)[0]
        if len(active_idx) <= 1:
            return
        # Restrict to arms with enough pulls to be evaluated.
        eligible = active_idx[self._counts[active_idx] >= self._min_pulls]
        if len(eligible) <= 1:
            return
        means = self._mean[eligible]
        # Per-arm confidence radii.
        radii = np.array([self._confidence_radius(int(self._counts[a])) for a in eligible])
        # Reward-range scaling: we use the running max of observed
        # rewards across all (active) arms.  Same caveat as UCB1.
        r_max = float(np.max(self._mean)) if np.any(self._mean > 0.0) else 1.0
        radii = radii * r_max
        ucb = means + radii
        lcb = means - radii
        best_lcb = float(np.max(lcb))
        # Eliminate any arm whose UCB < best LCB.
        for a, u in zip(eligible, ucb):
            if u < best_lcb:
                self._active[a] = False

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        L = self._L

        # Update statistics from the arm pulled last step.
        if self._last_arm is not None:
            a_prev = self._last_arm
            pk, pl = a_prev // L, a_prev % L
            reward = float(np.abs(state.observations[pk, pl]))
            n_new = self._counts[a_prev] + 1
            self._mean[a_prev] += (reward - self._mean[a_prev]) / n_new
            self._counts[a_prev] = n_new
            self._t += 1

        # If a winner has been declared, pull it forever.
        if self._winner is not None:
            k = self._winner // L
            l = self._winner % L
            self._last_arm = self._winner
            return k, l

        # Try elimination once per round (every n_arms / 2 pulls is a
        # reasonable heuristic; we just do it every n_active pulls).
        active_idx = np.where(self._active)[0]
        if len(active_idx) >= 2 and self._t > 0 and self._t % len(active_idx) == 0:
            self._eliminate()
            active_idx = np.where(self._active)[0]
            if len(active_idx) == 1:
                self._winner = int(active_idx[0])
                k = self._winner // L
                l = self._winner % L
                self._last_arm = self._winner
                return k, l

        # Round-robin over the active arms.
        active_idx = np.where(self._active)[0]
        if len(active_idx) == 0:
            # Defensive: should not happen, but if every arm got eliminated
            # we just return arm 0 to avoid crashing the runner.
            self._last_arm = 0
            return 0, 0

        choice = int(active_idx[self._round_robin_idx % len(active_idx)])
        self._round_robin_idx += 1
        k = choice // L
        l = choice % L
        self._last_arm = choice
        return k, l
