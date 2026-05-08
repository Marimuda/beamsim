"""Nearest-Neighbour Search: hill-climbing local-peak tracker.

Implements the NNS algorithm described in the predecessor MSc thesis
(Kristmundsson & Syberg, 2018), Section 5.4.4, Algorithm 4.

The report describes NNS as a steepest-ascent local search:
  1. Maintain a current centre (k_b, l_b) — the best beam-pair found so far.
  2. Each call pops the next candidate from a measurement queue P built from
     N(k_b, l_b) — the 4-connected neighbourhood (pattern A in Fig. 5.22).
  3. After measuring all neighbours, compare their observations against the
     current best xi; if any neighbour beats xi, move k_b to it and rebuild P.

Implementation choice — 4-connected neighbourhood (pattern A, Fig. 5.22):
  offsets = {(0,+1),(0,-1),(+1,0),(-1,0)}.
  This gives N_NB = 4, search time = 5 occasions per hill-climb step, and
  maximum tracking velocity TV_max = R / (N_NB + 1) * BW/(2pi) per
  Equation 5.24.  8-connected (pattern B) is a configurable option but
  increases cost to N_NB = 8.

When all neighbours have been observed recently (age < staleness_threshold),
the algorithm falls back to the highest-magnitude observed neighbour rather
than re-measuring stale ones, keeping measurement cost bounded.

Cold-start (no measurements yet): returns (0, 0) as the seed; the algorithm
begins climbing from there on the next occasion.

Reference:
  Kristmundsson & Syberg (2018), "Beam alignment methods for terminals in
  millimeter-wave wireless networks", Aalborg University MSc thesis WCS10-951,
  Section 5.4.4, Algorithm 4 (pp. 64–66).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState

_4CONNECTED = [(0, 1), (0, -1), (1, 0), (-1, 0)]
_8CONNECTED = [(dk, dl) for dk in (-1, 0, 1) for dl in (-1, 0, 1)
               if (dk, dl) != (0, 0)]


class NNS(Algorithm):
    """Hill-climbing nearest-neighbour search.

    Parameters
    ----------
    connectivity : int
        4 (default) or 8; controls how many neighbours define N(k_b, l_b).
    staleness_threshold : int
        Age in occasions below which a measurement is considered "recent"
        and the algorithm skips re-measuring it (picks highest-observed
        instead).  Default 0 disables this shortcut, always probing stale.
    """

    name = "nns"

    def __init__(self, connectivity: int = 4, staleness_threshold: int = 0):
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        self._offsets = _4CONNECTED if connectivity == 4 else _8CONNECTED
        self._staleness_threshold = staleness_threshold

    # ------------------------------------------------------------------
    # Algorithm interface
    # ------------------------------------------------------------------

    def reset(self, state: BPLMState, context: dict) -> None:
        self._kb: int = 0
        self._lb: int = 0
        self._xi: float = 0.0          # best observed magnitude at (kb, lb)
        self._queue: deque[tuple[int, int]] = deque()
        self._has_seed: bool = False    # True once we've measured the seed

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        if not np.any(state.measured_at >= 0):
            # Cold-start: probe the seed pair first
            return self._kb, self._lb

        # Update centre if OBP has drifted above our local best
        self._sync_centre(state)

        # If queue empty, build it from the neighbourhood of the centre
        if not self._queue:
            self._build_queue(state, m)

        if self._queue:
            return self._queue.popleft()
        # Degenerate case: no valid neighbours (edge of codebook)
        return self._kb, self._lb

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_centre(self, state: BPLMState) -> None:
        """Move the centre to the current OBP if it has higher magnitude."""
        ok, ol = state.obp()
        obp_mag = float(np.abs(state.observations[ok, ol]))
        if obp_mag > self._xi:
            self._kb, self._lb = ok, ol
            self._xi = obp_mag
            self._queue.clear()       # rebuild neighbourhood from new centre

    def _neighbours(self, state: BPLMState) -> list[tuple[int, int]]:
        K, L = state.K, state.L
        result = []
        for dk, dl in self._offsets:
            nk = self._kb + dk
            nl = self._lb + dl
            if 0 <= nk < K and 0 <= nl < L:
                result.append((nk, nl))
        return result

    def _build_queue(self, state: BPLMState, m: int) -> None:
        """Populate the measurement queue from N(k_b, l_b).

        Ordering strategy (report Sec. 5.4.4, step 2):
          - Prefer stale (oldest) neighbours — they give the most information.
          - If all are recent (age < staleness_threshold), still sort by age
            descending so the most stale is first; the threshold only
            affects whether we skip or include entries.
        """
        neighbours = self._neighbours(state)
        if not neighbours:
            return

        ages = []
        for nk, nl in neighbours:
            at = state.measured_at[nk, nl]
            age = (m - at) if at >= 0 else (m + 1)
            ages.append(age)

        # Sort by descending age (stalest first)
        order = sorted(range(len(neighbours)), key=lambda i: -ages[i])

        if self._staleness_threshold > 0:
            # Skip recently measured neighbours; only re-probe stale ones
            stale = [neighbours[i] for i in order if ages[i] >= self._staleness_threshold]
            if stale:
                self._queue = deque(stale)
                return
            # All recent: pick the single highest-magnitude neighbour
            best = max(neighbours, key=lambda p: float(np.abs(state.observations[p])))
            self._queue = deque([best])
        else:
            self._queue = deque(neighbours[i] for i in order)
