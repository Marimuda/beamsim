"""Nearest-Neighbour Search: hill-climbing local-peak tracker.

Implements Algorithm 4 from the predecessor MSc thesis
(Kristmundsson & Syberg, 2018), Section 5.4.4, faithfully.

Algorithm 4 (thesis, p. 65):
  1. if Initial Step:
       kb, lb <- Random
       xi <- 0
  2. if Y[k,l] > xi:
       kb, lb <- k, l
       xi <- Y[k,l]
  3. if Size P == 0:
       Push N(kb, lb) into P
       xi <- 0           # reset xi when centre moves / P is rebuilt
  4. pop [k, l] from top of P  (LIFO stack)

Key implementation choices matching the report:
  - P is a LIFO stack (list with append/pop).
  - Seed (kb, lb) drawn randomly per trial via np.random.default_rng().
  - xi resets to 0 every time P is rebuilt (line 11 in Algorithm 4).
  - 4-connected neighbourhood: offsets {(0,+1),(0,-1),(+1,0),(-1,0)}.

Reference:
  Kristmundsson & Syberg (2018), "Beam alignment methods for terminals in
  millimeter-wave wireless networks", Aalborg University MSc thesis WCS10-951,
  Section 5.4.4, Algorithm 4 (pp. 64-66).
"""

from __future__ import annotations

import numpy as np

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState

_4CONNECTED = [(0, 1), (0, -1), (1, 0), (-1, 0)]
_8CONNECTED = [(dk, dl) for dk in (-1, 0, 1) for dl in (-1, 0, 1)
               if (dk, dl) != (0, 0)]


class NNS(Algorithm):
    """Hill-climbing nearest-neighbour search (Algorithm 4, thesis Sec. 5.4.4).

    Parameters
    ----------
    connectivity : int
        4 (default, pattern A in Fig. 5.22) or 8 (pattern B).
    """

    name = "nns"

    def __init__(self, connectivity: int = 4):
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        self._offsets = _4CONNECTED if connectivity == 4 else _8CONNECTED

    # ------------------------------------------------------------------
    # Algorithm interface
    # ------------------------------------------------------------------

    def reset(self, state: BPLMState, context: dict) -> None:
        # Random initial seed per trial (Algorithm 4, line 2: "kb, lb <- Random")
        rng = np.random.default_rng()
        self._kb: int = int(rng.integers(0, state.K))
        self._lb: int = int(rng.integers(0, state.L))
        self._xi: float = 0.0      # best observed magnitude threshold
        self._stack: list[tuple[int, int]] = []   # LIFO stack P
        self._initial: bool = True  # flag for first occasion

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        # Cold-start: probe the random seed pair first
        if self._initial:
            self._initial = False
            return self._kb, self._lb

        # Algorithm 4 lines 5-8: if current OBP > xi, update centre
        ok, ol = state.obp()
        obp_mag = float(np.abs(state.observations[ok, ol]))
        if obp_mag > self._xi:
            self._kb, self._lb = ok, ol
            self._xi = obp_mag
            self._stack.clear()

        # Algorithm 4 lines 9-12: if P empty, rebuild from N(kb, lb) and reset xi
        if not self._stack:
            self._rebuild_stack(state)
            self._xi = 0.0   # Algorithm 4 line 11: xi <- 0

        if self._stack:
            return self._stack.pop()   # LIFO: pop from top
        return self._kb, self._lb

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _neighbours(self, state: BPLMState) -> list[tuple[int, int]]:
        K, L = state.K, state.L
        result = []
        for dk, dl in self._offsets:
            nk = self._kb + dk
            nl = self._lb + dl
            if 0 <= nk < K and 0 <= nl < L:
                result.append((nk, nl))
        return result

    def _rebuild_stack(self, state: BPLMState) -> None:
        """Populate the LIFO stack P from N(kb, lb)."""
        neighbours = self._neighbours(state)
        # Push all neighbours; pop order (LIFO) reverses the push order.
        # Report Algorithm 4 does not specify an ordering — push in natural order.
        self._stack = list(neighbours)
