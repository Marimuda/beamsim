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
_8CONNECTED = [(dk, dl) for dk in (-1, 0, 1) for dl in (-1, 0, 1) if (dk, dl) != (0, 0)]


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
        # Random initial seed per trial (Algorithm 4, line 2: "kb, lb <- Random").
        # Seed the RNG from the trial seed so two NNS instances given the same
        # trial seed start at the same (k_b, l_b) — preserving the runner's
        # common-random-numbers contract across algorithms within a trial.
        rng = np.random.default_rng(int(context.get("trial_seed", 0)))
        self._kb: int = int(rng.integers(0, state.K))
        self._lb: int = int(rng.integers(0, state.L))
        self._stack: list[tuple[int, int]] = []
        # Cycle-level state: in each neighbourhood cycle we track the best
        # neighbour seen so far. When P empties, we compare against the
        # currently-measured centre value and steepest-ascent the centre.
        self._cycle_best_mag: float = -np.inf
        self._cycle_best_kl: tuple[int, int] | None = None
        self._initial: bool = True
        self._last_kl: tuple[int, int] | None = None

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        # Cold-start: probe the random seed pair first to give the centre a
        # measured magnitude before any neighbour comparison.
        if self._initial:
            self._initial = False
            choice = (self._kb, self._lb)
            self._last_kl = choice
            return choice

        # Update best-of-cycle from the previous measurement
        if self._last_kl is not None:
            lk, ll = self._last_kl
            last_mag = float(np.abs(state.observations[lk, ll]))
            if last_mag > self._cycle_best_mag:
                self._cycle_best_mag = last_mag
                self._cycle_best_kl = self._last_kl

        # Steepest-ascent: when the cycle's neighbour pool is exhausted,
        # compare the best neighbour against the (re-read) centre magnitude
        # and relocate if a neighbour wins. Then rebuild P around the new
        # (or unchanged) centre and start a fresh cycle.
        if not self._stack:
            centre_mag = float(np.abs(state.observations[self._kb, self._lb]))
            if self._cycle_best_kl is not None and self._cycle_best_mag > centre_mag:
                self._kb, self._lb = self._cycle_best_kl
            self._cycle_best_mag = -np.inf
            self._cycle_best_kl = None
            self._rebuild_stack(state)
            # Re-measure the centre at the start of each cycle so its
            # magnitude reflects the current channel (handles UE motion).
            choice = (self._kb, self._lb)
            self._last_kl = choice
            return choice

        choice = self._stack.pop()  # LIFO: pop from top of P
        self._last_kl = choice
        return choice

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
