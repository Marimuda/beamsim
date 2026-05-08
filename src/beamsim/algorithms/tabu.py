"""Tabu search around the OBP.

Implements Algorithm 5 from the predecessor MSc thesis
(Kristmundsson & Syberg, 2018), Section 5.4.5 (pp. 67-68), with optional
Glover aspiration and diversification as extensions.

Algorithm 5 (thesis, p. 68):
  1. if Initial Step: kb, lb <- Random, T <- zeros(K,L)
  2. if Size P == 0:
       Push N(kb, lb) into P
       xi <- 0
  3. if Y[k,l] > xi:
       kb, lb <- k, l
       xi <- Y[k,l]
  4. if [k,l] == top of P: delete [k,l] from top of P
  5. [k,l] <- top of P  (read without deleting)
  6. T <- T + 1, T(T > 0) <- 0     # increment, cap at 0
  7. if T[k,l] < 0:
       k,l <- argmin_{{T=0}} ||[k_hat,l_hat] - [k,l]||  (closest non-tabu)
  8. T[k,l] <- -s                   # make tabu for s steps

Where T is the tabu matrix (0 = free, negative = tabu), s is the tenure.
Default tenure s = 20 (thesis Figure 5.23 caption).

Neighbourhood: 4-connected (same pattern A as NNS, Fig. 5.22).
The report's Algorithm 5 uses "N(kb,lb)" which is identical to NNS's
4-connected neighbourhood.

Extensions beyond Algorithm 5 (kept as configurable):
  - Aspiration (Glover 1989): if a tabu entry's observed magnitude exceeds
    the current global best, admit it regardless of tabu status.
  - Diversification: periodic random jump to escape long-running local regions.

References:
  Kristmundsson & Syberg (2018), WCS10-951, Section 5.4.5, Algorithm 5,
  Figures 5.23 (pp. 67-68).
"""

from __future__ import annotations

import numpy as np

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState

_4CONNECTED = [(0, 1), (0, -1), (1, 0), (-1, 0)]


class Tabu(Algorithm):
    """Tabu search (Algorithm 5, thesis Sec. 5.4.5).

    Parameters
    ----------
    tenure : int
        Tabu tenure s: occasions a chosen pair stays tabu.  Default 20.
    age_weight : float
        Weight [0,1] for staleness in non-tabu scoring (extension).
    diversification_period : int
        Periodic random jump every D calls (0 = disabled).  Default 50.
    """

    name = "tabu"

    def __init__(self,
                 tenure: int = 20,
                 radius: int = 1,          # kept for API compatibility; report uses 4-conn
                 age_weight: float = 0.5,
                 diversification_period: int = 50):
        self.tenure = tenure
        self.radius = radius               # ignored internally — always 4-connected
        self.age_weight = float(age_weight)
        self.diversification_period = diversification_period

    # ------------------------------------------------------------------
    # Algorithm interface
    # ------------------------------------------------------------------

    def reset(self, state: BPLMState, context: dict) -> None:
        K, L = state.K, state.L
        # Algorithm 5 line 1: T <- zeros(K,L)
        self._T: np.ndarray = np.zeros((K, L), dtype=np.int64)
        self._global_best_mag: float = 0.0
        self._call_count: int = 0
        # Initialise with random seed (Algorithm 5 line 1: kb, lb <- Random)
        rng = np.random.default_rng()
        self._kb: int = int(rng.integers(0, K))
        self._lb: int = int(rng.integers(0, L))
        self._xi: float = 0.0
        self._stack: list[tuple[int, int]] = []
        self._initial: bool = True

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        self._call_count += 1

        # Algorithm 5 line 16: T <- T + 1, T(T > 0) <- 0
        self._T[self._T < 0] += 1

        K, L = state.K, state.L
        obs_mag = np.abs(state.observations)

        # Update global best (for aspiration extension)
        measured_mask = state.measured_at >= 0
        if measured_mask.any():
            measured_max = float(obs_mag[measured_mask].max())
            if measured_max > self._global_best_mag:
                self._global_best_mag = measured_max

        # Cold start: return seed before any measurements
        if not np.any(measured_mask):
            if self._initial:
                self._initial = False
                choice = (self._kb, self._lb)
                self._T[choice] = -self.tenure
                return choice
            choice = (self._kb, self._lb)
            self._T[choice] = -self.tenure
            return choice

        # --- Aspiration extension (Glover 1989) ---
        # Check if any tabu entry beats the current global best observed magnitude.
        aspiration_threshold = self._global_best_mag
        tabu_ks, tabu_ls = np.where(self._T < 0)
        if tabu_ks.size > 0:
            tabu_mags = obs_mag[tabu_ks, tabu_ls]
            best_tabu_idx = int(np.argmax(tabu_mags))
            if tabu_mags[best_tabu_idx] > aspiration_threshold:
                choice = (int(tabu_ks[best_tabu_idx]), int(tabu_ls[best_tabu_idx]))
                self._global_best_mag = float(tabu_mags[best_tabu_idx])
                self._T[choice] = -self.tenure
                return choice

        # Algorithm 5 lines 8-11: if Y[k,l] > xi, update centre
        ok, ol = state.obp()
        obp_mag = float(obs_mag[ok, ol])
        if obp_mag > self._xi:
            self._kb, self._lb = ok, ol
            self._xi = obp_mag
            self._stack.clear()

        # Algorithm 5 lines 4-7: if P empty, rebuild N(kb, lb) and reset xi
        if not self._stack:
            self._rebuild_stack(state)
            self._xi = 0.0

        # Algorithm 5 lines 12-15: delete measured entry from top of P
        if self._stack:
            top = self._stack[-1]
            if top == (ok, ol):
                self._stack.pop()
                if not self._stack:
                    self._rebuild_stack(state)
                    self._xi = 0.0

        # Read next candidate from top of stack (without deleting)
        if self._stack:
            k, l = self._stack[-1]
        else:
            k, l = self._kb, self._lb

        # Algorithm 5 lines 17-19: if tabu, find closest non-tabu to OBP
        if self._T[k, l] < 0:
            k, l = self._closest_non_tabu(self._kb, self._lb, K, L)

        # --- Diversification extension (Glover 1989) ---
        if (self.diversification_period > 0
                and self._call_count % self.diversification_period == 0):
            rng = np.random.default_rng()
            free = list(zip(*np.where(self._T >= 0)))
            if free:
                idx = rng.integers(len(free))
                k, l = int(free[idx][0]), int(free[idx][1])

        choice = (k, l)
        # Algorithm 5 line 20: T[k,l] <- -s
        self._T[choice] = -self.tenure
        return choice

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _neighbourhood(self, ck: int, cl: int, K: int, L: int) -> list[tuple[int, int]]:
        """4-connected neighbourhood (Algorithm 5 / Fig. 5.22 pattern A)."""
        result = []
        for dk, dl in _4CONNECTED:
            nk, nl = ck + dk, cl + dl
            if 0 <= nk < K and 0 <= nl < L:
                result.append((nk, nl))
        return result

    def _rebuild_stack(self, state: BPLMState) -> None:
        neighbours = self._neighbourhood(self._kb, self._lb, state.K, state.L)
        self._stack = list(neighbours)

    def _closest_non_tabu(self, ck: int, cl: int, K: int, L: int) -> tuple[int, int]:
        """argmin_{T=0} ||[k_hat,l_hat] - [k,l]|| — Algorithm 5 line 18."""
        kk, ll = np.where(self._T >= 0)
        if kk.size == 0:
            # Entire codebook tabu — reset and return centre
            self._T[:] = 0
            return ck, cl
        dists = np.sqrt((kk - ck) ** 2 + (ll - cl) ** 2)
        best = int(np.argmin(dists))
        return int(kk[best]), int(ll[best])
