"""Tabu search around the OBP — Glover (1989) classical formulation.

Implements the Tabu Search beam-alignment algorithm described in the
predecessor MSc thesis (Kristmundsson & Syberg, 2018), Section 5.4.5,
Algorithm 5 (pp. 67-68), extended with Glover's classical tabu-search
aspiration criterion and a diversification restart.

Algorithm outline (Algorithm 5 in the thesis):
  - Maintain a tabu matrix T of size K x L.  T(k,l) < 0 means (k,l) is tabu.
  - Each call, push N(OBP) into a measurement queue P (same as NNS).
  - Pop the next (k,l) from P.  If tabu (T<0), skip to the closest non-tabu
    pair to the OBP (line 18 of Algorithm 5).
  - After measuring, decrement T everywhere, reset T(k,l) to -s for the
    chosen pair (s = tenure length), making it tabu for s occasions.

Upgrades beyond the thesis Algorithm 5:
  1. Aspiration (Glover 1989): if a tabu candidate's *observed* magnitude
     already exceeds the current global best observed magnitude, take it
     regardless of tabu status.  This prevents discarding a pair that is
     provably better than anything seen so far.
  2. Diversification: every D steps (default D=50), jump to a uniformly
     random pair that is not tabu.  This escapes long-running local regions
     per Glover (1989) Section 4.

Scoring of non-tabu candidates combines:
  - Staleness (age = m - measured_at): prefer older measurements (more
    information gain).
  - Proximity to OBP: prefer closer pairs (hill-climbing spirit of NNS).
  Score = age_weight * age_norm + (1 - age_weight) * proximity_norm.

References:
  Glover, F. (1989). "Tabu Search — Part I". ORSA Journal on Computing,
    1(3), 190-206.
  Gao, X. et al. (2016). Tabu-search-inspired beam tracking for mmW systems
    [referenced as [42] in the thesis, Section 5.4.1].
  Kristmundsson & Syberg (2018), WCS10-951, Section 5.4.5, Algorithm 5,
    Figures 5.23 (pp. 67-68).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState


class Tabu(Algorithm):
    """Tabu search with rigorous aspiration and periodic diversification.

    Parameters
    ----------
    tenure : int
        Number of occasions a pair stays tabu after being chosen (s in
        Algorithm 5 of the thesis).  Default 8.
    radius : int
        Chebyshev radius defining the neighbourhood N(OBP).  Default 2.
    age_weight : float
        Weight [0, 1] for staleness in the non-tabu candidate scoring.
        1 - age_weight is given to proximity.  Default 0.5.
    diversification_period : int
        Every this many calls, jump to a random non-tabu pair if no
        neighbourhood candidate improves on the global best.  D ≈ 50 per
        Glover (1989) Section 4.  Set to 0 to disable.  Default 50.
    """

    name = "tabu"

    def __init__(self,
                 tenure: int = 8,
                 radius: int = 2,
                 age_weight: float = 0.5,
                 diversification_period: int = 50):
        self.tenure = tenure
        self.radius = radius
        self.age_weight = float(age_weight)
        self.diversification_period = diversification_period

    # ------------------------------------------------------------------
    # Algorithm interface
    # ------------------------------------------------------------------

    def reset(self, state: BPLMState, context: dict) -> None:
        K, L = state.K, state.L
        # Tabu matrix: 0 = not tabu; negative value counts down to expiry.
        self._T: np.ndarray = np.zeros((K, L), dtype=np.int64)
        self._global_best_mag: float = 0.0
        self._call_count: int = 0

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        self._call_count += 1

        # Advance all tabu counters (one step closer to expiry)
        self._T[self._T < 0] += 1

        K, L = state.K, state.L
        obs_mag = np.abs(state.observations)

        # Update global best from all entries that have been measured so far.
        # This must happen FIRST so that aspiration_threshold captures the true
        # best BEFORE this call's selection (Glover 1989: aspiration = "would
        # beat the incumbent if accepted NOW").
        measured_mask = state.measured_at >= 0
        if measured_mask.any():
            measured_max = float(obs_mag[measured_mask].max())
            if measured_max > self._global_best_mag:
                self._global_best_mag = measured_max

        # Without any measurements, return a round-robin seed
        if not np.any(measured_mask):
            choice = (0, 0)
            self._make_tabu(choice)
            return choice

        # Aspiration criterion (Glover 1989): a candidate is admitted if its
        # *observed* magnitude exceeds the best magnitude updated above.
        # We use the CURRENT global best as the threshold — any tabu pair that
        # already exceeds it (because the channel shifted) is accepted.
        aspiration_threshold = self._global_best_mag

        ck, cl = state.obp()

        # --- Global aspiration check (Glover 1989) ---
        # Before restricting to the neighbourhood, check if ANY tabu entry's
        # observed magnitude exceeds the pre-call global best.  This handles
        # the case where the OBP itself is tabu (it is excluded from the
        # neighbourhood, so a neighbourhood-only check would miss it).
        #
        # We scan all tabu entries and pick the one with the highest magnitude
        # that exceeds the threshold.  This is O(K*L) but K*L is small (<300).
        tabu_ks, tabu_ls = np.where(self._T < 0)
        if tabu_ks.size > 0:
            tabu_mags = obs_mag[tabu_ks, tabu_ls]
            best_tabu_idx = int(np.argmax(tabu_mags))
            if tabu_mags[best_tabu_idx] > aspiration_threshold:
                choice = (int(tabu_ks[best_tabu_idx]), int(tabu_ls[best_tabu_idx]))
                self._global_best_mag = float(tabu_mags[best_tabu_idx])
                self._make_tabu(choice)
                return choice

        # Build candidate list from square neighbourhood of OBP
        candidates = self._neighbourhood(ck, cl, K, L)
        if not candidates:
            choice = (ck, cl)
            self._make_tabu(choice)
            return choice

        # --- Non-tabu candidates ---
        non_tabu = [c for c in candidates if not self._is_tabu(c)]

        if non_tabu:
            choice = self._best_candidate(non_tabu, ck, cl, obs_mag, m, state)
        else:
            # All neighbours tabu: thesis Algorithm 5 line 18 — pick non-tabu
            # pair closest (L2) to OBP across the entire codebook.
            choice = self._closest_non_tabu(ck, cl, K, L)

        # --- Diversification (Glover 1989 Section 4) ---
        if (self.diversification_period > 0
                and self._call_count % self.diversification_period == 0):
            rng = np.random.default_rng()
            free = list(zip(*np.where(self._T >= 0)))
            if free:
                idx = rng.integers(len(free))
                choice = free[idx]

        self._make_tabu(choice)
        return choice

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _neighbourhood(self, ck: int, cl: int, K: int, L: int) -> list[tuple[int, int]]:
        result = []
        for dk in range(-self.radius, self.radius + 1):
            for dl in range(-self.radius, self.radius + 1):
                if dk == 0 and dl == 0:
                    continue
                nk, nl = ck + dk, cl + dl
                if 0 <= nk < K and 0 <= nl < L:
                    result.append((nk, nl))
        return result

    def _is_tabu(self, pair: tuple[int, int]) -> bool:
        return bool(self._T[pair] < 0)

    def _make_tabu(self, pair: tuple[int, int]) -> None:
        self._T[pair] = -self.tenure

    def _best_candidate(self,
                        pool: list[tuple[int, int]],
                        ck: int, cl: int,
                        obs_mag: np.ndarray,
                        m: int,
                        state: BPLMState) -> tuple[int, int]:
        """Score candidates by staleness + proximity, pick highest score."""
        ages = np.array([
            (m - state.measured_at[p]) if state.measured_at[p] >= 0 else (m + 1)
            for p in pool
        ], dtype=float)
        dists = np.array([
            np.sqrt((p[0] - ck) ** 2 + (p[1] - cl) ** 2)
            for p in pool
        ], dtype=float)

        age_max = ages.max() + 1e-12
        dist_max = dists.max() + 1e-12

        age_norm = ages / age_max
        prox_norm = 1.0 - dists / dist_max   # closer = higher

        scores = self.age_weight * age_norm + (1.0 - self.age_weight) * prox_norm
        return pool[int(np.argmax(scores))]

    def _closest_non_tabu(self, ck: int, cl: int, K: int, L: int) -> tuple[int, int]:
        """Return the non-tabu (k, l) with smallest L2 distance to (ck, cl)."""
        kk, ll = np.where(self._T >= 0)
        if kk.size == 0:
            # Entire codebook is tabu — reset and return OBP
            self._T[:] = 0
            return ck, cl
        dists = np.sqrt((kk - ck) ** 2 + (ll - cl) ** 2)
        best = int(np.argmin(dists))
        return int(kk[best]), int(ll[best])
