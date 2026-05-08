"""Multi-Criteria Measurement Decision (MCMD) tracker.

Combines three criteria — age (exhaustive/spatial-survey),
tabu (forced exploration away from recent measurements), and
NNS (local refinement around the OBP) — into a single priority
matrix R(m) = sum_i w_i(m) C^(i)(m).  The weights interpolate
between two endpoint vectors via a tracking-priority scalar
w_t(m) = BQ(m) * (BQ(m) + v(m)) / 2 derived from observed beam
quality and channel volatility (Equations 5.34 and 5.35 in the thesis).

Criterion matrices (Section 5.5.1, Equations 5.25-5.28):
  C_age   -- age matrix: entry = m - measured_at(k,l), rewarding stale pairs
              (Eq. 5.25; implements exhaustive search on its own).
  C_tabu  -- tabu matrix T: entries count down from -s (tabu) to 0 (free),
              shifted to non-negative so argmax selects non-tabu pairs.
              Entries in the exploration annulus around the OBP are set to
              a positive reward (Eq. 5.27/5.28 — tracks the NNS P-list).
  C_nns   -- Gaussian-like bump centred on the OBP:
              C_nns(k,l) = exp(-||[k,l]-[ck,cl]||_inf / nns_radius)
              (Eq. 5.28 — reward entries near the current best beam).

Criterion ordering in R (thesis Fig. 5.26, Equation 5.29):
  R = w[0]*C_age + w[1]*C_tabu + w[2]*C_nns

Endpoint weight vectors from Fig. 5.26 (pie-chart percentages):
  w_low  = (0.43, 0.52, 0.05)  — 3 m/s:  age-heavy (beam acquisition)
  w_high = (0.16, 0.36, 0.49)  — 10 m/s: NNS-heavy  (tracking)
                                           ordered (age, tabu, NNS)

Reference:
  Kristmundsson & Syberg (2018), "Beam alignment methods for terminals in
  millimeter-wave wireless networks", Aalborg University MSc thesis
  WCS10-951, Section 5.5, Algorithms 7-8, Figures 5.26-5.29,
  Equations 5.25-5.35 (pp. 70-77).
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState


# Weight vectors from Fig. 5.26 — order is (age, tabu, NNS)
W_LOW = np.array([0.43, 0.52, 0.05])    # (age, tabu, NNS) at 3 m/s
W_HIGH = np.array([0.16, 0.36, 0.49])   # (age, tabu, NNS) at 10 m/s


class MCMD(Algorithm):
    """Multi-Criteria Measurement Decision tracker.

    Parameters
    ----------
    q : int
        History length for volatility and beam-quality moving averages.
    c_v : float
        Scaling constant for volatility estimate (Eq. 5.32).
    c_b : float
        Scaling constant for beam-quality estimate (Eq. 5.33).
    nns_radius : int
        Chebyshev radius controlling the NNS Gaussian bump width.
    tabu_tenure : int
        Tabu tenure s: how many occasions a chosen pair stays tabu.
    w_low : array-like, shape (3,)
        Weight vector for low-mobility endpoint, ordered (age, tabu, NNS).
    w_high : array-like, shape (3,)
        Weight vector for high-mobility endpoint, ordered (age, tabu, NNS).
    """

    name = "mcmd"

    def __init__(self,
                 q: int = 10,
                 c_v: float = 1.0,
                 c_b: float = 1.0,
                 nns_radius: int = 2,
                 tabu_tenure: int = 8,
                 w_low: NDArray[np.float64] = W_LOW,
                 w_high: NDArray[np.float64] = W_HIGH):
        self.q = q
        self.c_v = c_v
        self.c_b = c_b
        self.nns_radius = nns_radius
        self.tabu_tenure = tabu_tenure
        self.w_low = np.asarray(w_low, dtype=float)
        self.w_high = np.asarray(w_high, dtype=float)

    def reset(self, state: BPLMState, context: dict) -> None:
        K, L = state.K, state.L
        # Proper tabu matrix (thesis Algorithm 5, T initialised to zeros(K,L))
        # Negative entries = tabu; 0 = free.
        self._T: NDArray[np.int64] = np.zeros((K, L), dtype=np.int64)
        # Window length q+2 in snapshots gives q+1 consecutive-pair diffs,
        # so np.mean() over the diff list divides by q+1 = report's (l+1)
        # in Eq. 5.32. Matches the BQ window in Eq. 5.33 the same way.
        self._snapshots: deque[NDArray[np.complex128]] = deque(maxlen=self.q + 2)
        self._bq_history: deque[float] = deque(maxlen=self.q + 1)

    # ------------------------------------------------------------------
    # Core selection
    # ------------------------------------------------------------------

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        K, L = state.K, state.L
        self._update_history(state)
        # Advance tabu counters one step (negative -> toward 0)
        self._T[self._T < 0] += 1

        ck, cl = state.obp() if state.measured_at.max() >= 0 else (K // 2, L // 2)

        # --- Criterion 1: C_age (Eq. 5.25) ---
        # Rewards stale entries: value = age in occasions.
        # Never-measured entries get age = m + 1 (maximally stale).
        C_age = state.age_matrix(current_m=m)          # shape (K, L), not normalised

        # --- Criterion 2: C_tabu (Eq. 5.27/5.28) ---
        # The tabu matrix T: tabu entries have T < 0.
        # We shift to: C_tabu(k,l) = T(k,l) + tabu_tenure  so free entries = tabu_tenure
        # and tabu entries < tabu_tenure.  This means argmax(R) will avoid tabu pairs
        # naturally whenever age and NNS do not overwhelm the difference.
        # Additionally, entries in the NNS exploration annulus get an extra bump.
        C_tabu = (self._T + self.tabu_tenure).astype(float)
        kk, ll = np.indices((K, L))
        dist = np.maximum(np.abs(kk - ck), np.abs(ll - cl))
        exploration_ring = ((dist >= 1) & (dist <= self.nns_radius + 1)).astype(float)
        C_tabu += exploration_ring * float(self.tabu_tenure)  # extra reward in the ring

        # --- Criterion 3: C_nns (Eq. 5.28) ---
        # Gaussian-like bump centred on OBP; maximum at OBP = 1, decays with Chebyshev dist.
        C_nns = np.exp(-dist.astype(float) / max(self.nns_radius, 1))

        # --- Adaptive weight (Eqs. 5.34, 5.35) ---
        v = self._volatility()
        bq = self._beam_quality()
        w_t = float(np.clip(bq * (bq + v) / 2.0, 0.0, 1.0))
        # Interpolate: w_out = w_low + (w_high - w_low) * w_t  (Eq. 5.35)
        weights = self.w_low + (self.w_high - self.w_low) * w_t

        # --- Aggregate priority matrix R (Eq. 5.29) ---
        # Order: R = w[0]*C_age + w[1]*C_tabu + w[2]*C_nns  (matches Fig. 5.26)
        R = weights[0] * C_age + weights[1] * C_tabu + weights[2] * C_nns

        flat = int(np.argmax(R))
        choice = (flat // L, flat % L)
        # Mark chosen pair tabu for tenure occasions
        self._T[choice] = -self.tabu_tenure
        return choice

    # ------------------------------------------------------------------
    # State tracking helpers
    # ------------------------------------------------------------------

    def _update_history(self, state: BPLMState) -> None:
        self._snapshots.append(state.snapshot())
        if state.measured_at.max() >= 0:
            self._bq_history.append(float(np.abs(state.obp_value())))

    def _volatility(self) -> float:
        """Channel volatility v(m): moving-average of max |Y(m)-Y(m-1)| (Eq. 5.32)."""
        if len(self._snapshots) < 2:
            return 0.0
        diffs = []
        snaps = list(self._snapshots)
        for a, b in zip(snaps[:-1], snaps[1:]):
            diffs.append(float(np.max(np.abs(b - a))))
        v = self.c_v * float(np.mean(diffs))
        return float(np.clip(v, 0.0, 1.0))

    def _beam_quality(self) -> float:
        """Beam quality BQ(m): moving-average of |Y_OBP| (Eq. 5.33)."""
        if not self._bq_history:
            return 0.0
        bq = self.c_b * float(np.mean(self._bq_history))
        return float(np.clip(bq, 0.0, 1.0))
