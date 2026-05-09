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
  C_tabu  -- tabu matrix T directly (Eq. 5.27): entries are negative (tabu) or
              zero (free).  argmax(R) naturally avoids tabu pairs since their
              contribution is negative, and rewards free entries.
  C_nns   -- binary NNS P-list matrix (Eq. 5.28):
              C_nns(i,j) = 1 if (i,j) in P (the internal NNS stack), 0 otherwise.

Criterion ordering in R (thesis Fig. 5.26, Equation 5.29):
  R = w[0]*C_age + w[1]*C_tabu + w[2]*C_nns

Endpoint weight vectors from Fig. 5.26 (pie-chart percentages):
  w_low  = (0.43, 0.52, 0.05)  -- 3 m/s:  age-heavy (beam acquisition)
  w_high = (0.16, 0.36, 0.49)  -- 10 m/s: NNS-heavy  (tracking)
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
W_LOW = np.array([0.43, 0.52, 0.05])  # (age, tabu, NNS) at 3 m/s
W_HIGH = np.array([0.16, 0.36, 0.49])  # (age, tabu, NNS) at 10 m/s

_4CONNECTED = [(0, 1), (0, -1), (1, 0), (-1, 0)]


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
        Chebyshev radius controlling the NNS neighbourhood for the P-list.
        (kept for API compatibility; internal NNS uses 4-connected, radius=1)
    tabu_tenure : int
        Tabu tenure s: how many occasions a chosen pair stays tabu.
    w_low : array-like, shape (3,)
        Weight vector for low-mobility endpoint, ordered (age, tabu, NNS).
    w_high : array-like, shape (3,)
        Weight vector for high-mobility endpoint, ordered (age, tabu, NNS).
    """

    name = "mcmd"

    def __init__(
        self,
        q: int = 10,
        c_v: float = 30.0,  # tuned so v saturates in fast rotation (Sec 5.5)
        c_b: float = 1.5,  # tuned so BQ saturates with healthy OBP magnitudes
        nns_radius: int = 2,
        tabu_tenure: int = 20,
        w_low: NDArray[np.float64] = W_LOW,
        w_high: NDArray[np.float64] = W_HIGH,
    ):
        self.q = q
        self.c_v = c_v
        self.c_b = c_b
        self.nns_radius = nns_radius
        self.tabu_tenure = tabu_tenure
        self.w_low = np.asarray(w_low, dtype=float)
        self.w_high = np.asarray(w_high, dtype=float)

    def reset(self, state: BPLMState, context: dict) -> None:
        K, L = state.K, state.L
        # Tabu matrix: 0 = free, negative = tabu (Algorithm 5 / Eq. 5.27).
        self._T: NDArray[np.int64] = np.zeros((K, L), dtype=np.int64)
        # Internal NNS state for the binary C_nns P-list (Eq. 5.28).
        self._nns_kb: int = 0
        self._nns_lb: int = 0
        self._nns_xi: float = 0.0
        self._nns_stack: list[tuple[int, int]] = []
        # Window for volatility and beam-quality estimates.
        self._snapshots: deque[NDArray[np.complex128]] = deque(maxlen=self.q + 2)
        self._bq_history: deque[float] = deque(maxlen=self.q + 1)

    # ------------------------------------------------------------------
    # Core selection
    # ------------------------------------------------------------------

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        K, L = state.K, state.L
        self._update_history(state)
        # Advance tabu counters (Algorithm 5 line 16: T <- T+1, T(T>0) <- 0)
        self._T[self._T < 0] += 1

        ck, cl = state.obp() if state.measured_at.max() >= 0 else (K // 2, L // 2)

        # --- Criterion 1: C_age (Eq. 5.25) ---
        C_age = state.age_matrix(current_m=m)

        # --- Criterion 2: C_tabu (Eq. 5.27) ---
        # C_tabu = T directly: free entries = 0, tabu entries < 0.
        # argmax(R) naturally avoids tabu pairs (negative contribution).
        C_tabu = self._T.astype(float)

        # --- Criterion 3: C_nns binary P-list (Eq. 5.28) ---
        # Update internal NNS P-list, then set C_nns[k,l] = 1 if (k,l) in P.
        self._update_nns(state, ck, cl)
        C_nns = np.zeros((K, L), dtype=float)
        for pk, pl in self._nns_stack:
            C_nns[pk, pl] = 1.0

        # --- Adaptive weight (Eqs. 5.34, 5.35) ---
        v = self._volatility()
        bq = self._beam_quality()
        w_t = float(np.clip(bq * (bq + v) / 2.0, 0.0, 1.0))
        weights = self.w_low + (self.w_high - self.w_low) * w_t

        # --- Aggregate priority matrix R (Eq. 5.29) ---
        R = weights[0] * C_age + weights[1] * C_tabu + weights[2] * C_nns

        flat = int(np.argmax(R))
        choice = (flat // L, flat % L)
        # Mark chosen pair tabu (Algorithm 5 line 20: T[k,l] <- -s)
        self._T[choice] = -self.tabu_tenure
        return choice

    # ------------------------------------------------------------------
    # Internal NNS P-list (binary C_nns per Eq. 5.28)
    # ------------------------------------------------------------------

    def _update_nns(self, state: BPLMState, ck: int, cl: int) -> None:
        """Maintain internal NNS P-list so C_nns = 1 iff (k,l) in P.

        Mirrors the steepest-ascent semantics of the standalone NNS:
        compare the (k,l) pair directly under the NNS centre against the
        passed-in OBP (ck, cl) — when OBP is a stronger pair than the
        currently-stored centre, relocate. Avoids resetting xi to 0 on
        rebuild, which would let any arbitrary pair pull the centre.
        """
        centre_mag = float(np.abs(state.observations[self._nns_kb, self._nns_lb]))
        obp_mag = float(np.abs(state.observations[ck, cl]))
        if obp_mag > centre_mag and obp_mag > self._nns_xi:
            self._nns_kb = ck
            self._nns_lb = cl
            self._nns_xi = obp_mag
            self._nns_stack = []

        # Rebuild neighbour list around the (potentially new) centre.
        if not self._nns_stack:
            self._nns_stack = self._nns_neighbours(state)

    def _nns_neighbours(self, state: BPLMState) -> list[tuple[int, int]]:
        K, L = state.K, state.L
        result = []
        for dk, dl in _4CONNECTED:
            nk = self._nns_kb + dk
            nl = self._nns_lb + dl
            if 0 <= nk < K and 0 <= nl < L:
                result.append((nk, nl))
        return result

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
