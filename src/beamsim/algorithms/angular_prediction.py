"""Angular prediction: gradient-sum AoA/AoD tracker.

Implements Algorithm 3 from the predecessor MSc thesis
(Kristmundsson & Syberg, 2018), Section 5.4.3 (p. 63), faithfully.

Algorithm 3 (thesis, p. 63):
  Line 1: h(m) <- a(k_hat, l_hat)
  Line 2: (k,l) <- cb( h(m) + (1/z) * sum_{i=1}^{z} F(i) * (h(m-i+1) - h(m-i)) )

where:
  h      is the angular history: a 2-vector [theta_AoA, theta_AoD] read from
         the codebook angles of the current OBP.
  F(i)   weighting kernel over the z most recent gradient steps.
  z      normalisation value: z = sum_i F(i)  (Eq. 5.23).
  a()    translates codebook index to angle (codebook.theta[idx]).
  cb()   translates angle to codebook index (nearest-beam lookup).

F(i) choice: the report says F "should be designed depending on the desired
aggressiveness". With no further specification we use uniform weights
F(i) = 1 for all i, giving z = history_len (simple moving-gradient average).
This is the simplest faithful implementation of the report's predictor.

Cold start: when fewer than 2 OBP samples have been collected (not enough to
form a gradient), the algorithm defaults to the current OBP (Algorithm 3
lines 1-2 collapse to cb(h(m)) = OBP itself). During this phase the algorithm
uses exhaustive round-robin to accumulate measurements so that an OBP is
established quickly.

Reference:
  Kristmundsson & Syberg (2018), "Beam alignment methods for terminals in
  millimeter-wave wireless networks", Aalborg University MSc thesis WCS10-951,
  Section 5.4.3, Algorithm 3, Figure 5.19 (pp. 63-64).
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState


class AngularPrediction(Algorithm):
    """Gradient-sum AoA/AoD predictor (Algorithm 3, thesis Sec. 5.4.3).

    Parameters
    ----------
    history_len : int
        Number of past gradient steps to accumulate (z in Eq. 5.23).
        Default 3 (three-step gradient window).
    warmup : int
        Minimum number of OBP samples before activating prediction.
        During warmup the algorithm uses exhaustive round-robin.
        Default 2 (need at least 2 samples to form one gradient step).
    """

    name = "angular_prediction"

    def __init__(self, warmup: int = 2, history_len: int = 3,
                 # legacy Kalman params accepted and ignored for API compat
                 q_angle: float = 1e-3, q_rate: float = 1e-2,
                 sigma_obs_factor: float = 1.0):
        self.warmup = max(warmup, 2)
        self.history_len = max(history_len, 1)

    # ------------------------------------------------------------------
    # Algorithm interface
    # ------------------------------------------------------------------

    def reset(self, state: BPLMState, context: dict) -> None:
        # h_history stores the last (history_len + 1) OBP angle vectors
        # so we can compute history_len gradient steps.
        self._h_history: deque[NDArray[np.float64]] = deque(
            maxlen=self.history_len + 1
        )
        self._obp_count: int = 0
        self._sweep_idx: int = 0

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        # Algorithm 3 line 1: h(m) <- a(k_hat, l_hat)
        if np.any(state.measured_at >= 0):
            ok, ol = state.obp()
            h_current = np.array([
                float(state.ue_codebook.theta[ok]),
                float(state.bs_codebook.theta[ol])
            ])
            # Record if this is a new OBP angle (deduplicate consecutive repeats)
            if (not self._h_history
                    or not np.allclose(h_current, self._h_history[-1], atol=1e-9)):
                self._h_history.append(h_current)
                self._obp_count += 1

        # Cold start: not enough history to form a gradient
        if self._obp_count < self.warmup:
            return self._cold_start_step(state)

        # Algorithm 3 line 2: predict next angle
        h_pred = self._predict()
        k = _nearest_beam(h_pred[0], state.ue_codebook.theta)
        l = _nearest_beam(h_pred[1], state.bs_codebook.theta)
        return k, l

    # ------------------------------------------------------------------
    # Gradient-sum predictor
    # ------------------------------------------------------------------

    def _predict(self) -> NDArray[np.float64]:
        """Compute h(m) + (1/z) * sum_{i=1}^{z} F(i) * (h(m-i+1) - h(m-i)).

        With uniform weights F(i) = 1, z = min(history_len, len(history) - 1).
        """
        hist = list(self._h_history)
        # Compute available gradient steps: h(t) - h(t-1) for the last z steps
        n_grad = min(self.history_len, len(hist) - 1)
        if n_grad <= 0:
            return hist[-1].copy()

        grads = np.array([hist[-(i)] - hist[-(i + 1)] for i in range(1, n_grad + 1)])
        # F(i) = 1 (uniform), z = n_grad
        gradient_sum = np.sum(grads, axis=0)
        h_current = hist[-1]
        h_pred = h_current + gradient_sum / n_grad
        return h_pred

    def _cold_start_step(self, state: BPLMState) -> tuple[int, int]:
        """Round-robin sweep during warmup phase."""
        total = state.K * state.L
        idx = self._sweep_idx % total
        self._sweep_idx += 1
        return idx // state.L, idx % state.L


def _nearest_beam(angle: float, theta: NDArray[np.float64]) -> int:
    """Index of the codebook beam whose steering angle is closest to `angle`."""
    diffs = np.abs(_wrap_pi(theta - angle))
    return int(np.argmin(diffs))


def _wrap_pi(a: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
