"""Angular prediction: track the OBP indices over time and extrapolate."""

from __future__ import annotations

from collections import deque

import numpy as np

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState


class AngularPrediction(Algorithm):
    """Constant-velocity predictor on the OBP-index trajectory.

    The OBP at occasion m gives a (k, l) sample. We fit a slope over a
    short history and predict the next (k, l). Robust to slow motion;
    degrades when the OBP jumps because of a multipath swap.
    """

    name = "angular_prediction"

    def __init__(self, history: int = 5):
        self.history = history

    def reset(self, state: BPLMState, context: dict) -> None:
        self._history: deque[tuple[int, int]] = deque(maxlen=self.history)

    def select_next_mbp(self, state, m, context):
        if np.any(state.measured_at >= 0):
            self._history.append(state.obp())
        if len(self._history) < 2:
            # Cold start: fall back to a sweep
            idx = m % (state.K * state.L)
            return idx // state.L, idx % state.L
        ks = np.array([h[0] for h in self._history], dtype=float)
        ls = np.array([h[1] for h in self._history], dtype=float)
        # Linear fit
        t = np.arange(len(self._history))
        if t.std() > 0:
            slope_k = np.polyfit(t, ks, 1)[0]
            slope_l = np.polyfit(t, ls, 1)[0]
        else:
            slope_k, slope_l = 0.0, 0.0
        pred_k = int(np.round(ks[-1] + slope_k))
        pred_l = int(np.round(ls[-1] + slope_l))
        return int(np.clip(pred_k, 0, state.K - 1)), int(np.clip(pred_l, 0, state.L - 1))
