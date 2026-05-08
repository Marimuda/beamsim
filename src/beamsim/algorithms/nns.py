"""Nearest-neighbour search: cycles around the OBP through 4-connected neighbours."""

from __future__ import annotations

import numpy as np

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState


class NNS(Algorithm):
    name = "nns"

    def reset(self, state: BPLMState, context: dict) -> None:
        # Visit centre + 4 neighbours in a fixed pattern (refreshes the local peak).
        self._offsets = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
        self._step = 0

    def select_next_mbp(self, state, m, context):
        ck, cl = state.obp() if np.any(state.measured_at >= 0) else (0, 0)
        dk, dl = self._offsets[self._step % len(self._offsets)]
        self._step += 1
        k = int(np.clip(ck + dk, 0, state.K - 1))
        l = int(np.clip(cl + dl, 0, state.L - 1))
        return k, l
