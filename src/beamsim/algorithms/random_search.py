"""Random-permutation BPLM scan baseline.

Mirrors the predecessor MATLAB simulator's ``updateRand.m``: at the
start of each scan cycle (or whenever every cell has been visited)
draw a fresh ``randperm`` of all ``K * L`` beam pairs, then march
through it in order. Once the permutation is exhausted, draw a new one.

The RNG is seeded from ``context["trial_seed"]`` so multi-trial runs
are reproducible.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState


class RandomSearch(Algorithm):
    """Random-permutation BPLM scan with refresh on exhaustion."""

    name = "random"

    def reset(self, state: BPLMState, context: dict) -> None:
        self._rng = np.random.default_rng(int(context.get("trial_seed", 0)))
        self._order: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self._cursor = 0
        self._refill(state)

    def _refill(self, state: BPLMState) -> None:
        self._order = self._rng.permutation(state.K * state.L)
        self._cursor = 0

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        if self._cursor >= self._order.size:
            self._refill(state)
        flat = int(self._order[self._cursor])
        self._cursor += 1
        return flat // state.L, flat % state.L
