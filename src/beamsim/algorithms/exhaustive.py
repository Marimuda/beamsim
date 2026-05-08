"""Exhaustive search: sequential round-robin over all (k, l) pairs."""

from __future__ import annotations

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState


class Exhaustive(Algorithm):
    name = "exhaustive"

    def reset(self, state: BPLMState, context: dict) -> None:
        self._index = 0

    def select_next_mbp(self, state, m, context):
        K, L = state.K, state.L
        idx = self._index % (K * L)
        self._index += 1
        return idx // L, idx % L
