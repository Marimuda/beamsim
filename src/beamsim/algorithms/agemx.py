"""Age-matrix scan: standalone least-recently-measured policy.

Mirrors the predecessor MATLAB simulator's ``updateAgemx.m`` used as a
*standalone* MBP policy (independent of MCMD's age criterion). At every
step the algorithm increments an age matrix and resets the just-measured
cell, then selects the cell with the largest age (the longest-stale beam
pair). When ties occur (notably at the start of a trial when every cell
has age 0) the first row-major flat index is selected, matching
``np.argmax`` semantics.

This is essentially a recency-driven exhaustive scan: the first
``K * L`` selections sweep every beam pair (in the order numpy's argmax
breaks ties), and subsequent selections cycle through the same order at
the slowest possible refresh rate.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState


class AgeMx(Algorithm):
    """Standalone age-driven least-recently-measured beam-pair scan."""

    name = "agemx"

    def reset(self, state: BPLMState, context: dict) -> None:
        self._age: NDArray[np.int64] = np.zeros((state.K, state.L), dtype=np.int64)

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        # Increment every cell, then choose the oldest. argmax picks the
        # first row-major index on ties, which gives a deterministic
        # row-major sweep on the first pass.
        self._age += 1
        flat = int(np.argmax(self._age))
        k, l = flat // state.L, flat % state.L
        # Reset the chosen cell so it ages from 0 again.
        self._age[k, l] = 0
        return k, l
