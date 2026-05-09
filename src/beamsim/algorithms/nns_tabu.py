"""Steepest-ascent neighbourhood search with a tabu-style relocation step.

Mirrors the predecessor MATLAB simulator's ``updateAscentmx_Tabu.m``.
This is the algorithm MCMD's slot-7 weight (``W_High[7] = 0.8742`` in
the MATLAB simulator) actually points at, distinct from plain NNS by
its **global** argmax relocation when the local probe list is exhausted.

Algorithm
---------
The policy maintains an explicit five-cell measurement *list* drawn
around a centre ``(k_b, l_b)``: the centre itself plus four cells
offset by two beam slots in each cardinal direction (with circular
wrap-around). Each step:

1. Pop the next pending cell from the list and report it as the MBP.
2. When the list is exhausted, take the **global** argmax of
   ``|Y_obs|`` across the entire BPLM, treat that cell as the new
   centre, and rebuild the five-cell list around it.

The "global argmax for relocation" is what distinguishes this variant
from :class:`beamsim.algorithms.NNS`, which only relocates *within* the
last five-cell list. In MCMD this difference matters at high mobility,
where the global view recovers from drift more quickly than the
local-only relocation.

Reference: predecessor MSc thesis Algorithm 4 with tabu-style global
relocation; MATLAB ``tracking_algos/updateAscentmx_Tabu.m``.
"""

from __future__ import annotations

import numpy as np

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState


def _five_cell_list(k: int, l: int, K: int, L: int) -> list[tuple[int, int]]:
    """Centre plus four cardinal neighbours at offset +/-2 with wrap."""
    return [
        (k, l),
        ((k - 2) % K, l),
        ((k + 2) % K, l),
        (k, (l - 2) % L),
        (k, (l + 2) % L),
    ]


class NNSTabu(Algorithm):
    """NNS-with-tabu (Ascent_Tabu): five-probe scan with global relocation."""

    name = "nns_tabu"

    def reset(self, state: BPLMState, context: dict) -> None:
        rng = np.random.default_rng(int(context.get("trial_seed", 0)))
        # Random initial centre, matching the predecessor's Algorithm 4
        # warm-up convention (a fresh tabu cycle starts here).
        k0 = int(rng.integers(state.K))
        l0 = int(rng.integers(state.L))
        self._pending: list[tuple[int, int]] = _five_cell_list(k0, l0, state.K, state.L)

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        if not self._pending:
            # All five cells of the current cycle measured: relocate via
            # GLOBAL argmax of |Y_obs| (the tabu-style step).
            obs = state.observations
            flat = int(np.argmax(np.abs(obs)))
            kb, lb = flat // state.L, flat % state.L
            self._pending = _five_cell_list(kb, lb, state.K, state.L)
        return self._pending.pop(0)
