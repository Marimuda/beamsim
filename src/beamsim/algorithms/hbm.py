"""Hierarchical Beam Management (HBM) algorithm.

Based on: Alkhateeb, El Ayach, Leus, Heath (2014) — "Channel estimation and
hybrid precoding for millimeter wave cellular systems."

Idea: a two-level codebook scan.  A coarse sub-codebook (every ``coarse_factor``-th
BS beam, giving ``L // coarse_factor`` sectors) identifies the best sector in one
sweep; a fine hill-climb (NNS-style steepest-ascent) then tracks within that
sector.  The sector scan is refreshed every ``refresh_every`` steps to track
angle drift.

Beamspace basis: UE codeword index ``k`` is cycled uniformly; BS coarse/fine
indices select columns of the 32-beam BS codebook.
"""

from __future__ import annotations

import numpy as np

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState

_MODE_COARSE = "coarse"
_MODE_FINE = "fine"


class HBM(Algorithm):
    """Hierarchical beam management: coarse sector scan then fine hill-climb.

    Parameters
    ----------
    coarse_factor:
        Sub-sample stride for the coarse codebook.  With L=32 and
        coarse_factor=4 this gives 8 coarse sectors at indices 0,4,8,...,28.
    refresh_every:
        Re-enter coarse mode every this many total steps to track angle drift.
    """

    name = "hbm"

    def __init__(self, coarse_factor: int = 4, refresh_every: int = 100) -> None:
        self._coarse_factor = coarse_factor
        self._refresh_every = refresh_every

    def reset(self, state: BPLMState, context: dict) -> None:
        L = state.bs_codebook.n_beams
        self._coarse_beams: list[int] = list(range(0, L, self._coarse_factor))
        self._n_coarse = len(self._coarse_beams)
        self._mode: str = _MODE_COARSE
        self._step: int = 0
        self._coarse_idx: int = 0  # index into _coarse_beams for current sweep
        self._best_sector: int = 0  # coarse beam index of best sector
        self._best_sector_mag: float = -np.inf
        self._fine_centre: int = 0  # fine BS beam (absolute)
        self._fine_stack: list[int] = []  # fine hill-climb stack (BS beam only)
        self._cycle_best_l: int = 0
        self._cycle_best_mag: float = -np.inf
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        # Trigger refresh: re-enter coarse mode periodically
        if self._step > 0 and self._step % self._refresh_every == 0:
            self._enter_coarse(state)

        if self._mode == _MODE_COARSE:
            return self._coarse_step(state, m)
        return self._fine_step(state, m)

    # ------------------------------------------------------------------
    def _coarse_step(self, state: BPLMState, m: int) -> tuple[int, int]:
        l = self._coarse_beams[self._coarse_idx]
        k = int(self._rng.integers(0, state.K))
        # Update best-sector from previous coarse measurement
        if self._coarse_idx > 0:
            prev_l = self._coarse_beams[self._coarse_idx - 1]
            # Use max over all UE beams: k is the *next* random draw, not the one
            # paired with prev_l.  Consistent with _enter_fine which does the same.
            mag = float(np.max(np.abs(state.observations[:, prev_l])))
            if mag > self._best_sector_mag:
                self._best_sector_mag = mag
                self._best_sector = prev_l
        self._coarse_idx += 1
        self._step += 1
        if self._coarse_idx >= self._n_coarse:
            self._enter_fine(state)
        return k, l

    def _fine_step(self, state: BPLMState, m: int) -> tuple[int, int]:
        k = int(self._rng.integers(0, state.K))
        # Update cycle-best from previous measurement at fine_centre.
        # Use max over all UE beams: k is the new random draw, not the one measured.
        prev_mag = float(np.max(np.abs(state.observations[:, self._fine_centre])))
        if prev_mag > self._cycle_best_mag:
            self._cycle_best_mag = prev_mag
            self._cycle_best_l = self._fine_centre

        if not self._fine_stack:
            # End of neighbourhood cycle: steepest-ascent centre relocation.
            # Use max over all UE beams for same reason as prev_mag above.
            centre_mag = float(np.max(np.abs(state.observations[:, self._fine_centre])))
            if self._cycle_best_mag > centre_mag:
                self._fine_centre = self._cycle_best_l
            self._cycle_best_mag = -np.inf
            self._rebuild_fine_stack(state)

        l = self._fine_stack.pop()
        self._step += 1
        return k, l

    # ------------------------------------------------------------------
    def _enter_coarse(self, state: BPLMState) -> None:
        self._mode = _MODE_COARSE
        self._coarse_idx = 0
        self._best_sector_mag = -np.inf

    def _enter_fine(self, state: BPLMState) -> None:
        """Transition to fine mode: pick best sector, seed hill-climb centre."""
        # Check last coarse beam observation
        last_l = self._coarse_beams[self._n_coarse - 1]
        mag = float(np.max(np.abs(state.observations[:, last_l])))
        if mag > self._best_sector_mag:
            self._best_sector_mag = mag
            self._best_sector = last_l
        self._fine_centre = self._best_sector
        self._cycle_best_mag = -np.inf
        self._cycle_best_l = self._fine_centre
        self._fine_stack = []
        self._rebuild_fine_stack(state)
        self._mode = _MODE_FINE

    def _rebuild_fine_stack(self, state: BPLMState) -> None:
        """Fine neighbours: BS beams within ±coarse_factor of centre."""
        L = state.bs_codebook.n_beams
        stride = self._coarse_factor
        neighbours = [
            self._fine_centre + dl
            for dl in range(-stride, stride + 1)
            if dl != 0 and 0 <= self._fine_centre + dl < L
        ]
        self._fine_stack = list(set(neighbours))
