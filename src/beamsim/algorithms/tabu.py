"""Tabu search around the OBP with aspiration override."""

from __future__ import annotations

from collections import deque

import numpy as np

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState


class Tabu(Algorithm):
    name = "tabu"

    def __init__(self, tenure: int = 8, radius: int = 2):
        self.tenure = tenure
        self.radius = radius

    def reset(self, state: BPLMState, context: dict) -> None:
        self._tabu: deque[tuple[int, int]] = deque(maxlen=self.tenure)

    def select_next_mbp(self, state, m, context):
        K, L = state.K, state.L
        if not np.any(state.measured_at >= 0):
            choice = (0, 0)
        else:
            ck, cl = state.obp()
            # Candidate neighbourhood: square of radius `radius` around OBP, excluding centre.
            candidates = []
            for dk in range(-self.radius, self.radius + 1):
                for dl in range(-self.radius, self.radius + 1):
                    if dk == 0 and dl == 0:
                        continue
                    k = ck + dk
                    l = cl + dl
                    if 0 <= k < K and 0 <= l < L:
                        candidates.append((k, l))
            if not candidates:
                choice = (ck, cl)
            else:
                # Score = current observed magnitude (encourage exploration where
                # observed is low or stale, with aspiration overriding tabu when
                # the candidate exceeds the current global maximum).
                obs_mag = np.abs(state.observations)
                global_max = obs_mag[ck, cl]
                non_tabu = [c for c in candidates if c not in self._tabu]
                pool = non_tabu if non_tabu else candidates
                # Prefer staler candidates (lower measured_at) and lower observed magnitude.
                ages = np.array([m - state.measured_at[c] if state.measured_at[c] >= 0 else m + 1 for c in pool])
                mags = np.array([obs_mag[c] for c in pool])
                # Score combines age (exploration) with raw magnitude (refinement).
                scores = ages + mags / (global_max + 1e-12) * 5.0
                # Aspiration: if any tabu candidate would beat the global max
                # by raw magnitude, take it.
                aspirated = [c for c in candidates if c in self._tabu and obs_mag[c] > global_max]
                if aspirated:
                    choice = aspirated[0]
                else:
                    choice = pool[int(np.argmax(scores))]
        self._tabu.append(choice)
        return choice
