"""UCB1 multi-armed bandit beam selection.

Each (k, l) beam pair is an arm; the reward is the post-beamforming SNR
magnitude |y(k,l)|.  UCB1 balances exploration/exploitation via an upper
confidence bound on the empirical mean.

Algorithm: score(k,l) = X_bar(k,l) + sqrt(2 * ln(t) / N(k,l)).
Select argmax.  During cold-start (t < K*L) each arm is pulled exactly once
before the UCB rule applies.

References:
    Auer, Cesa-Bianchi, Fischer (2002). "Finite-time analysis of the
    multiarmed bandit problem." Machine Learning 47, 235-256.

Domain refs:
    Hashemi et al. (2018). "Efficient beam alignment in mmWave systems using
    contextual bandits." IEEE Trans. Wireless Commun.
    Va et al. "Online learning for position-aided millimeter wave beam training."
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState


class UCB1(Algorithm):
    """UCB1 bandit beam selection (Auer et al. 2002).

    Tracks running per-arm means and pull counts internally since the BPLM
    only retains the most recent observation per arm.
    """

    name = "ucb1"

    def reset(self, state: BPLMState, context: dict) -> None:
        K, L = state.K, state.L
        self._mean: NDArray[np.float64] = np.zeros((K, L), dtype=np.float64)
        self._counts: NDArray[np.int_] = np.zeros((K, L), dtype=np.int_)
        self._t: int = 0
        self._cold_index: int = 0  # next arm to pull in cold-start
        self._last_kl: tuple[int, int] | None = None

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        K, L = state.K, state.L

        # Update statistics from the arm pulled last step
        if self._last_kl is not None:
            pk, pl = self._last_kl
            reward = float(np.abs(state.observations[pk, pl]))
            n = self._counts[pk, pl] + 1
            self._mean[pk, pl] += (reward - self._mean[pk, pl]) / n
            self._counts[pk, pl] = n
            self._t += 1

        # Cold-start: pull every arm once
        if self._cold_index < K * L:
            k, l = self._cold_index // L, self._cold_index % L
            self._cold_index += 1
            self._last_kl = (k, l)
            return k, l

        # UCB1 rule: argmax mean + sqrt(2 * ln(t) / N)
        ln_t = np.log(float(self._t))
        bonus = np.sqrt(2.0 * ln_t / self._counts)  # counts all >= 1 here
        scores = self._mean + bonus
        flat = int(np.argmax(scores))
        k, l = flat // L, flat % L
        self._last_kl = (k, l)
        return k, l
