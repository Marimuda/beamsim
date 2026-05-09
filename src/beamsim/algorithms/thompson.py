"""Thompson sampling (Gaussian) multi-armed bandit beam selection.

Each (k, l) beam pair is an arm with reward = |y(k,l)|, modelled as
Gaussian with unknown mean mu and known variance sigma^2 = noise_amplitude^2.

Posterior on mu after n samples with empirical mean X_bar:
    mu | data ~ N(X_bar, sigma^2 / n).

Each call samples one mu_kl per arm and pulls the argmax (Thompson, 1933).
During cold-start every arm is pulled once so all posteriors are proper.

References:
    Thompson, W. R. (1933). "On the likelihood that one unknown probability
    exceeds another in view of the evidence of two samples." Biometrika 25.
    (Gaussian variant: Chapelle & Li, NeurIPS 2011.)

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


class ThompsonGaussian(Algorithm):
    """Gaussian Thompson sampling beam selection (Chapelle & Li, NeurIPS 2011).

    Posterior variance = sigma^2 / n where sigma = state.noise_amplitude.
    Uses a per-instance seeded Generator for reproducibility.
    """

    name = "thompson"

    def reset(self, state: BPLMState, context: dict) -> None:
        K, L = state.K, state.L
        self._mean: NDArray[np.float64] = np.zeros((K, L), dtype=np.float64)
        self._counts: NDArray[np.int_] = np.zeros((K, L), dtype=np.int_)
        self._sigma: float = state.noise_amplitude
        self._cold_index: int = 0
        self._last_kl: tuple[int, int] | None = None
        # Seeded from trial_seed when available so different trials diverge.
        # Falls back to an unseeded RNG for unit-test / ad-hoc use.
        self._rng = np.random.default_rng(context.get("trial_seed"))

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        K, L = state.K, state.L

        # Update statistics from the arm pulled last step
        if self._last_kl is not None:
            pk, pl = self._last_kl
            reward = float(np.abs(state.observations[pk, pl]))
            n = self._counts[pk, pl] + 1
            self._mean[pk, pl] += (reward - self._mean[pk, pl]) / n
            self._counts[pk, pl] = n

        # Cold-start: pull every arm once
        if self._cold_index < K * L:
            k, l = self._cold_index // L, self._cold_index % L
            self._cold_index += 1
            self._last_kl = (k, l)
            return k, l

        # Thompson sampling: sample mu_kl ~ N(mean_kl, sigma^2 / n_kl)
        std = self._sigma / np.sqrt(self._counts.astype(np.float64))
        samples = self._mean + std * self._rng.standard_normal((K, L))
        flat = int(np.argmax(samples))
        k, l = flat // L, flat % L
        self._last_kl = (k, l)
        return k, l
