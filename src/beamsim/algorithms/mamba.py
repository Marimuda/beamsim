"""MAMBA: Multi-Armed Bandit framework for beam tracking with adaptive
Thompson sampling and beam-neighbourhood correlation.

Reference:
    Aykin, I., Akgun, B., Feng, M., Krunz, M. (2020).
    "MAMBA: A Multi-armed Bandit Framework for Beam Tracking in Millimeter-
    wave Systems." IEEE INFOCOM 2020, pp. 1469-1478.
    DOI: 10.1109/INFOCOM41043.2020.9155408
    Journal extension: Krunz, M., Aykin, I., Sarkar, S., Akgun, B. (2024).
    "Online Reinforcement Learning for Beam Tracking and Rate Adaptation in
    Millimeter-Wave Systems." IEEE TMC 23(2), 1830-1845.
    DOI: 10.1109/TMC.2023.3243910

Key differences from our stationary ThompsonGaussian (``thompson.py``):

  1. Discounted update.  Per-arm running mean is *discounted* by a factor
     ``gamma ∈ (0, 1)`` per step, so older measurements decay.  This
     handles non-stationary rewards (channel ageing, UE mobility) that
     break the stationary-arm assumption of vanilla Thompson sampling.
  2. Neighbourhood-correlated exploration.  When the current best arm's
     observed reward drops by more than a threshold below its running
     mean, we restrict Thompson sampling to a ``radius=1`` 4-connected
     neighbourhood around the previous best — exploiting the spatial
     correlation between adjacent beams' reward distributions to
     re-acquire after a beam-mismatch event.
  3. Eliminated cold-start scan.  MAMBA does not require pulling every
     arm once before exploiting, which would cost K*L = 256 steps; it
     bootstraps with Thompson posteriors at the prior and lets the
     discounted update concentrate them as evidence accumulates.

The implementation is intentionally minimal: discounted Welford-style
running mean per arm, scalar neighbourhood-explore trigger.  The Aykin
2020 paper additionally couples beam selection with MCS adaptation, which
we do not model (our channel returns a complex-valued ``y`` and the
runner reports SNR — there is no MCS).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState

_4CONNECTED = [(0, 1), (0, -1), (1, 0), (-1, 0)]


class MAMBA(Algorithm):
    """Adaptive Thompson sampling with beam-neighbourhood correlation.

    Parameters
    ----------
    gamma:
        Per-step discount factor on the running mean.  ``gamma=1`` recovers
        the stationary running average; the Aykin 2020 paper uses
        ``gamma=0.95-0.99`` on dynamic mmWave traces.
    sigma_floor:
        Minimum Thompson posterior std-dev so the sampler stays
        exploratory.  Without a floor, the posterior collapses after a
        few pulls and the algorithm reduces to greedy.
    explore_threshold:
        If the most recent reward at the current best arm drops below
        ``(1 - explore_threshold) * running_mean``, switch into a
        neighbourhood-only Thompson regime for ``explore_horizon`` steps.
    explore_horizon:
        Number of steps to spend in the neighbourhood-restricted regime
        once triggered.
    radius:
        Neighbourhood radius (in 4-connected hops) during the
        neighbourhood-explore phase.
    """

    name = "mamba"

    def __init__(
        self,
        gamma: float = 0.97,
        sigma_floor: float = 0.05,
        explore_threshold: float = 0.30,
        explore_horizon: int = 8,
        radius: int = 1,
    ) -> None:
        self._gamma = float(gamma)
        self._sigma_floor = float(sigma_floor)
        self._explore_threshold = float(explore_threshold)
        self._explore_horizon = int(explore_horizon)
        self._radius = int(radius)

    def reset(self, state: BPLMState, context: dict) -> None:
        K, L = state.K, state.L
        self._mean: NDArray[np.float64] = np.zeros((K, L), dtype=np.float64)
        # Effective count under discounting: n_eff = sum_t gamma^(T-t).
        # Bounded above by 1/(1-gamma); we cap to avoid overflow.
        self._n_eff: NDArray[np.float64] = np.zeros((K, L), dtype=np.float64)
        self._n_eff_cap = 1.0 / max(1.0 - self._gamma, 1e-6)
        self._best_kl: tuple[int, int] = (0, 0)
        self._best_mean: float = 0.0
        self._explore_counter: int = 0
        self._last_kl: tuple[int, int] | None = None
        self._rng = np.random.default_rng(context.get("trial_seed"))

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        K, L = state.K, state.L

        # Apply per-step discount before incorporating the new measurement.
        # n_eff <- gamma * n_eff (mean is unchanged by discounting alone).
        self._n_eff *= self._gamma

        if self._last_kl is not None:
            pk, pl = self._last_kl
            reward = float(np.abs(state.observations[pk, pl]))
            n_old = self._n_eff[pk, pl]
            n_new = min(n_old + 1.0, self._n_eff_cap)
            # Welford update at effective count.
            self._mean[pk, pl] += (reward - self._mean[pk, pl]) / max(n_new, 1e-9)
            self._n_eff[pk, pl] = n_new

            # Neighbourhood-explore trigger: if the just-pulled arm was the
            # best one and its single-pull reward fell well below the running
            # mean, kick into local-explore for a horizon.
            if (
                (pk, pl) == self._best_kl
                and self._best_mean > 0.0
                and reward < (1.0 - self._explore_threshold) * self._best_mean
            ):
                self._explore_counter = self._explore_horizon

        # Update best-arm tracker (after the update above so it stays current).
        flat_best = int(np.argmax(self._mean))
        self._best_kl = (flat_best // L, flat_best % L)
        self._best_mean = float(self._mean.flat[flat_best])

        # Posterior std-dev: sigma_floor / sqrt(max(1, n_eff)).  Sigma_floor
        # plays the role of a per-arm reward volatility prior; it does not
        # decay so the sampler remains exploratory even at saturation.
        std = self._sigma_floor / np.sqrt(np.maximum(self._n_eff, 1.0))
        samples = self._mean + std * self._rng.standard_normal((K, L))

        if self._explore_counter > 0:
            # Restrict argmax to the 4-connected neighbourhood of the
            # current best arm.  Mask out everything else to -inf so it
            # cannot win.
            mask = np.full_like(samples, -np.inf)
            kb, lb = self._best_kl
            for r in range(self._radius + 1):
                # Include the centre (r=0) and 4-connected neighbours up to radius.
                if r == 0:
                    mask[kb, lb] = samples[kb, lb]
                    continue
                for dk, dl in _4CONNECTED:
                    nk = kb + dk * r
                    nl = lb + dl * r
                    if 0 <= nk < K and 0 <= nl < L:
                        mask[nk, nl] = samples[nk, nl]
            samples = mask
            self._explore_counter -= 1

        flat = int(np.argmax(samples))
        k, l = flat // L, flat % L
        self._last_kl = (k, l)
        return k, l
