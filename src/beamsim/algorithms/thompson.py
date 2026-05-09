"""Thompson sampling (Gaussian) multi-armed bandit beam selection
(stationary, sanity-check baseline).

Each (k, l) beam pair is an arm with reward = |y(k,l)|, modelled as
Gaussian with unknown mean mu and (effectively) known variance sigma^2.
Posterior on mu after n samples with empirical mean X_bar is the standard
conjugate Gaussian-Gaussian update:
    mu | data ~ N(X_bar, sigma^2 / n).

Each call samples one mu_kl per arm and pulls the argmax (Thompson, 1933).
During cold-start every arm is pulled once so every posterior is proper.

References:
    Thompson, W. R. (1933). "On the likelihood that one unknown probability
    exceeds another in view of the evidence of two samples." Biometrika 25.
    Chapelle, O., Li, L. (2011). "An Empirical Evaluation of Thompson
    Sampling." NeurIPS — Gaussian-Gaussian conjugate variant.

Caveats:
    1. Sigma estimation.  The original Chapelle & Li formulation assumes
       the *reward* noise variance is known.  The naive choice
       sigma = state.noise_amplitude collapses the posterior to a delta
       after one sample because the reward variability across arms is
       dominated by signal magnitude (~1) rather than by measurement
       noise (~1e-3).  We instead track a running cross-arm sample
       standard deviation of |y(k,l)|; this is a reasonable proxy for the
       *effective* reward volatility and keeps Thompson sampling
       genuinely exploratory.  The Gaussian likelihood is still
       misspecified for the underlying Rician magnitude distribution,
       which is acknowledged but unfixed in this baseline.
    2. Stationary-arm assumption.  Same caveat as UCB1 — under mobility
       the per-arm posterior never forgets stale measurements; MAMBA
       (Phase 4B) replaces this with a discounted-update variant.
    3. Cold-start.  K*L = 256 forced pulls before the posterior matters.
       Below ~300-step trials Thompson collapses to row-major scanning.
    4. Hashemi 2018 / Va 2019 — previously cited — are *contextual*
       bandits.  Use ``PositionMAB`` (Phase 4B) for the position-aware
       variant; this implementation is intentionally context-free.
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
        # Welford-style running sum-of-squared-deviations for the cross-arm
        # reward distribution, used to estimate sigma online.
        self._reward_count: int = 0
        self._reward_mean: float = 0.0
        self._reward_m2: float = 0.0
        # Floor for sigma during cold start, so the first samples do not
        # collapse the posterior.  Once we have enough rewards (>= 8) we
        # switch to the empirical estimate.
        self._sigma_floor: float = max(state.noise_amplitude, 1e-3)
        self._cold_index: int = 0
        self._last_kl: tuple[int, int] | None = None
        # Seeded from trial_seed when available so different trials diverge.
        # Falls back to an unseeded RNG for unit-test / ad-hoc use.
        self._rng = np.random.default_rng(context.get("trial_seed"))

    def _update_reward_stats(self, reward: float) -> None:
        """Welford's online algorithm for the cross-arm reward variance."""
        self._reward_count += 1
        delta = reward - self._reward_mean
        self._reward_mean += delta / self._reward_count
        self._reward_m2 += delta * (reward - self._reward_mean)

    @property
    def _sigma(self) -> float:
        # Use empirical cross-arm sigma once enough samples are collected;
        # otherwise fall back to a noise-amplitude floor so cold-start
        # posteriors are not immediately collapsed.
        if self._reward_count < 8:
            return self._sigma_floor
        var = self._reward_m2 / max(self._reward_count - 1, 1)
        return max(float(np.sqrt(var)), self._sigma_floor)

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        K, L = state.K, state.L

        # Update statistics from the arm pulled last step
        if self._last_kl is not None:
            pk, pl = self._last_kl
            reward = float(np.abs(state.observations[pk, pl]))
            n = self._counts[pk, pl] + 1
            self._mean[pk, pl] += (reward - self._mean[pk, pl]) / n
            self._counts[pk, pl] = n
            self._update_reward_stats(reward)

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
