"""Position-aided contextual MAB for beam selection.

Reference:
    Va, V., Shimizu, T., Bansal, G., Heath, R. W. (2019). "Online Learning
    for Position-Aided Millimeter Wave Beam Training." IEEE Access 7,
    30507-30526.  arXiv:1809.03014.

Key idea: the optimal beam pair is a function of UE position (and yaw).
Va et al. partition the spatial-domain context into a finite set of
*spatial bins* and maintain an independent Thompson posterior per bin.
At each step the algorithm:

    1. Reads the UE pose from ``context["ue_pose_at"](m)``.
    2. Maps it to a discrete bin (we use a fixed (x, y, yaw) grid;
       Va et al. use offline KMeans or a uniform spatial grid).
    3. Thompson-samples over arms within that bin's posterior and pulls
       the argmax.
    4. Updates the bin's per-arm running mean using the observed reward.

What it gets right that a non-contextual MAB does not
-----------------------------------------------------
On a revisited spatial position, the bin's posterior is already
informative — no fresh cold-start.  This is most visible in:
  * Rotational scenarios where the UE returns through the same yaws.
  * Handover scenarios where multiple BSs share the same UE-position
    distribution but differ only in geometry (the bin context handles
    this by including (bs_index) as part of the bin key, but here we
    treat each BS independently because the algorithm runs per-BS).

Implementation choices
----------------------
We use a deterministic (x, y, yaw) grid rather than online KMeans:
  * Grid resolution is configurable via the constructor.
  * Bin assignment is O(1) per step.
  * Different runs and trials get the same bin partition, which is
    important for reproducibility.

This is intentionally simpler than Va et al.'s offline-trained KMeans;
their gain comes from the *contextualisation* itself, not the clustering
algorithm.  An offline-clustered variant is left as a Phase 4C upgrade.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState


class PositionMAB(Algorithm):
    """Contextual Thompson sampling indexed by spatial bin.

    Parameters
    ----------
    n_bins_x, n_bins_y:
        Number of grid cells in x and y.  The grid spans
        ``[-x_extent, x_extent] x [-y_extent, y_extent]``.
    n_bins_yaw:
        Number of yaw bins on (-pi, pi].  ``1`` disables yaw context.
    x_extent, y_extent:
        Half-width of the spatial grid (metres).  Out-of-range positions
        are clipped to the boundary bin.
    sigma_floor:
        Minimum Thompson posterior std-dev per bin so a freshly-visited
        bin still produces exploratory samples.
    """

    name = "position_mab"

    def __init__(
        self,
        n_bins_x: int = 8,
        n_bins_y: int = 8,
        n_bins_yaw: int = 8,
        x_extent: float = 200.0,
        y_extent: float = 200.0,
        sigma_floor: float = 0.05,
    ) -> None:
        self._nx = max(int(n_bins_x), 1)
        self._ny = max(int(n_bins_y), 1)
        self._nyaw = max(int(n_bins_yaw), 1)
        self._x_extent = float(x_extent)
        self._y_extent = float(y_extent)
        self._sigma_floor = float(sigma_floor)

    def reset(self, state: BPLMState, context: dict) -> None:
        K, L = state.K, state.L
        n_bins = self._nx * self._ny * self._nyaw
        self._n_bins = n_bins
        # Per-bin per-arm running mean and pull count.
        self._mean: NDArray[np.float64] = np.zeros((n_bins, K, L), dtype=np.float64)
        self._counts: NDArray[np.int_] = np.zeros((n_bins, K, L), dtype=np.int_)
        # Pose lookup function comes from the runner context.
        self._pose_at = context.get("ue_pose_at")
        self._last_kl: tuple[int, int] | None = None
        self._last_bin: int | None = None
        self._rng = np.random.default_rng(context.get("trial_seed"))

    def _bin_index(self, ue_xy: NDArray[np.float64], ue_yaw: float) -> int:
        # Clip to extent then map to bin.
        x = float(np.clip(ue_xy[0], -self._x_extent, self._x_extent))
        y = float(np.clip(ue_xy[1], -self._y_extent, self._y_extent))
        # x_bin in [0, nx-1].
        x_bin = int(np.floor((x + self._x_extent) / (2 * self._x_extent) * self._nx))
        x_bin = max(0, min(self._nx - 1, x_bin))
        y_bin = int(np.floor((y + self._y_extent) / (2 * self._y_extent) * self._ny))
        y_bin = max(0, min(self._ny - 1, y_bin))
        # Yaw bin in [0, nyaw-1] over (-pi, pi].
        yaw_norm = (float(ue_yaw) + np.pi) % (2 * np.pi)
        yaw_bin = int(np.floor(yaw_norm / (2 * np.pi) * self._nyaw))
        yaw_bin = max(0, min(self._nyaw - 1, yaw_bin))
        return x_bin * (self._ny * self._nyaw) + y_bin * self._nyaw + yaw_bin

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        K, L = state.K, state.L

        # Update the previously-pulled bin's posterior with the observed reward.
        if self._last_kl is not None and self._last_bin is not None:
            pk, pl = self._last_kl
            reward = float(np.abs(state.observations[pk, pl]))
            n_old = self._counts[self._last_bin, pk, pl]
            n_new = n_old + 1
            self._mean[self._last_bin, pk, pl] += (
                reward - self._mean[self._last_bin, pk, pl]
            ) / n_new
            self._counts[self._last_bin, pk, pl] = n_new

        # Map the current pose to a bin.
        if self._pose_at is None:
            # Without pose context this degrades to a single-bin Thompson;
            # not a hard error so unit tests can still exercise the wrapper.
            bin_idx = 0
        else:
            ue_xy, ue_yaw = self._pose_at(m)
            bin_idx = self._bin_index(np.asarray(ue_xy), float(ue_yaw))

        # Thompson sampling within this bin's posterior.
        counts = self._counts[bin_idx]
        mean = self._mean[bin_idx]
        # Std-dev: floor / sqrt(max(1, n)).  A bin first-visited at this
        # step has all counts = 0; the floor keeps the sampler exploratory.
        std = self._sigma_floor / np.sqrt(np.maximum(counts.astype(np.float64), 1.0))
        samples = mean + std * self._rng.standard_normal((K, L))
        flat = int(np.argmax(samples))
        k, l = flat // L, flat % L
        self._last_kl = (k, l)
        self._last_bin = bin_idx
        return k, l
