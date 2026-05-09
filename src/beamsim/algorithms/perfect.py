"""Perfect-knowledge baseline: noiseless oracle that always picks the true OBP.

This algorithm has zero measurement cost — it reads the noise-free channel
matrix ``H`` injected into ``context["true_H"]`` by the runner and returns
the (k, l) that maximises |w_k^H H f_l| over the full codebook product.
It is a reference upper bound, not a practical algorithm.
"""

from __future__ import annotations

import numpy as np

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState


class Perfect(Algorithm):
    """Oracle baseline: argmax of noiseless beam-gain over all (k, l)."""

    name = "perfect"

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        H: np.ndarray = context["true_H"]
        W = state.ue_codebook.matrix  # (n_ue, K)
        F = state.bs_codebook.matrix  # (n_bs, L)
        gains = np.abs(W.conj().T @ H @ F)  # (K, L)
        k, l = np.unravel_index(np.argmax(gains), gains.shape)
        return int(k), int(l)
