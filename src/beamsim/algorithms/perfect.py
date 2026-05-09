"""Perfect-knowledge baseline: noiseless codebook oracle.

This algorithm has zero measurement cost — it reads the noise-free channel
matrix ``H`` injected into ``context["true_H"]`` by the runner and returns
the (k, l) that maximises ``|w_k^H H f_l|`` over the full UE × BS codebook
product. It is the *codebook* oracle: the strongest SNR an algorithm could
ever report on the simulated finite codebook for the same channel
realisation. It is **not** a Shannon-capacity oracle and **not** a
deployable policy — it is a reference upper bound for diagnostic plots
only.
"""

from __future__ import annotations

import numpy as np

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState


class Perfect(Algorithm):
    """Codebook-oracle baseline: argmax of noiseless beam-gain over all (k, l)."""

    name = "perfect"

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        H: np.ndarray = context["true_H"]
        W = state.ue_codebook.matrix  # (n_ue, K)
        F = state.bs_codebook.matrix  # (n_bs, L)
        gains = np.abs(W.conj().T @ H @ F)  # (K, L)
        k, l = np.unravel_index(np.argmax(gains), gains.shape)
        return int(k), int(l)
