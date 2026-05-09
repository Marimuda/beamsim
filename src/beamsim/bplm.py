"""Beam-Pair Link Matrix (BPLM) state and one-entry-per-occasion update."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from beamsim.codebook import Codebook, PlanarCodebook


@dataclass
class BPLMState:
    """Partially stale observation matrix Y_tilde plus per-entry age and history.

    The simulator updates exactly one entry per measurement occasion;
    elsewhere the entries reflect whatever was last measured at that pair.

    Either codebook side can be a :class:`Codebook` (cosine-spaced ULA) or a
    :class:`PlanarCodebook` (uniform planar array): both expose the same
    ``n_beams``, ``codeword(k)``, and ``theta`` interface that the BPLM
    measurement and the algorithms rely on.
    """

    ue_codebook: Codebook | PlanarCodebook
    bs_codebook: Codebook | PlanarCodebook
    noise_amplitude: float = 1e-3  # sqrt of noise variance, sets the noise floor

    def __post_init__(self) -> None:
        self.K = self.ue_codebook.n_beams  # rows
        self.L = self.bs_codebook.n_beams  # cols
        self.observations = np.zeros((self.K, self.L), dtype=np.complex128)
        self.measured_at = -np.ones((self.K, self.L), dtype=np.int64)  # -1 = never measured
        self.tx_amp = 1.0  # set by runner; absorbs Tx power scaling
        # history queues for MCMD's beam-quality (BQ) and volatility (v)
        self.history_obp_value: list[float] = []  # |y_obp(m)| over time
        self.previous_observations: NDArray[np.complex128] | None = None

    def measure(
        self, k: int, l: int, channel: NDArray[np.complex128], m: int, rng: np.random.Generator
    ) -> complex:
        """Update the (k, l) entry by sampling y = w_k^H H f_l x + w_k^H n."""
        w = self.ue_codebook.codeword(k)
        f = self.bs_codebook.codeword(l)
        signal = self.tx_amp * (w.conj() @ channel @ f)
        # w_k^H n with n ~ CN(0, sigma^2 I) has variance sigma^2 (unit-norm w)
        noise = (self.noise_amplitude / np.sqrt(2)) * (
            rng.standard_normal() + 1j * rng.standard_normal()
        )
        y = signal + noise
        self.observations[k, l] = y
        self.measured_at[k, l] = m
        return y

    def obp(self) -> tuple[int, int]:
        """Output beam-pair argmax of |Y_tilde|."""
        flat = np.argmax(np.abs(self.observations))
        return int(flat // self.L), int(flat % self.L)

    def obp_value(self) -> complex:
        k, l = self.obp()
        return self.observations[k, l]

    def age_matrix(self, current_m: int) -> NDArray[np.float64]:
        """Per-entry age in occasions; never-measured entries return current_m+1."""
        ages = np.where(self.measured_at < 0, current_m + 1, current_m - self.measured_at)
        return ages.astype(np.float64)

    def snapshot(self) -> NDArray[np.complex128]:
        return self.observations.copy()
