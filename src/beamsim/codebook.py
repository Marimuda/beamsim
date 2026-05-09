"""Cosine-spaced linear-phase ULA codebook."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def steering_vector(n_elements: int, theta: float, spacing: float = 0.5) -> NDArray[np.complex128]:
    """Half-wavelength-spaced ULA steering vector at angle ``theta`` (radians).

    Returns a length-``n_elements`` complex vector, energy-normalised to unit
    norm so beam-pair products give array gain in [0, 1] for a single ray.
    """
    n = np.arange(n_elements)
    phase = -2.0 * np.pi * spacing * n * np.sin(theta)
    a = np.exp(1j * phase) / np.sqrt(n_elements)
    return a.astype(np.complex128)


@dataclass(frozen=True)
class Codebook:
    """Cosine-spaced (i.e. uniform-in-sin-theta) ULA codebook.

    For ``n_elements`` antennas and ``n_beams`` codewords, the steering
    directions are ``theta_k = arcsin(u_k)`` with ``u_k`` uniform on
    ``(-1, 1)`` (Chebyshev-like sampling, avoiding the endpoints which
    correspond to grazing incidence). This matches the predecessor MSc
    report's "cosine-spaced beams, linearly-phased array" convention.
    """

    n_elements: int
    n_beams: int
    spacing: float = 0.5

    @property
    def theta(self) -> NDArray[np.float64]:
        u = (2 * np.arange(self.n_beams) + 1) / self.n_beams - 1.0
        return np.arcsin(u)

    @property
    def matrix(self) -> NDArray[np.complex128]:
        """Returns an (n_elements, n_beams) matrix whose columns are codewords."""
        return np.column_stack(
            [steering_vector(self.n_elements, t, self.spacing) for t in self.theta]
        )

    def codeword(self, k: int) -> NDArray[np.complex128]:
        return steering_vector(self.n_elements, self.theta[k], self.spacing)

    def array_response(self, theta: float) -> NDArray[np.complex128]:
        """Per-beam complex gain when a single ray arrives from ``theta``.

        Returns a length-``n_beams`` vector with entries ``w_k^H a(theta)``.
        """
        a = steering_vector(self.n_elements, theta, self.spacing)
        return self.matrix.conj().T @ a


def make_default_ue_codebook() -> Codebook:
    """4-element UE ULA, 8 beams (predecessor baseline)."""
    return Codebook(n_elements=4, n_beams=8)


def make_default_bs_codebook() -> Codebook:
    """16-element BS ULA, 32 beams (predecessor baseline)."""
    return Codebook(n_elements=16, n_beams=32)
