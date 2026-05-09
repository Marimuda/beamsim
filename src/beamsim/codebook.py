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


# ---------------------------------------------------------------------------
# Uniform Planar Array (UPA) codebook
# ---------------------------------------------------------------------------


def planar_steering_vector(
    n_x: int,
    n_y: int,
    theta_az: float,
    theta_el: float = 0.0,
    spacing: float = 0.5,
) -> NDArray[np.complex128]:
    """Half-wavelength-spaced UPA steering vector at ``(theta_az, theta_el)``.

    Mirrors the MATLAB ``placodebook.m`` convention: elements are arranged
    on a rectangular grid in the xy-plane at positions ``(i*d, j*d, 0)`` for
    ``i in [0, n_x)``, ``j in [0, n_y)``, with ``d = spacing * lambda``.
    Azimuth ``theta_az`` is measured CCW from the +x-axis; elevation
    ``theta_el`` from the xy-plane (positive toward +z). At ``theta_el=0``
    the beam steers entirely in-plane.

    The output is flattened in row-major order so the returned shape is
    ``(n_x * n_y,)``: index ``i * n_y + j`` corresponds to element
    ``(i, j)``. Each codeword is unit-norm to match the ULA
    :func:`steering_vector` convention.
    """
    cos_el = float(np.cos(theta_el))
    i = np.arange(n_x).reshape(n_x, 1)
    j = np.arange(n_y).reshape(1, n_y)
    phase = 2.0 * np.pi * spacing * cos_el * (np.cos(theta_az) * i + np.sin(theta_az) * j)
    a = np.exp(1j * phase) / np.sqrt(n_x * n_y)
    return a.reshape(-1).astype(np.complex128)


@dataclass(frozen=True)
class PlanarCodebook:
    """Uniform Planar Array (UPA) codebook with azimuth-only steering.

    Mirrors MATLAB ``placodebook.m``: an ``n_x x n_y`` rectangular array
    in the xy-plane at half-wavelength spacing. The codebook samples
    azimuth uniformly over ``[0, 2*pi)`` at fixed elevation 0 by default.
    The number of beams defaults to ``3 * max(n_x, n_y)`` per the MATLAB
    rule (``Naz = max(ar.x, ar.y) * 3``).

    For ``BPLMState`` and the algorithms this class behaves exactly like
    :class:`Codebook`: it exposes ``n_elements``, ``n_beams``, ``theta``,
    ``matrix``, ``codeword``, and ``array_response``. The fundamental
    difference is the sampling: ULA codebooks sample uniformly in
    ``sin(theta)`` over ``(-1, 1)``; planar codebooks sample uniformly
    in azimuth over ``[0, 2*pi)``.
    """

    n_x: int
    n_y: int
    n_beams: int
    spacing: float = 0.5
    elevation: float = 0.0

    @property
    def n_elements(self) -> int:
        return self.n_x * self.n_y

    @property
    def theta(self) -> NDArray[np.float64]:
        """Azimuth angles uniformly sampled over ``[0, 2*pi)``."""
        return (np.arange(self.n_beams) * (2.0 * np.pi / self.n_beams)).astype(np.float64)

    @property
    def matrix(self) -> NDArray[np.complex128]:
        """``(n_elements, n_beams)`` matrix whose columns are codewords."""
        return np.column_stack(
            [
                planar_steering_vector(self.n_x, self.n_y, az, self.elevation, self.spacing)
                for az in self.theta
            ]
        )

    def codeword(self, k: int) -> NDArray[np.complex128]:
        return planar_steering_vector(
            self.n_x, self.n_y, float(self.theta[k]), self.elevation, self.spacing
        )

    def array_response(self, theta_az: float, theta_el: float = 0.0) -> NDArray[np.complex128]:
        """Per-beam complex gain when a single ray arrives from ``(theta_az, theta_el)``.

        Returns a length-``n_beams`` vector with entries ``w_k^H a(az, el)``
        where ``w_k`` is the k-th codeword and ``a`` is the array steering
        vector at the requested direction.
        """
        a = planar_steering_vector(self.n_x, self.n_y, theta_az, theta_el, self.spacing)
        return self.matrix.conj().T @ a


def make_default_planar_ue_codebook() -> PlanarCodebook:
    """2x2 UE UPA, 6 beams (3 * max(2,2)) — the MATLAB ``placodebook.m`` default."""
    return PlanarCodebook(n_x=2, n_y=2, n_beams=6)


def make_default_planar_bs_codebook() -> PlanarCodebook:
    """4x4 BS UPA, 12 beams (3 * max(4,4)) — typical mmWave UPA configuration."""
    return PlanarCodebook(n_x=4, n_y=4, n_beams=12)
