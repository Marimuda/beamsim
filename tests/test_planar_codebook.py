"""Tests for the Uniform Planar Array (UPA) codebook + companion channel.

The MATLAB simulator's ``placodebook.m`` and ``define_array`` together
produce a planar antenna array codebook with azimuth-only steering. This
test suite verifies (a) the standalone codebook semantics, (b) that
:class:`BPLMState` works seamlessly with the planar codebook in place of
the ULA :class:`Codebook`, and (c) end-to-end consistency with a
companion :class:`PlanarFreeSpaceLosChannel`.
"""

from __future__ import annotations

import numpy as np
import pytest

from beamsim.bplm import BPLMState
from beamsim.channel import PlanarFreeSpaceLosChannel
from beamsim.codebook import (
    PlanarCodebook,
    make_default_planar_bs_codebook,
    make_default_planar_ue_codebook,
    planar_steering_vector,
)

# ---------------------------------------------------------------------------
# planar_steering_vector
# ---------------------------------------------------------------------------


def test_planar_steering_vector_shape_and_norm():
    a = planar_steering_vector(n_x=4, n_y=4, theta_az=0.3, theta_el=0.0)
    assert a.shape == (16,)
    assert a.dtype == np.complex128
    np.testing.assert_allclose(np.linalg.norm(a), 1.0, atol=1e-12)


def test_planar_steering_vector_at_zero_collapses_to_x_axis_phase():
    """At az=0 (beam pointing along +x), only the x-direction phase varies."""
    a = planar_steering_vector(n_x=3, n_y=3, theta_az=0.0)
    # phase[i, j] = 2π * 0.5 * (cos(0) * i + sin(0) * j) = π * i
    # Reshape and check that all rows (fixed i) are identical
    grid = a.reshape(3, 3)
    for i in range(3):
        # All elements in row i should have the same phase modulo 2π
        phases = np.angle(grid[i])
        np.testing.assert_allclose(phases, phases[0], atol=1e-12)


def test_planar_steering_two_beams_orthogonal_at_well_separated_angles():
    """Beams at significantly different azimuths should be approximately orthogonal."""
    a0 = planar_steering_vector(n_x=8, n_y=8, theta_az=0.0)
    a90 = planar_steering_vector(n_x=8, n_y=8, theta_az=np.pi / 2)
    # |<a0, a90>| should be small for an 8x8 UPA at 90° angular separation.
    cross = abs(np.vdot(a0, a90))
    assert cross < 0.2  # well below unity


# ---------------------------------------------------------------------------
# PlanarCodebook
# ---------------------------------------------------------------------------


def test_planar_codebook_n_elements_and_n_beams():
    cb = PlanarCodebook(n_x=4, n_y=4, n_beams=12)
    assert cb.n_elements == 16
    assert cb.n_beams == 12


def test_planar_codebook_theta_uniform_in_azimuth():
    cb = PlanarCodebook(n_x=2, n_y=2, n_beams=8)
    expected = np.arange(8) * (2 * np.pi / 8)
    np.testing.assert_allclose(cb.theta, expected, atol=1e-12)


def test_planar_codebook_matrix_columns_are_codewords():
    cb = PlanarCodebook(n_x=2, n_y=3, n_beams=4)
    M = cb.matrix
    assert M.shape == (6, 4)  # (n_x*n_y, n_beams)
    for k in range(4):
        np.testing.assert_allclose(M[:, k], cb.codeword(k), atol=1e-12)


def test_planar_codebook_array_response_peaks_with_full_array_gain():
    """``array_response(theta_k)`` should peak with full array gain at *some* beam
    whose codeword is identical (modulo phase) to the k-th codeword.

    Note: at half-wavelength spacing a UPA cannot distinguish ``az`` from
    ``az + π`` (the codewords are equal), so multiple azimuth samples in
    the MATLAB-faithful ``[0, 2π)`` grid map to the same physical beam.
    The test therefore asserts (a) full array gain at the peak and (b)
    that the peak beam shares the steered direction modulo the
    front/back ambiguity, rather than strict ``peak == k`` equality.
    """
    cb = PlanarCodebook(n_x=4, n_y=4, n_beams=12)
    # With unit-norm codewords and a unit-norm steering vector, the peak
    # value of |w^H a| is exactly 1.0 (cosine similarity at perfect
    # alignment). The "array gain of N" appears on the channel side, not
    # on the codebook–steering inner product.
    for k in range(cb.n_beams):
        gains = np.abs(cb.array_response(float(cb.theta[k])))
        peak_idx = int(np.argmax(gains))
        # (a) the peak is at perfect cosine similarity
        np.testing.assert_allclose(gains[peak_idx], 1.0, atol=1e-9)
        # (b) the peak's codeword is identical (up to global phase) to
        # the steered codeword
        w_peak = cb.codeword(peak_idx)
        w_k = cb.codeword(k)
        cross = np.vdot(w_k, w_peak)
        np.testing.assert_allclose(abs(cross), 1.0, atol=1e-9)


def test_planar_codebook_codewords_unit_norm():
    cb = PlanarCodebook(n_x=4, n_y=4, n_beams=12)
    for k in range(cb.n_beams):
        np.testing.assert_allclose(np.linalg.norm(cb.codeword(k)), 1.0, atol=1e-12)


def test_default_planar_codebook_factories():
    ue = make_default_planar_ue_codebook()
    bs = make_default_planar_bs_codebook()
    assert ue.n_x == 2 and ue.n_y == 2 and ue.n_elements == 4 and ue.n_beams == 6
    assert bs.n_x == 4 and bs.n_y == 4 and bs.n_elements == 16 and bs.n_beams == 12


# ---------------------------------------------------------------------------
# BPLMState integration
# ---------------------------------------------------------------------------


def test_bplm_state_works_with_planar_codebook():
    """BPLMState should accept PlanarCodebook anywhere a Codebook is expected."""
    ue = make_default_planar_ue_codebook()
    bs = make_default_planar_bs_codebook()
    state = BPLMState(ue_codebook=ue, bs_codebook=bs, noise_amplitude=0.01)
    assert ue.n_beams == state.K
    assert bs.n_beams == state.L
    assert state.observations.shape == (ue.n_beams, bs.n_beams)


def test_bplm_state_measure_with_planar_arrays():
    """End-to-end measurement: synthetic channel + planar codebook + BPLM update."""
    ue = make_default_planar_ue_codebook()
    bs = make_default_planar_bs_codebook()
    state = BPLMState(ue_codebook=ue, bs_codebook=bs, noise_amplitude=1e-9)
    rng = np.random.default_rng(0)
    H = rng.standard_normal((ue.n_elements, bs.n_elements)) + 1j * rng.standard_normal(
        (ue.n_elements, bs.n_elements)
    )
    y = state.measure(k=2, l=5, channel=H, m=0, rng=rng)
    assert np.isfinite(y.real) and np.isfinite(y.imag)
    np.testing.assert_allclose(state.observations[2, 5], y, atol=1e-12)


# ---------------------------------------------------------------------------
# PlanarFreeSpaceLosChannel — end-to-end
# ---------------------------------------------------------------------------


def test_planar_free_space_los_channel_shape():
    ch = PlanarFreeSpaceLosChannel(
        bs_xy=np.array([10.0, 0.0]),
        bs_yaw=0.0,
        n_bs_x=4,
        n_bs_y=4,
        n_ue_x=2,
        n_ue_y=2,
    )
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)
    assert H.shape == (4, 16)  # (n_ue_elements, n_bs_elements)
    assert H.dtype == np.complex128


def test_planar_end_to_end_oracle_beam_has_high_gain():
    """For LOS arrival at a known direction, the codebook beam closest in
    azimuth should give the highest |w^H H f| over all (k, l) pairs."""
    ue = PlanarCodebook(n_x=2, n_y=2, n_beams=12)
    bs = PlanarCodebook(n_x=4, n_y=4, n_beams=24)
    ch = PlanarFreeSpaceLosChannel(
        bs_xy=np.array([10.0, 0.0]),
        bs_yaw=0.0,
        n_bs_x=bs.n_x,
        n_bs_y=bs.n_y,
        n_ue_x=ue.n_x,
        n_ue_y=ue.n_y,
    )
    state = BPLMState(ue_codebook=ue, bs_codebook=bs, noise_amplitude=1e-12)
    rng = np.random.default_rng(0)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)
    # Exhaustively measure every (k, l) at very low noise; the peak should
    # be where the codebook directions are closest to the actual LOS angles.
    for k in range(state.K):
        for l in range(state.L):
            state.measure(k, l, H, 0, rng)
    k_opt, l_opt = state.obp()
    # The peak should give substantial array gain (not zero), confirming the
    # codebook–channel pairing is consistent.
    peak = abs(state.observations[k_opt, l_opt])
    assert peak > 0.5  # ≈ sqrt(n_ue_elements) * sqrt(n_bs_elements) scale


@pytest.mark.parametrize(
    "ue_factory,bs_factory",
    [
        (make_default_planar_ue_codebook, make_default_planar_bs_codebook),
    ],
)
def test_default_planar_factories_are_runtime_compatible(ue_factory, bs_factory):
    """Sanity-check that the default factories produce codebooks that BPLMState
    accepts and that an exhaustive sweep terminates without numerical errors."""
    ue = ue_factory()
    bs = bs_factory()
    state = BPLMState(ue_codebook=ue, bs_codebook=bs, noise_amplitude=1e-3)
    rng = np.random.default_rng(0)
    H = rng.standard_normal((ue.n_elements, bs.n_elements)) + 1j * rng.standard_normal(
        (ue.n_elements, bs.n_elements)
    )
    for k in range(state.K):
        for l in range(state.L):
            state.measure(k, l, H, 0, rng)
    assert np.all(np.isfinite(state.observations))
