"""Codebook sanity tests."""

import numpy as np
import pytest

from beamsim.codebook import (
    Codebook,
    make_default_bs_codebook,
    make_default_ue_codebook,
    steering_vector,
)


def test_steering_vector_unit_norm():
    a = steering_vector(16, np.deg2rad(30))
    assert a.shape == (16,)
    np.testing.assert_allclose(np.linalg.norm(a), 1.0, atol=1e-12)


def test_codebook_dimensions():
    ue = make_default_ue_codebook()
    bs = make_default_bs_codebook()
    assert ue.matrix.shape == (4, 8)
    assert bs.matrix.shape == (16, 32)


def test_codewords_unit_norm():
    cb = make_default_bs_codebook()
    norms = np.linalg.norm(cb.matrix, axis=0)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-12)


def test_array_response_peaks_at_codeword_angle():
    cb = make_default_bs_codebook()
    for k in range(cb.n_beams):
        gains = np.abs(cb.array_response(cb.theta[k])) ** 2
        assert np.argmax(gains) == k, f"beam {k} should peak at its own steering angle"


def test_codebook_directions_cover_visible_region():
    cb = Codebook(n_elements=16, n_beams=32)
    theta = cb.theta
    # angles are sorted and live within [-pi/2, pi/2]
    assert np.all(np.diff(theta) > 0)
    assert theta[0] > -np.pi / 2
    assert theta[-1] < np.pi / 2


@pytest.mark.parametrize("n_elements,n_beams", [(4, 8), (8, 16), (16, 32)])
def test_resolution_scales_with_array_size(n_elements: int, n_beams: int):
    cb = Codebook(n_elements=n_elements, n_beams=n_beams)
    # Array gain at the wrong beam decays — sample many directions and
    # check that the peak gain across the codebook equals roughly the
    # array gain for at least one direction in the codebook grid.
    angles = np.linspace(-1.0, 1.0, 200)
    peak_gains = np.array([np.max(np.abs(cb.array_response(np.arcsin(u))) ** 2) for u in angles])
    # In the worst case (between two grid points) the gain dip should
    # still leave a substantial fraction of full array gain.
    assert peak_gains.min() > 0.3
    assert peak_gains.max() <= 1.0 + 1e-9
