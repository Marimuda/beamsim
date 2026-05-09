"""Property-based tests for beamsim invariants using hypothesis."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from beamsim.algorithms import ALL_ALGORITHMS
from beamsim.bplm import BPLMState
from beamsim.codebook import Codebook, steering_vector
from beamsim.geometry import rotation_track

# ---------------------------------------------------------------------------
# Codebook properties
# ---------------------------------------------------------------------------


@settings(max_examples=50)
@given(
    n_elements=st.integers(min_value=2, max_value=32),
    n_beams=st.integers(min_value=2, max_value=64),
)
def test_codebook_unit_norm_holds(n_elements: int, n_beams: int) -> None:
    """Every codeword in any codebook has unit L2 norm."""
    cb = Codebook(n_elements=n_elements, n_beams=n_beams)
    norms = np.linalg.norm(cb.matrix, axis=0)
    np.testing.assert_allclose(norms, np.ones(n_beams), atol=1e-12)


@settings(max_examples=50)
@given(
    n_elements=st.integers(min_value=2, max_value=32),
    n_beams=st.integers(min_value=2, max_value=64),
)
def test_codebook_dimensions_match(n_elements: int, n_beams: int) -> None:
    """Codebook.matrix.shape is exactly (n_elements, n_beams)."""
    cb = Codebook(n_elements=n_elements, n_beams=n_beams)
    assert cb.matrix.shape == (n_elements, n_beams)


@settings(max_examples=50)
@given(theta=st.floats(min_value=-math.pi / 2, max_value=math.pi / 2, allow_nan=False))
def test_steering_vector_norm(theta: float) -> None:
    """Steering vector is unit-norm for any valid elevation angle."""
    a = steering_vector(n_elements=16, theta=theta)
    assert abs(np.linalg.norm(a) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# BPLM properties
# ---------------------------------------------------------------------------


def _filled_bplm(K: int, L: int, seed: int) -> BPLMState:
    """Create a BPLMState with all entries set from random complex values."""
    ue_cb = Codebook(n_elements=4, n_beams=K)
    bs_cb = Codebook(n_elements=16, n_beams=L)
    state = BPLMState(ue_codebook=ue_cb, bs_codebook=bs_cb)
    rng = np.random.default_rng(seed)
    state.observations[:] = rng.standard_normal((K, L)) + 1j * rng.standard_normal((K, L))
    state.measured_at[:] = 0
    return state


@settings(max_examples=50)
@given(
    K=st.integers(min_value=2, max_value=16),
    L=st.integers(min_value=2, max_value=32),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_obp_returns_argmax(K: int, L: int, seed: int) -> None:
    """bplm.obp() always returns the (k, l) index of argmax|observations|."""
    state = _filled_bplm(K, L, seed)
    k, l = state.obp()
    expected_flat = int(np.argmax(np.abs(state.observations)))
    expected_k, expected_l = divmod(expected_flat, L)
    assert k == expected_k
    assert l == expected_l


# ---------------------------------------------------------------------------
# Algorithm validity properties
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=2000)
@given(
    algo_name=st.sampled_from(sorted(ALL_ALGORITHMS.keys())),
    K=st.integers(min_value=4, max_value=16),
    L=st.integers(min_value=4, max_value=32),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_algorithm_returns_valid_index(algo_name: str, K: int, L: int, seed: int) -> None:
    """Every algorithm returns (k, l) in [0, K) x [0, L) for any BPLM state."""
    n_ue, n_bs = 4, 16
    state = _filled_bplm(K, L, seed)

    # Perfect algo needs true_H of shape (n_ue_elements, n_bs_elements)
    rng = np.random.default_rng(seed + 1)
    true_H = rng.standard_normal((n_ue, n_bs)) + 1j * rng.standard_normal((n_ue, n_bs))

    bs_xy = np.array([10.0, 0.0])
    context: dict = {
        "ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0),
        "bs_xy": bs_xy,
        "bs_yaw": 0.0,
        "true_H": true_H,
    }

    algo = ALL_ALGORITHMS[algo_name]()
    algo.reset(state, context)
    k, l = algo.select_next_mbp(state, m=0, context=context)

    assert 0 <= k < K, f"{algo_name}: k={k} out of [0, {K})"
    assert 0 <= l < L, f"{algo_name}: l={l} out of [0, {L})"


# ---------------------------------------------------------------------------
# Track / geometry properties
# ---------------------------------------------------------------------------


@settings(max_examples=50)
@given(
    n_steps=st.integers(min_value=2, max_value=200),
    rpm=st.floats(min_value=0.1, max_value=600.0, allow_nan=False),
)
def test_track_orientation_step_size(n_steps: int, rpm: float) -> None:
    """rotation_track orientations advance by exactly omega*dt per step.

    rotation_track stores raw (cumulative) angles, not wrapped. The invariant
    is that consecutive differences equal 2*pi*rpm/60 * dt uniformly.
    """
    dt = 1e-3
    track = rotation_track(position_xy=(0.0, 0.0), rpm=rpm, n_steps=n_steps, dt=dt)

    omega = rpm * 2 * math.pi / 60.0  # rad/s
    expected_step = omega * dt
    diffs = np.diff(track.orientations)
    np.testing.assert_allclose(diffs, expected_step, rtol=1e-10)

    # Wrapped versions (via geometry._wrap_pi idiom) should lie in (-pi, pi]
    wrapped = (track.orientations + math.pi) % (2 * math.pi) - math.pi
    assert np.all(wrapped >= -math.pi - 1e-12)
    assert np.all(wrapped <= math.pi + 1e-12)


# ---------------------------------------------------------------------------
# MAB algorithm property tests
# ---------------------------------------------------------------------------


@settings(max_examples=30, deadline=3000)
@given(
    K=st.integers(min_value=2, max_value=8),
    L=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_thompson_returns_valid_index_property(K: int, L: int, seed: int) -> None:
    """ThompsonGaussian always returns (k, l) in [0, K) x [0, L)."""
    from beamsim.algorithms.thompson import ThompsonGaussian

    state = _filled_bplm(K, L, seed)
    algo = ThompsonGaussian()
    algo.reset(state, {})

    rng = np.random.default_rng(seed + 7)
    for m in range(K * L + 5):
        k, l = algo.select_next_mbp(state, m, {})
        assert 0 <= k < K, f"ThompsonGaussian k={k} out of [0, {K})"
        assert 0 <= l < L, f"ThompsonGaussian l={l} out of [0, {L})"
        reward = float(rng.random())
        state.observations[k, l] = complex(reward)
        state.measured_at[k, l] = m


@settings(max_examples=30, deadline=3000)
@given(
    K=st.integers(min_value=2, max_value=8),
    L=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_ucb1_returns_valid_index_property(K: int, L: int, seed: int) -> None:
    """UCB1 always returns (k, l) in [0, K) x [0, L)."""
    from beamsim.algorithms.ucb1 import UCB1

    state = _filled_bplm(K, L, seed)
    algo = UCB1()
    algo.reset(state, {})

    rng = np.random.default_rng(seed + 13)
    for m in range(K * L + 5):
        k, l = algo.select_next_mbp(state, m, {})
        assert 0 <= k < K, f"UCB1 k={k} out of [0, {K})"
        assert 0 <= l < L, f"UCB1 l={l} out of [0, {L})"
        reward = float(rng.random())
        state.observations[k, l] = complex(reward)
        state.measured_at[k, l] = m
