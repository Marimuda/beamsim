"""Algorithm-level smoke tests."""

import numpy as np
import pytest

from beamsim.algorithms import ALL_ALGORITHMS
from beamsim.bplm import BPLMState
from beamsim.channel import FreeSpaceLosChannel
from beamsim.codebook import make_default_bs_codebook, make_default_ue_codebook


def make_state():
    return BPLMState(ue_codebook=make_default_ue_codebook(),
                      bs_codebook=make_default_bs_codebook(),
                      noise_amplitude=0.01)


@pytest.mark.parametrize("name", sorted(ALL_ALGORITHMS))
def test_algorithm_returns_valid_indices(name):
    cls = ALL_ALGORITHMS[name]
    algo = cls()
    state = make_state()
    bs_xy = np.array([10.0, 0.0])
    context = {
        "ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0),
        "bs_xy": bs_xy,
        "bs_yaw": 0.0,
    }
    algo.reset(state, context)
    rng = np.random.default_rng(0)
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0,
                              n_bs_elements=16, n_ue_elements=4)
    for m in range(50):
        k, l = algo.select_next_mbp(state, m, context)
        assert 0 <= k < state.K
        assert 0 <= l < state.L
        H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)
        state.measure(k, l, H, m, rng)


def test_exhaustive_visits_every_pair_in_one_cycle():
    from beamsim.algorithms import Exhaustive
    algo = Exhaustive()
    state = make_state()
    algo.reset(state, {})
    visited = set()
    for m in range(state.K * state.L):
        visited.add(algo.select_next_mbp(state, m, {}))
    assert len(visited) == state.K * state.L


def test_ci_picks_geometry_aligned_pair():
    from beamsim.algorithms import ContextInformation
    algo = ContextInformation()
    state = make_state()
    bs_xy = np.array([10.0, 0.0])
    context = {
        "ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0),
        "bs_xy": bs_xy,
        "bs_yaw": 0.0,
    }
    algo.reset(state, context)
    k, l = algo.select_next_mbp(state, 0, context)
    # AoA at UE = 0 (BS lies along +x in body frame). Expect k near middle of codebook (theta=0).
    ue_theta = state.ue_codebook.theta
    bs_theta = state.bs_codebook.theta
    assert abs(ue_theta[k]) < ue_theta[k + 1] - ue_theta[k] + 1e-6
    # AoD at BS = pi (UE behind BS along -x). Wrap and check that the chosen
    # BS beam is the one whose theta matches the wrapped AoD-pi best.
    # Codebook spans (-pi/2, pi/2); aod_rel = pi wraps to -pi which is outside
    # the visible region — argmin will pick the extreme beam.
    assert l in {0, state.L - 1}
