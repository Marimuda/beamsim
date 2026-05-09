"""Tests for the four algorithms ported from the MATLAB simulator's
``tracking_algos/`` directory: AgeMx, RandomSearch, NNSTabu (Ascent_Tabu),
and ContextInformationMBS (the multi-BS CI variant). The corresponding
MATLAB sources live in ``05_Clean_Simulator/tracking_algos/`` in the
parent research directory; the comparison-of-record is
``docs/MATLAB_PARITY.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from beamsim.algorithms import (
    AgeMx,
    ContextInformationMBS,
    NNSTabu,
    RandomSearch,
)
from beamsim.bplm import BPLMState
from beamsim.codebook import make_default_bs_codebook, make_default_ue_codebook


def _state() -> BPLMState:
    return BPLMState(
        ue_codebook=make_default_ue_codebook(),
        bs_codebook=make_default_bs_codebook(),
        noise_amplitude=0.01,
    )


# ---------------------------------------------------------------------------
# AgeMx
# ---------------------------------------------------------------------------


def test_agemx_first_pass_visits_every_pair():
    """Over K*L steps AgeMx must hit every (k, l) exactly once."""
    state = _state()
    algo = AgeMx()
    algo.reset(state, {"trial_seed": 0})
    visited: set[tuple[int, int]] = set()
    for m in range(state.K * state.L):
        kl = algo.select_next_mbp(state, m, {})
        visited.add(kl)
    assert len(visited) == state.K * state.L


def test_agemx_repeats_after_full_pass():
    """After the first sweep, AgeMx revisits in the same order at the slowest rate."""
    state = _state()
    algo = AgeMx()
    algo.reset(state, {"trial_seed": 0})
    first_pass = [algo.select_next_mbp(state, m, {}) for m in range(state.K * state.L)]
    second_pass = [
        algo.select_next_mbp(state, m, {}) for m in range(state.K * state.L, 2 * state.K * state.L)
    ]
    assert first_pass == second_pass


# ---------------------------------------------------------------------------
# RandomSearch
# ---------------------------------------------------------------------------


def test_random_first_pass_visits_every_pair():
    state = _state()
    algo = RandomSearch()
    algo.reset(state, {"trial_seed": 7})
    visited = {algo.select_next_mbp(state, m, {}) for m in range(state.K * state.L)}
    assert len(visited) == state.K * state.L


def test_random_is_deterministic_under_same_seed():
    state = _state()
    algo_a = RandomSearch()
    algo_b = RandomSearch()
    algo_a.reset(state, {"trial_seed": 42})
    algo_b.reset(state, {"trial_seed": 42})
    n = 5 * state.K * state.L
    seq_a = [algo_a.select_next_mbp(state, m, {}) for m in range(n)]
    seq_b = [algo_b.select_next_mbp(state, m, {}) for m in range(n)]
    assert seq_a == seq_b


def test_random_differs_under_different_seeds():
    state = _state()
    algo_a = RandomSearch()
    algo_b = RandomSearch()
    algo_a.reset(state, {"trial_seed": 1})
    algo_b.reset(state, {"trial_seed": 2})
    seq_a = [algo_a.select_next_mbp(state, m, {}) for m in range(state.K * state.L)]
    seq_b = [algo_b.select_next_mbp(state, m, {}) for m in range(state.K * state.L)]
    assert seq_a != seq_b


# ---------------------------------------------------------------------------
# NNSTabu (Ascent_Tabu)
# ---------------------------------------------------------------------------


def test_nns_tabu_emits_five_cell_pattern():
    """Each cycle of Ascent_Tabu emits centre + 4 cardinal neighbours at offset 2."""
    state = _state()
    algo = NNSTabu()
    algo.reset(state, {"trial_seed": 13})

    first_five = [algo.select_next_mbp(state, m, {}) for m in range(5)]
    assert len(set(first_five)) == 5  # all distinct
    centre = first_five[0]
    K, L = state.K, state.L
    expected = {
        centre,
        ((centre[0] - 2) % K, centre[1]),
        ((centre[0] + 2) % K, centre[1]),
        (centre[0], (centre[1] - 2) % L),
        (centre[0], (centre[1] + 2) % L),
    }
    assert set(first_five) == expected


def test_nns_tabu_relocates_to_global_argmax_when_list_empty():
    """The defining property: relocation uses GLOBAL argmax of |Y_obs|."""
    state = _state()
    algo = NNSTabu()
    algo.reset(state, {"trial_seed": 0})

    # Drain the first 5-cell list. Inject a dominant signal far from the
    # initial centre so the global argmax lands there.
    far_kl = (state.K - 1, state.L - 1)
    state.observations[far_kl] = 100.0 + 0.0j

    for m in range(5):
        algo.select_next_mbp(state, m, {})

    # Sixth call rebuilds the list around the global argmax.
    next_kl = algo.select_next_mbp(state, 5, {})
    centre = far_kl
    K, L = state.K, state.L
    expected = {
        centre,
        ((centre[0] - 2) % K, centre[1]),
        ((centre[0] + 2) % K, centre[1]),
        (centre[0], (centre[1] - 2) % L),
        (centre[0], (centre[1] + 2) % L),
    }
    assert next_kl in expected


def test_nns_tabu_indices_in_range():
    state = _state()
    algo = NNSTabu()
    rng = np.random.default_rng(0)
    algo.reset(state, {"trial_seed": 99})
    for m in range(50):
        # Inject some noise into observations so global argmax wanders.
        state.observations += 0.01 * (
            rng.standard_normal(state.observations.shape)
            + 1j * rng.standard_normal(state.observations.shape)
        )
        k, l = algo.select_next_mbp(state, m, {})
        assert 0 <= k < state.K
        assert 0 <= l < state.L


# ---------------------------------------------------------------------------
# ContextInformationMBS
# ---------------------------------------------------------------------------


def test_ci_mbs_picks_closest_bs():
    """When two BSs are configured, CIMBS aims at the closest one."""
    state = _state()
    algo = ContextInformationMBS()
    ue_xy = np.array([0.0, 0.0])
    ue_yaw = 0.0
    bs_far = np.array([1000.0, 0.0])
    bs_near = np.array([0.0, 50.0])
    context = {
        "ue_pose_at": lambda m: (ue_xy, ue_yaw),
        "bs_list": [
            {"bs_xy": bs_far, "bs_yaw": 0.0},
            {"bs_xy": bs_near, "bs_yaw": 0.0},
        ],
    }
    algo.reset(state, context)
    k_near, l_near = algo.select_next_mbp(state, 0, context)

    # Compare against single-BS CI on the near BS only.
    near_only_context = {
        "ue_pose_at": lambda m: (ue_xy, ue_yaw),
        "bs_xy": bs_near,
        "bs_yaw": 0.0,
    }
    from beamsim.algorithms import ContextInformation

    ci = ContextInformation()
    ci.reset(state, near_only_context)
    k_ci, l_ci = ci.select_next_mbp(state, 0, near_only_context)
    assert (k_near, l_near) == (k_ci, l_ci)


def test_ci_mbs_falls_back_to_single_bs_context():
    """Without bs_list, CIMBS uses the single bs_xy / bs_yaw fields."""
    state = _state()
    algo = ContextInformationMBS()
    ue_xy = np.array([0.0, 0.0])
    bs_xy = np.array([100.0, 0.0])
    context = {
        "ue_pose_at": lambda m: (ue_xy, 0.0),
        "bs_xy": bs_xy,
        "bs_yaw": 0.0,
    }
    algo.reset(state, context)
    k, l = algo.select_next_mbp(state, 0, context)
    assert 0 <= k < state.K
    assert 0 <= l < state.L


# ---------------------------------------------------------------------------
# All four algorithms reachable through ALL_ALGORITHMS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["agemx", "random", "nns_tabu", "ci_mbs"])
def test_all_algorithms_dict_exposes_new_ports(name: str):
    from beamsim.algorithms import ALL_ALGORITHMS

    assert name in ALL_ALGORITHMS
    cls = ALL_ALGORITHMS[name]
    instance = cls()
    state = _state()
    bs_xy = np.array([10.0, 0.0])
    context = {
        "ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0),
        "bs_xy": bs_xy,
        "bs_yaw": 0.0,
        "trial_seed": 0,
    }
    instance.reset(state, context)
    k, l = instance.select_next_mbp(state, 0, context)
    assert 0 <= k < state.K
    assert 0 <= l < state.L
