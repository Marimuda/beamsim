"""BPLM tests."""

import numpy as np

from beamsim.bplm import BPLMState
from beamsim.channel import FreeSpaceLosChannel
from beamsim.codebook import make_default_bs_codebook, make_default_ue_codebook


def test_bplm_shape_and_initial_state():
    s = BPLMState(
        ue_codebook=make_default_ue_codebook(),
        bs_codebook=make_default_bs_codebook(),
        noise_amplitude=0.0,
    )
    assert s.observations.shape == (8, 32)
    assert np.all(s.measured_at == -1)


def test_bplm_obp_returns_largest_entry():
    s = BPLMState(
        ue_codebook=make_default_ue_codebook(),
        bs_codebook=make_default_bs_codebook(),
        noise_amplitude=0.0,
    )
    s.observations[3, 17] = 5.0 + 0j
    s.observations[1, 1] = 1.0 + 0j
    assert s.obp() == (3, 17)


def test_bplm_measure_records_age():
    s = BPLMState(
        ue_codebook=make_default_ue_codebook(),
        bs_codebook=make_default_bs_codebook(),
        noise_amplitude=0.0,
    )
    ch = FreeSpaceLosChannel(
        bs_xy=np.array([10.0, 0.0]), bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4
    )
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)
    s.measure(2, 5, H, m=42, rng=np.random.default_rng(0))
    assert s.measured_at[2, 5] == 42
    age = s.age_matrix(current_m=50)
    assert age[2, 5] == 8
    # Never-measured entries report current_m+1
    assert age[0, 0] == 51


def test_bplm_los_match_recovers_full_array_gain():
    """With LOS aligned to a codebook beam, |y|^2 with no noise = N_BS * N_UE = 64."""
    s = BPLMState(
        ue_codebook=make_default_ue_codebook(),
        bs_codebook=make_default_bs_codebook(),
        noise_amplitude=0.0,
    )
    ue_cb = s.ue_codebook
    bs_cb = s.bs_codebook
    aoa_target = ue_cb.theta[4]  # desired UE-body AoA
    aod_target = bs_cb.theta[10]  # desired BS-body AoD
    bs_yaw = 0.0
    ue_yaw = 0.0
    # AoD at BS (body frame) = aod_target -> world direction from BS to UE
    aod_world = bs_yaw + aod_target
    # Place BS so that the UE (at origin) is 10 m along that direction from BS.
    bs_xy = -10.0 * np.array([np.cos(aod_world), np.sin(aod_world)])
    # World direction from UE to BS is aod_world + pi; for UE-body AoA = aoa_target
    # we set ue_yaw = (aod_world + pi) - aoa_target.
    ue_yaw = (aod_world + np.pi) - aoa_target
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=bs_yaw, n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), ue_yaw)
    s.tx_amp = np.sqrt(64.0)
    y = s.measure(4, 10, H, m=0, rng=np.random.default_rng(0))
    assert abs(y) ** 2 > 0.99 * 64
