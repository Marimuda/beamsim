"""Channel model sanity tests."""

import numpy as np
import pytest

from beamsim.channel import (
    ChannelParams,
    ChannelRealisation,
    FreeSpaceLosChannel,
    umi_path_loss_db,
)


def test_umi_path_loss_monotone_in_distance():
    pl_50 = umi_path_loss_db(50.0, 28e9, 10.0, 1.5, los=True)
    pl_200 = umi_path_loss_db(200.0, 28e9, 10.0, 1.5, los=True)
    assert pl_200 > pl_50
    # Sanity: 28 GHz at 100 m UMi LOS should be ~100 dB
    pl_100 = umi_path_loss_db(100.0, 28e9, 10.0, 1.5, los=True)
    assert 90 < pl_100 < 110


def test_freespace_los_channel_shape():
    ch = FreeSpaceLosChannel(bs_xy=np.array([10.0, 0.0]), bs_yaw=0.0,
                              n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)
    assert H.shape == (16, 4)
    # Pure LOS rank-1 outer product => |H|_F^2 = 1 (unit gain)
    np.testing.assert_allclose(np.linalg.norm(H), 1.0, atol=1e-9)


def test_realisation_reproducibility_with_seed():
    params = ChannelParams()
    rng = np.random.default_rng(seed=12345)
    bs_xy = np.array([50.0, 50.0])
    real_a = ChannelRealisation(params=params, bs_xy=bs_xy, bs_yaw=0.0,
                                 n_bs_elements=16, n_ue_elements=4,
                                 rng=np.random.default_rng(seed=12345))
    real_b = ChannelRealisation(params=params, bs_xy=bs_xy, bs_yaw=0.0,
                                 n_bs_elements=16, n_ue_elements=4,
                                 rng=np.random.default_rng(seed=12345))
    H_a = real_a.channel_matrix(np.array([0.0, 0.0]), 0.0)
    H_b = real_b.channel_matrix(np.array([0.0, 0.0]), 0.0)
    np.testing.assert_allclose(H_a, H_b, atol=1e-12)


def test_realisation_channel_finite_and_nonzero():
    params = ChannelParams()
    rng = np.random.default_rng(seed=42)
    real = ChannelRealisation(params=params, bs_xy=np.array([50.0, 50.0]),
                               bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4, rng=rng)
    H = real.channel_matrix(np.array([0.0, 0.0]), 0.0)
    assert np.all(np.isfinite(H))
    assert np.linalg.norm(H) > 0


def test_los_dominates_when_k_high():
    params = ChannelParams(k_factor_mean_db=30.0, k_factor_std_db=0.0)
    rng = np.random.default_rng(seed=7)
    real = ChannelRealisation(params=params, bs_xy=np.array([50.0, 50.0]),
                               bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4, rng=rng)
    H = real.channel_matrix(np.array([0.0, 0.0]), 0.0)
    # With K=30 dB, LOS contributes >99% of the energy. Singular-value
    # ratio between the dominant SV and the rest should be >> 1.
    s = np.linalg.svd(H, compute_uv=False)
    assert s[0] / s[1] > 5
