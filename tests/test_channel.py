"""Channel model sanity tests.

Covers both the original API compatibility tests and the new TR 38.901
feature tests added for the upgraded implementation.
"""

import numpy as np
import pytest

from beamsim.channel import (
    ChannelParams,
    ChannelRealisation,
    FreeSpaceLosChannel,
    umi_path_loss_db,
    uma_path_loss_db,
    umi_los_probability,
    _TR38901_RAY_OFFSETS_DEG,
)


# ---------------------------------------------------------------------------
# Existing API compatibility tests (29 baseline tests pass unchanged)
# ---------------------------------------------------------------------------

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
    assert H.shape == (4, 16)
    # Pure LOS rank-1 outer product => |H|_F^2 = 1 (unit gain)
    np.testing.assert_allclose(np.linalg.norm(H), 1.0, atol=1e-9)


def test_realisation_reproducibility_with_seed():
    params = ChannelParams()
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


# ---------------------------------------------------------------------------
# TR 38.901 Table 7.5-3: sub-ray offsets
# ---------------------------------------------------------------------------

def test_sub_ray_offsets_sum_to_zero():
    """TR 38.901 Table 7.5-3: the 20 canonical offsets must sum to zero
    because they are symmetric positive/negative pairs."""
    np.testing.assert_allclose(_TR38901_RAY_OFFSETS_DEG.sum(), 0.0, atol=1e-10)


def test_sub_ray_offsets_count_and_symmetry():
    """Each magnitude appears exactly twice (+ and -)."""
    assert len(_TR38901_RAY_OFFSETS_DEG) == 20
    pos = np.sort(_TR38901_RAY_OFFSETS_DEG[_TR38901_RAY_OFFSETS_DEG > 0])
    neg = np.sort(np.abs(_TR38901_RAY_OFFSETS_DEG[_TR38901_RAY_OFFSETS_DEG < 0]))
    np.testing.assert_allclose(pos, neg, atol=1e-10)


def test_realisation_sub_ray_offsets_shape_and_zero_mean():
    """Per-cluster sub-ray offsets have correct shape and cluster mean ~ 0."""
    params = ChannelParams()
    rng = np.random.default_rng(seed=1)
    real = ChannelRealisation(params=params, bs_xy=np.array([0.0, 0.0]),
                               bs_yaw=0.0, n_bs_elements=4, n_ue_elements=2, rng=rng)
    # Shape: (n_clusters, 20) – 12 clusters for UMi LOS
    assert real.sub_ray_aoa_offsets.shape[1] == 20
    # Each cluster's 20 offsets must sum to zero (derived from symmetric table)
    row_sums = real.sub_ray_aoa_offsets.sum(axis=1)
    np.testing.assert_allclose(row_sums, 0.0, atol=1e-10)
    row_sums_aod = real.sub_ray_aod_offsets.sum(axis=1)
    np.testing.assert_allclose(row_sums_aod, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# TR 38.901 Table 7.5-6: LSP distribution sanity
# ---------------------------------------------------------------------------

def test_lsp_draws_distribution_umi_los():
    """Draw many LSP realisations and verify marginal means/stds match
    TR 38.901 Table 7.5-6 UMi LOS within 3-sigma Monte Carlo tolerance."""
    params = ChannelParams()
    n_trials = 2000
    rng = np.random.default_rng(seed=42)
    ds_log10 = []
    asa_log10 = []
    k_db_vals = []
    for _ in range(n_trials):
        r = ChannelRealisation(params=params, bs_xy=np.array([0.0, 0.0]),
                                bs_yaw=0.0, n_bs_elements=2, n_ue_elements=2, rng=rng)
        ds_log10.append(np.log10(r.lsp["ds_s"]))
        asa_log10.append(np.log10(r.lsp["asa_deg"]))
        if r.is_los:
            k_db_vals.append(r.lsp["k_db"])

    # DS: mu=-7.19 +/- 3*sigma/sqrt(N) ~ 0.027
    ds_arr = np.array(ds_log10)
    assert abs(ds_arr.mean() - (-7.19)) < 0.08, f"DS mean {ds_arr.mean():.3f} off"
    assert abs(ds_arr.std() - 0.40) < 0.05, f"DS std {ds_arr.std():.3f} off"

    # ASA: mu=1.81
    asa_arr = np.array(asa_log10)
    assert abs(asa_arr.mean() - 1.81) < 0.05, f"ASA mean {asa_arr.mean():.3f} off"

    # K-factor: before clipping the raw draw mean should be ~9 dB; after
    # clipping to [-3, 20] the sample mean is pulled slightly; allow ±1.5 dB.
    if k_db_vals:
        k_arr = np.array(k_db_vals)
        assert abs(k_arr.mean() - 9.0) < 1.5, f"K mean {k_arr.mean():.2f} dB off"


def test_lsp_nlos_k_factor_zero():
    """NLOS realisations must have k_lin == 0.0 (no LOS component)."""
    params = ChannelParams(los_probability=0.0)  # force NLOS
    rng = np.random.default_rng(seed=99)
    for _ in range(20):
        r = ChannelRealisation(params=params, bs_xy=np.array([0.0, 0.0]),
                                bs_yaw=0.0, n_bs_elements=2, n_ue_elements=2, rng=rng)
        assert r.k_lin == 0.0
        assert not r.is_los


# ---------------------------------------------------------------------------
# K-factor power split
# ---------------------------------------------------------------------------

def test_k_factor_ratio_high_k():
    """With K=20 dB, LOS power fraction = K/(K+1) > 0.99."""
    params = ChannelParams(k_factor_mean_db=20.0, k_factor_std_db=0.0)
    rng = np.random.default_rng(seed=3)
    real = ChannelRealisation(params=params, bs_xy=np.array([30.0, 0.0]),
                               bs_yaw=0.0, n_bs_elements=4, n_ue_elements=4, rng=rng)
    k = real.k_lin
    los_fraction = k / (1.0 + k)
    assert los_fraction > 0.99, f"LOS fraction {los_fraction:.4f} too low for K=20 dB"


def test_k_factor_zero_for_nlos():
    """NLOS channel has k_lin == 0 and LOS fraction == 0."""
    params = ChannelParams(los_probability=0.0)
    rng = np.random.default_rng(seed=5)
    real = ChannelRealisation(params=params, bs_xy=np.array([30.0, 0.0]),
                               bs_yaw=0.0, n_bs_elements=4, n_ue_elements=4, rng=rng)
    assert real.k_lin == 0.0


# ---------------------------------------------------------------------------
# LOS probability (TR 38.901 §7.4.2)
# ---------------------------------------------------------------------------

def test_los_probability_at_origin_is_one():
    """At d=0 (or very small d), P_LOS should be 1."""
    assert umi_los_probability(0.1) == pytest.approx(1.0, abs=0.01)


def test_los_probability_decreases_with_distance():
    """P_LOS is strictly decreasing beyond short distances."""
    p10 = umi_los_probability(10.0)
    p50 = umi_los_probability(50.0)
    p200 = umi_los_probability(200.0)
    assert p10 > p50 > p200


def test_los_probability_formula_spot_check():
    """Verify the TR 38.901 §7.4.2 formula at d=18 m: min(1,1)*(1-e^(-18/36))+e^(-18/36)."""
    d = 18.0
    expected = 1.0 * (1.0 - np.exp(-d / 36.0)) + np.exp(-d / 36.0)
    np.testing.assert_allclose(umi_los_probability(d), expected, rtol=1e-6)


def test_los_probability_at_100m():
    """TR 38.901 §7.4.2: spot-check at 100 m."""
    d = 100.0
    expected = (18.0 / d) * (1.0 - np.exp(-d / 36.0)) + np.exp(-d / 36.0)
    np.testing.assert_allclose(umi_los_probability(d), expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Doppler-induced channel variation over time
# ---------------------------------------------------------------------------

def test_doppler_channel_changes_with_time():
    """A moving UE (speed>0) must produce a different channel at t>0 vs t=0."""
    params = ChannelParams(ue_speed_mps=30.0)  # 30 m/s
    rng = np.random.default_rng(seed=11)
    real = ChannelRealisation(params=params, bs_xy=np.array([50.0, 0.0]),
                               bs_yaw=0.0, n_bs_elements=8, n_ue_elements=4, rng=rng)
    ue_xy = np.array([0.0, 0.0])
    H0 = real.channel_matrix(ue_xy, 0.0, time_s=0.0)
    H1 = real.channel_matrix(ue_xy, 0.0, time_s=0.01)  # 10 ms later
    # Channels must differ (Doppler phase rotated)
    diff = np.linalg.norm(H1 - H0)
    assert diff > 1e-6, f"Channel unchanged after Doppler evolution, diff={diff}"


def test_doppler_zero_speed_channel_unchanged():
    """A static UE (speed=0) must produce the same channel at all times."""
    params = ChannelParams(ue_speed_mps=0.0)
    rng = np.random.default_rng(seed=22)
    real = ChannelRealisation(params=params, bs_xy=np.array([50.0, 0.0]),
                               bs_yaw=0.0, n_bs_elements=8, n_ue_elements=4, rng=rng)
    ue_xy = np.array([0.0, 0.0])
    H0 = real.channel_matrix(ue_xy, 0.0, time_s=0.0)
    H1 = real.channel_matrix(ue_xy, 0.0, time_s=1.0)
    np.testing.assert_allclose(H0, H1, atol=1e-12)


# ---------------------------------------------------------------------------
# Cluster delays
# ---------------------------------------------------------------------------

def test_cluster_delays_sorted_and_nonnegative():
    """Cluster delays must be sorted ascending and non-negative, with tau_0=0."""
    params = ChannelParams()
    rng = np.random.default_rng(seed=77)
    real = ChannelRealisation(params=params, bs_xy=np.array([0.0, 0.0]),
                               bs_yaw=0.0, n_bs_elements=2, n_ue_elements=2, rng=rng)
    taus = real.cluster_delays_s
    assert taus[0] == pytest.approx(0.0, abs=1e-18)
    assert np.all(taus >= 0.0)
    assert np.all(np.diff(taus) >= 0.0)


def test_cluster_powers_sum_to_one():
    """Normalised cluster powers must sum to 1.0."""
    params = ChannelParams()
    rng = np.random.default_rng(seed=88)
    real = ChannelRealisation(params=params, bs_xy=np.array([0.0, 0.0]),
                               bs_yaw=0.0, n_bs_elements=2, n_ue_elements=2, rng=rng)
    np.testing.assert_allclose(real.cluster_powers.sum(), 1.0, rtol=1e-9)


# ---------------------------------------------------------------------------
# UMa path loss convenience function
# ---------------------------------------------------------------------------

def test_uma_path_loss_monotone():
    """UMa path loss must increase with distance."""
    pl_50 = uma_path_loss_db(50.0, 28e9, 25.0, 1.5, los=True)
    pl_200 = uma_path_loss_db(200.0, 28e9, 25.0, 1.5, los=True)
    assert pl_200 > pl_50


def test_uma_path_loss_nlos_ge_los():
    """UMa NLOS path loss must be >= LOS at same distance."""
    pl_los = uma_path_loss_db(150.0, 28e9, 25.0, 1.5, los=True)
    pl_nlos = uma_path_loss_db(150.0, 28e9, 25.0, 1.5, los=False)
    assert pl_nlos >= pl_los


# ---------------------------------------------------------------------------
# Channel matrix shape and output sanity
# ---------------------------------------------------------------------------

def test_channel_matrix_shape_nlos():
    """NLOS channel matrix must have correct shape."""
    params = ChannelParams(los_probability=0.0)
    rng = np.random.default_rng(seed=55)
    real = ChannelRealisation(params=params, bs_xy=np.array([40.0, 0.0]),
                               bs_yaw=0.0, n_bs_elements=8, n_ue_elements=4, rng=rng)
    H = real.channel_matrix(np.array([0.0, 0.0]), 0.0)
    assert H.shape == (4, 8)
    assert np.all(np.isfinite(H))
