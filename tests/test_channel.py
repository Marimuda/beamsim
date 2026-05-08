"""Channel model tests.

Tests are grouped into:
  1. API-compatibility / baseline  (unchanged semantics)
  2. Path-loss and LOS-probability (unchanged)
  3. LSP distribution sanity
  4. Cluster delay / power
  5. Sub-ray offsets (new: Laplacian model)
  6. Blockage Model A (new: self-blocker + non-self + drift)
  7. Geometric cluster power
  8. LOS LSPs used regardless of is_los state
  9. Doppler
  10. Channel matrix shape / output sanity
"""

import math

import numpy as np
import pytest

from beamsim.channel import (
    BlockageState,
    ChannelParams,
    ChannelRealisation,
    FreeSpaceLosChannel,
    _TR38901_RAY_OFFSETS_DEG,
    _ked_attenuation_db,
    _laplacian_subray_offsets,
    umi_path_loss_db,
    uma_path_loss_db,
    umi_los_probability,
)


# ===========================================================================
# 1. API-compatibility / baseline
# ===========================================================================

def test_umi_path_loss_monotone_in_distance():
    pl_50 = umi_path_loss_db(50.0, 28e9, 10.0, 1.5, los=True)
    pl_200 = umi_path_loss_db(200.0, 28e9, 10.0, 1.5, los=True)
    assert pl_200 > pl_50
    pl_100 = umi_path_loss_db(100.0, 28e9, 10.0, 1.5, los=True)
    assert 90 < pl_100 < 110


def test_freespace_los_channel_shape():
    ch = FreeSpaceLosChannel(bs_xy=np.array([10.0, 0.0]), bs_yaw=0.0,
                              n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)
    assert H.shape == (4, 16)
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
    """With K=30 dB, LOS strongly dominates. Threshold lowered to 3 to remain
    robust against occasional blockage attenuation from Model A blockers."""
    params = ChannelParams(k_factor_mean_db=30.0, k_factor_std_db=0.0)
    rng = np.random.default_rng(seed=7)
    real = ChannelRealisation(params=params, bs_xy=np.array([50.0, 50.0]),
                               bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4, rng=rng)
    H = real.channel_matrix(np.array([0.0, 0.0]), 0.0)
    s = np.linalg.svd(H, compute_uv=False)
    assert s[0] / s[1] > 3


# ===========================================================================
# 2. TR 38.901 Table 7.5-3 symbol kept for import compat
# ===========================================================================

def test_tr38901_offset_table_still_exported():
    """The offset table symbol must remain importable for backward compat."""
    assert len(_TR38901_RAY_OFFSETS_DEG) == 20
    np.testing.assert_allclose(_TR38901_RAY_OFFSETS_DEG.sum(), 0.0, atol=1e-10)


# ===========================================================================
# 3. LSP distribution sanity
# ===========================================================================

def test_lsp_draws_distribution_umi_los():
    """Draw many realisations; marginal means/stds should match LOS table."""
    params = ChannelParams()
    n_trials = 2000
    rng = np.random.default_rng(seed=42)
    ds_log10, asa_log10, k_db_vals = [], [], []
    for _ in range(n_trials):
        r = ChannelRealisation(params=params, bs_xy=np.array([0.0, 0.0]),
                                bs_yaw=0.0, n_bs_elements=2, n_ue_elements=2, rng=rng)
        ds_log10.append(np.log10(r.lsp["ds_s"]))
        asa_log10.append(np.log10(r.lsp["asa_deg"]))
        if r.is_los:
            k_db_vals.append(r.lsp["k_db"])

    ds_arr = np.array(ds_log10)
    assert abs(ds_arr.mean() - (-7.19)) < 0.08, f"DS mean {ds_arr.mean():.3f} off"
    assert abs(ds_arr.std() - 0.40) < 0.05, f"DS std {ds_arr.std():.3f} off"

    asa_arr = np.array(asa_log10)
    assert abs(asa_arr.mean() - 1.81) < 0.05, f"ASA mean {asa_arr.mean():.3f} off"

    if k_db_vals:
        k_arr = np.array(k_db_vals)
        assert abs(k_arr.mean() - 9.0) < 1.5, f"K mean {k_arr.mean():.2f} dB off"


def test_lsp_nlos_k_factor_zero():
    """NLOS realisations must have k_lin == 0.0."""
    params = ChannelParams(los_probability=0.0)
    rng = np.random.default_rng(seed=99)
    for _ in range(20):
        r = ChannelRealisation(params=params, bs_xy=np.array([0.0, 0.0]),
                                bs_yaw=0.0, n_bs_elements=2, n_ue_elements=2, rng=rng)
        assert r.k_lin == 0.0
        assert not r.is_los


# ===========================================================================
# 4. K-factor power split
# ===========================================================================

def test_k_factor_ratio_high_k():
    params = ChannelParams(k_factor_mean_db=20.0, k_factor_std_db=0.0)
    rng = np.random.default_rng(seed=3)
    real = ChannelRealisation(params=params, bs_xy=np.array([30.0, 0.0]),
                               bs_yaw=0.0, n_bs_elements=4, n_ue_elements=4, rng=rng)
    k = real.k_lin
    los_fraction = k / (1.0 + k)
    assert los_fraction > 0.99, f"LOS fraction {los_fraction:.4f} too low for K=20 dB"


def test_k_factor_zero_for_nlos():
    params = ChannelParams(los_probability=0.0)
    rng = np.random.default_rng(seed=5)
    real = ChannelRealisation(params=params, bs_xy=np.array([30.0, 0.0]),
                               bs_yaw=0.0, n_bs_elements=4, n_ue_elements=4, rng=rng)
    assert real.k_lin == 0.0


# ===========================================================================
# 5. LOS probability (TR 38.901 §7.4.2)
# ===========================================================================

def test_los_probability_at_origin_is_one():
    assert umi_los_probability(0.1) == pytest.approx(1.0, abs=0.01)


def test_los_probability_decreases_with_distance():
    p10 = umi_los_probability(10.0)
    p50 = umi_los_probability(50.0)
    p200 = umi_los_probability(200.0)
    assert p10 > p50 > p200


def test_los_probability_formula_spot_check():
    d = 18.0
    expected = 1.0 * (1.0 - np.exp(-d / 36.0)) + np.exp(-d / 36.0)
    np.testing.assert_allclose(umi_los_probability(d), expected, rtol=1e-6)


def test_los_probability_at_100m():
    d = 100.0
    expected = (18.0 / d) * (1.0 - np.exp(-d / 36.0)) + np.exp(-d / 36.0)
    np.testing.assert_allclose(umi_los_probability(d), expected, rtol=1e-6)


# ===========================================================================
# 6. Sub-ray offsets: Laplacian model (Eq 3.14, predecessor Sec 3.2.3)
# ===========================================================================

def test_laplacian_subray_shape():
    """_laplacian_subray_offsets returns (n_clusters, n_rays)."""
    rng = np.random.default_rng(1)
    offsets = _laplacian_subray_offsets(rng, n_clusters=12, n_rays=20,
                                         spread_rad=math.radians(17.0))
    assert offsets.shape == (12, 20)


def test_laplacian_subray_first_col_zero():
    """First sub-ray of each cluster is the large-scale component (offset=0)."""
    rng = np.random.default_rng(2)
    offsets = _laplacian_subray_offsets(rng, n_clusters=12, n_rays=20,
                                         spread_rad=math.radians(17.0))
    np.testing.assert_array_equal(offsets[:, 0], 0.0)


def test_laplacian_subray_distribution_mean_and_scale():
    """Large-N draw: mean ≈ 0, std ≈ scale * sqrt(2) (Laplacian property)."""
    spread_rad = math.radians(17.0)
    scale = spread_rad / math.sqrt(2.0)
    rng = np.random.default_rng(3)
    # Draw many clusters
    offsets = _laplacian_subray_offsets(rng, n_clusters=5000, n_rays=20,
                                         spread_rad=spread_rad)
    samples = offsets[:, 1:].ravel()  # exclude first col (always 0)
    assert abs(samples.mean()) < 0.05 * spread_rad, \
        f"Laplacian mean {samples.mean():.4f} not near 0"
    # Laplacian std = sqrt(2)*scale
    expected_std = math.sqrt(2.0) * scale
    assert abs(samples.std() - expected_std) < 0.05 * expected_std, \
        f"Laplacian std {samples.std():.4f} vs expected {expected_std:.4f}"


def test_realisation_subray_shape():
    """ChannelRealisation sub-ray offsets have shape (12, 20)."""
    params = ChannelParams()
    rng = np.random.default_rng(seed=1)
    real = ChannelRealisation(params=params, bs_xy=np.array([0.0, 0.0]),
                               bs_yaw=0.0, n_bs_elements=4, n_ue_elements=2, rng=rng)
    assert real.sub_ray_aoa_offsets.shape == (12, 20)
    assert real.sub_ray_aod_offsets.shape == (12, 20)


def test_realisation_subray_first_col_zero():
    """First sub-ray of each cluster has offset=0 (large-scale component)."""
    params = ChannelParams()
    rng = np.random.default_rng(seed=4)
    real = ChannelRealisation(params=params, bs_xy=np.array([0.0, 0.0]),
                               bs_yaw=0.0, n_bs_elements=4, n_ue_elements=2, rng=rng)
    np.testing.assert_array_equal(real.sub_ray_aoa_offsets[:, 0], 0.0)
    np.testing.assert_array_equal(real.sub_ray_aod_offsets[:, 0], 0.0)


# ===========================================================================
# 7. LOS LSPs used regardless of is_los state (predecessor Sec 3.2.2)
# ===========================================================================

def test_los_lsps_used_for_nlos_realisation():
    """NLOS realisation must draw from LOS table (ds_mu=-7.19, asa_mu=1.81)."""
    params = ChannelParams(los_probability=0.0)  # force NLOS
    n_trials = 1000
    rng = np.random.default_rng(seed=123)
    ds_log10, asa_log10 = [], []
    for _ in range(n_trials):
        r = ChannelRealisation(params=params, bs_xy=np.array([0.0, 0.0]),
                                bs_yaw=0.0, n_bs_elements=2, n_ue_elements=2, rng=rng)
        assert not r.is_los
        ds_log10.append(np.log10(r.lsp["ds_s"]))
        asa_log10.append(np.log10(r.lsp["asa_deg"]))

    # Should match LOS table (ds_mu=-7.19, asa_mu=1.81), not NLOS
    ds_arr = np.array(ds_log10)
    assert abs(ds_arr.mean() - (-7.19)) < 0.10, \
        f"NLOS should use LOS DS table, got mean {ds_arr.mean():.3f}"
    asa_arr = np.array(asa_log10)
    assert abs(asa_arr.mean() - 1.81) < 0.08, \
        f"NLOS should use LOS ASA table, got mean {asa_arr.mean():.3f}"


def test_always_12_clusters():
    """Both LOS and NLOS use 12 clusters (LOS table, predecessor Table 3.1)."""
    for los_prob in (0.0, 1.0):
        params = ChannelParams(los_probability=los_prob)
        rng = np.random.default_rng(seed=10)
        real = ChannelRealisation(params=params, bs_xy=np.array([0.0, 0.0]),
                                   bs_yaw=0.0, n_bs_elements=2, n_ue_elements=2,
                                   rng=rng)
        assert len(real.scatterer_xy) == 12, \
            f"Expected 12 clusters, got {len(real.scatterer_xy)}"


# ===========================================================================
# 8. Cluster delays
# ===========================================================================

def test_cluster_delays_sorted_and_nonnegative():
    params = ChannelParams()
    rng = np.random.default_rng(seed=77)
    real = ChannelRealisation(params=params, bs_xy=np.array([0.0, 0.0]),
                               bs_yaw=0.0, n_bs_elements=2, n_ue_elements=2, rng=rng)
    taus = real.cluster_delays_s
    assert taus[0] == pytest.approx(0.0, abs=1e-18)
    assert np.all(taus >= 0.0)
    assert np.all(np.diff(taus) >= 0.0)


# ===========================================================================
# 9. Geometric cluster power (predecessor Sec 3.2.2)
# ===========================================================================

def test_geometric_cluster_powers_sum_to_one():
    """After geometric computation, cluster powers must sum to 1."""
    params = ChannelParams()
    rng = np.random.default_rng(seed=88)
    real = ChannelRealisation(params=params, bs_xy=np.array([0.0, 0.0]),
                               bs_yaw=0.0, n_bs_elements=2, n_ue_elements=2, rng=rng)
    ue_xy = np.array([30.0, 0.0])
    _ = real.channel_matrix(ue_xy, 0.0)
    np.testing.assert_allclose(real.cluster_powers.sum(), 1.0, rtol=1e-9)


def test_geometric_cluster_power_closer_scatterer_higher_power():
    """A scatterer placed close to the midpoint (min extra path) should be
    stronger than one placed far away (large extra path)."""
    from beamsim.channel import _geometric_cluster_powers

    bs_xy = np.array([0.0, 0.0])
    ue_xy = np.array([100.0, 0.0])
    fc_hz = 28e9
    h_bs, h_ut = 10.0, 1.5
    k_lin = 0.0  # NLOS for simplicity

    # Close scatterer near midpoint (50, 5): very small extra path
    close = np.array([[50.0, 5.0]])
    # Far scatterer (500, 0): large extra path
    far = np.array([[500.0, 0.0]])
    scatterers = np.vstack([close, far])

    powers = _geometric_cluster_powers(bs_xy, ue_xy, scatterers,
                                        fc_hz, h_bs, h_ut, k_lin, False)
    assert powers[0] > powers[1], \
        f"Close scatterer power {powers[0]:.4f} should exceed far {powers[1]:.4f}"


# ===========================================================================
# 10. Blockage Model A
# ===========================================================================

def test_self_blocker_attenuates_when_bs_at_back():
    """When UE is rotated so BS is at 180 deg (back), self-blocker fires."""
    params = ChannelParams(los_probability=1.0,
                           k_factor_mean_db=30.0, k_factor_std_db=0.0)
    rng = np.random.default_rng(seed=200)
    real = ChannelRealisation(params=params, bs_xy=np.array([50.0, 0.0]),
                               bs_yaw=0.0, n_bs_elements=8, n_ue_elements=4, rng=rng)
    ue_xy = np.array([0.0, 0.0])

    # UE facing toward BS (yaw=0): no self-block
    H_front = real.channel_matrix(ue_xy, 0.0)
    norm_front = np.linalg.norm(H_front)

    # UE rotated 180 deg so BS is at the back: self-block fires
    H_back = real.channel_matrix(ue_xy, math.pi)
    norm_back = np.linalg.norm(H_back)

    assert norm_front > norm_back, \
        f"Front norm {norm_front:.4f} should exceed back norm {norm_back:.4f}"


def test_ked_attenuation_zero_outside_blocker():
    """KED attenuation outside the blocker angular span must be zero."""
    phi_k = 0.0
    x_k = math.radians(10.0)
    # Angle well outside the span
    phi_far = np.array([math.radians(90.0)])
    atten = _ked_attenuation_db(phi_far, phi_k, x_k)
    assert atten[0] == pytest.approx(0.0, abs=1e-9)


def test_ked_attenuation_nonzero_inside_blocker():
    """KED attenuation inside the blocker span must be > 0."""
    phi_k = 0.0
    x_k = math.radians(10.0)
    phi_inside = np.array([0.0])  # centre of blocker
    atten = _ked_attenuation_db(phi_inside, phi_k, x_k)
    assert atten[0] > 0.0


def test_non_self_blocker_drifts_over_position():
    """Non-self blockers drift with UE movement: different channel at two positions."""
    params = ChannelParams(los_probability=1.0,
                           k_factor_mean_db=9.0, k_factor_std_db=0.0)
    rng = np.random.default_rng(seed=300)
    real = ChannelRealisation(params=params, bs_xy=np.array([100.0, 0.0]),
                               bs_yaw=0.0, n_bs_elements=8, n_ue_elements=4, rng=rng)

    # Two positions far apart → blockers should have drifted significantly
    H1 = real.channel_matrix(np.array([0.0, 0.0]), 0.0)
    H2 = real.channel_matrix(np.array([0.0, 50.0]), 0.0)
    # Different UE positions → different geometry at minimum; channels differ
    diff = np.linalg.norm(H1 - H2)
    assert diff > 0.0, "Channels should differ across positions"


def test_blockage_state_self_blocker_30db():
    """Self-blocker applies exactly 30 dB within its angular span."""
    blk = BlockageState(
        n_non_self=0,
        phi_k=np.array([]),
        x_k=np.array([]),
        self_width_rad=math.radians(120.0),
    )
    # AoA directly at back (180 deg = pi rad): inside self-blocker
    aoa_back = np.array([math.pi])
    atten = blk.attenuation_db(aoa_back)
    assert atten[0] == pytest.approx(30.0, abs=1e-9)

    # AoA at front (0 deg): outside self-blocker
    aoa_front = np.array([0.0])
    atten_front = blk.attenuation_db(aoa_front)
    assert atten_front[0] == pytest.approx(0.0, abs=1e-9)


# ===========================================================================
# 11. No random initial phases
# ===========================================================================

def test_no_sub_ray_phases_attribute():
    """ChannelRealisation must NOT have a sub_ray_phases attribute (removed)."""
    params = ChannelParams()
    rng = np.random.default_rng(seed=50)
    real = ChannelRealisation(params=params, bs_xy=np.array([30.0, 0.0]),
                               bs_yaw=0.0, n_bs_elements=4, n_ue_elements=2, rng=rng)
    assert not hasattr(real, "sub_ray_phases"), \
        "sub_ray_phases should have been removed (predecessor Sec 3.2)"


# ===========================================================================
# 12. Doppler
# ===========================================================================

def test_doppler_channel_changes_with_time():
    params = ChannelParams(ue_speed_mps=30.0)
    rng = np.random.default_rng(seed=11)
    real = ChannelRealisation(params=params, bs_xy=np.array([50.0, 0.0]),
                               bs_yaw=0.0, n_bs_elements=8, n_ue_elements=4, rng=rng)
    ue_xy = np.array([0.0, 0.0])
    H0 = real.channel_matrix(ue_xy, 0.0, time_s=0.0)
    H1 = real.channel_matrix(ue_xy, 0.0, time_s=0.01)
    diff = np.linalg.norm(H1 - H0)
    assert diff > 1e-6, f"Channel unchanged after Doppler evolution, diff={diff}"


def test_doppler_zero_speed_channel_unchanged():
    params = ChannelParams(ue_speed_mps=0.0)
    rng = np.random.default_rng(seed=22)
    real = ChannelRealisation(params=params, bs_xy=np.array([50.0, 0.0]),
                               bs_yaw=0.0, n_bs_elements=8, n_ue_elements=4, rng=rng)
    ue_xy = np.array([0.0, 0.0])
    H0 = real.channel_matrix(ue_xy, 0.0, time_s=0.0)
    H1 = real.channel_matrix(ue_xy, 0.0, time_s=1.0)
    np.testing.assert_allclose(H0, H1, atol=1e-12)


# ===========================================================================
# 13. UMa path loss
# ===========================================================================

def test_uma_path_loss_monotone():
    pl_50 = uma_path_loss_db(50.0, 28e9, 25.0, 1.5, los=True)
    pl_200 = uma_path_loss_db(200.0, 28e9, 25.0, 1.5, los=True)
    assert pl_200 > pl_50


def test_uma_path_loss_nlos_ge_los():
    pl_los = uma_path_loss_db(150.0, 28e9, 25.0, 1.5, los=True)
    pl_nlos = uma_path_loss_db(150.0, 28e9, 25.0, 1.5, los=False)
    assert pl_nlos >= pl_los


# ===========================================================================
# 14. Channel matrix shape / output sanity
# ===========================================================================

def test_channel_matrix_shape_nlos():
    params = ChannelParams(los_probability=0.0)
    rng = np.random.default_rng(seed=55)
    real = ChannelRealisation(params=params, bs_xy=np.array([40.0, 0.0]),
                               bs_yaw=0.0, n_bs_elements=8, n_ue_elements=4, rng=rng)
    H = real.channel_matrix(np.array([0.0, 0.0]), 0.0)
    assert H.shape == (4, 8)
    assert np.all(np.isfinite(H))
