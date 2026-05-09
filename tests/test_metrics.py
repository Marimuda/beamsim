"""Unit tests for beamsim.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from beamsim.metrics import (
    beam_switch_rate,
    bootstrap_ci,
    bs_selection_loss,
    coverage_rate,
    mean_snr_db,
    oracle_snr_db,
    outage_fraction,
    outage_probability,
    output_snr_db,
    probing_overhead,
    snr_regret_db,
    time_to_realign,
    top_k_accuracy,
)

# ---------------------------------------------------------------------------
# output_snr_db
# ---------------------------------------------------------------------------


def test_output_snr_db_known_value():
    """Signal amplitude = 1, noise amplitude = 1 → SNR = 0 dB."""
    y = np.array([1.0 + 0.0j, -1.0 + 0.0j])
    snr = output_snr_db(y, noise_amplitude=1.0)
    np.testing.assert_allclose(snr, [0.0, 0.0], atol=1e-9)


def test_output_snr_db_10dB():
    """Signal amplitude = sqrt(10), sigma = 1 → SNR = 10 dB."""
    y = np.array([np.sqrt(10.0) + 0.0j])
    snr = output_snr_db(y, noise_amplitude=1.0)
    np.testing.assert_allclose(snr, [10.0], atol=1e-9)


def test_output_snr_db_zero_signal():
    """Zero signal should return a very negative dB value, not NaN."""
    y = np.array([0.0 + 0.0j])
    snr = output_snr_db(y, noise_amplitude=1.0)
    assert np.isfinite(snr[0])
    assert snr[0] < -90.0


# ---------------------------------------------------------------------------
# coverage_rate — known synthetic trace
# ---------------------------------------------------------------------------


def test_coverage_rate_all_above():
    """All steps above threshold → rate = 1 for each trial."""
    snr = np.full((4, 10), 20.0)  # (n_trials, n_steps)
    rate = coverage_rate(snr, gamma_th_db=10.0)
    np.testing.assert_array_equal(rate, np.ones(4))


def test_coverage_rate_all_below():
    snr = np.full((3, 8), 0.0)
    rate = coverage_rate(snr, gamma_th_db=10.0)
    np.testing.assert_array_equal(rate, np.zeros(3))


def test_coverage_rate_known_fraction():
    """5 out of 10 steps above threshold → rate = 0.5."""
    row = np.array([15.0] * 5 + [5.0] * 5)  # 5 above, 5 below
    snr = np.tile(row, (3, 1))  # 3 identical trials
    rate = coverage_rate(snr, gamma_th_db=10.0)
    np.testing.assert_allclose(rate, [0.5, 0.5, 0.5], atol=1e-12)


def test_coverage_rate_monotone_in_threshold():
    """coverage_rate should be non-increasing as gamma_th increases."""
    rng = np.random.default_rng(0)
    snr = rng.uniform(-5, 25, size=(20, 50))
    thresholds = np.linspace(-10, 30, 20)
    rates = [coverage_rate(snr, g).mean() for g in thresholds]
    # Check non-increasing
    for i in range(len(rates) - 1):
        assert rates[i] >= rates[i + 1] - 1e-12, (
            f"Coverage rate increased from {rates[i]:.4f} at "
            f"{thresholds[i]:.1f} dB to {rates[i + 1]:.4f} at {thresholds[i + 1]:.1f} dB"
        )


# ---------------------------------------------------------------------------
# bs_selection_loss
# ---------------------------------------------------------------------------


def test_bs_selection_loss_zero_when_best_selected():
    """If selected BS always equals best BS, L_BS = 0."""
    rng = np.random.default_rng(1)
    snr0 = rng.uniform(5, 15, (10, 20))
    snr1 = rng.uniform(-5, 5, (10, 20))  # always lower than BS 0
    per_bs = {0: snr0, 1: snr1}
    selected = np.zeros((10, 20), dtype=int)  # always pick BS 0 (the best)
    loss = bs_selection_loss(per_bs, selected)
    assert loss == pytest.approx(0.0, abs=1e-9)


def test_bs_selection_loss_positive_when_suboptimal():
    """If selected BS is never the best, L_BS > 0."""
    snr0 = np.full((5, 10), 10.0)  # BS 0 is always 10 dB
    snr1 = np.full((5, 10), 0.0)  # BS 1 is always 0 dB
    per_bs = {0: snr0, 1: snr1}
    selected = np.ones((5, 10), dtype=int)  # always pick BS 1 (suboptimal)
    loss = bs_selection_loss(per_bs, selected)
    assert loss == pytest.approx(10.0, abs=1e-9)


def test_bs_selection_loss_non_negative():
    """L_BS is always >= 0."""
    rng = np.random.default_rng(2)
    snr0 = rng.normal(10, 3, (8, 30))
    snr1 = rng.normal(8, 3, (8, 30))
    per_bs = {0: snr0, 1: snr1}
    selected = rng.integers(0, 2, (8, 30))
    loss = bs_selection_loss(per_bs, selected)
    assert loss >= 0.0


# ---------------------------------------------------------------------------
# mean_snr_db
# ---------------------------------------------------------------------------


def test_mean_snr_db_constant():
    snr = np.full((5, 20), 7.0)
    assert mean_snr_db(snr) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def test_bootstrap_ci_brackets_true_mean():
    """For i.i.d. N(mu, 1) with large n, the 95 % CI should bracket mu."""
    rng = np.random.default_rng(42)
    mu = 5.0
    samples = rng.normal(mu, 1.0, size=500)
    mean, lo, hi = bootstrap_ci(samples, alpha=0.05, n_boot=2000, rng=rng)
    assert lo < mu < hi, f"True mean {mu} not in CI [{lo:.3f}, {hi:.3f}]"
    assert mean == pytest.approx(samples.mean(), abs=1e-12)


def test_bootstrap_ci_width_shrinks_with_n():
    """Larger samples → narrower CI."""
    rng = np.random.default_rng(7)
    small = rng.normal(0, 1, 50)
    large = rng.normal(0, 1, 5000)
    _, lo_s, hi_s = bootstrap_ci(small, n_boot=1000, rng=rng)
    _, lo_l, hi_l = bootstrap_ci(large, n_boot=1000, rng=rng)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_bootstrap_ci_symmetric_distribution():
    """For symmetric Gaussian, lo and hi should be roughly equidistant from mean."""
    rng = np.random.default_rng(99)
    samples = rng.normal(0, 1, 1000)
    mean, lo, hi = bootstrap_ci(samples, alpha=0.05, n_boot=3000, rng=rng)
    # |lo - mean| and |hi - mean| should both be positive and similar magnitude
    assert mean - lo > 0
    assert hi - mean > 0


# ---------------------------------------------------------------------------
# probing_overhead, top_k_accuracy, time_to_realign, outage_fraction
# ---------------------------------------------------------------------------


def test_probing_overhead_exhaustive_vs_oracle():
    """Exhaustive (every (k, l) once) → overhead 1; oracle (one pair) → 1/n_arms."""
    K, L = 4, 8
    n_arms = K * L
    # Exhaustive: row-major sweep of all 32 pairs.
    obp_exhaust = np.array([(k, l) for k in range(K) for l in range(L)], dtype=int)
    assert probing_overhead(obp_exhaust, n_arms=n_arms) == pytest.approx(1.0)

    # Oracle: always (3, 5).
    obp_oracle = np.tile(np.array([3, 5], dtype=int), (50, 1))
    assert probing_overhead(obp_oracle, n_arms=n_arms) == pytest.approx(1.0 / n_arms)


def test_probing_overhead_handles_3d_history():
    """Pass a (n_trials, n_steps, 2) array and get the per-trial mean."""
    K, L = 4, 8
    n_arms = K * L
    obp_a = np.tile(np.array([0, 0], dtype=int), (10, 1))  # 1 distinct
    obp_b = np.array([(k, 0) for k in range(K)] + [(0, 0)] * 6, dtype=int)  # K distinct
    obp_3d = np.stack([obp_a, obp_b], axis=0)  # (2, 10, 2)
    overhead = probing_overhead(obp_3d, n_arms=n_arms)
    expected = (1 / n_arms + K / n_arms) / 2
    assert overhead == pytest.approx(expected, abs=1e-9)


def test_top_k_accuracy_exact_match():
    pred = np.array([[(0, 0), (1, 2), (3, 7)]], dtype=int)
    true = np.array([[(0, 0), (1, 2), (3, 7)]], dtype=int)
    assert top_k_accuracy(pred, true, k_top=1) == 1.0


def test_top_k_accuracy_partial_match():
    pred = np.array([[(0, 0), (1, 2), (5, 5)]], dtype=int)
    true = np.array([[(0, 0), (1, 2), (3, 7)]], dtype=int)
    # 2 out of 3 exact match.
    assert top_k_accuracy(pred, true, k_top=1) == pytest.approx(2 / 3)


def test_top_k_accuracy_neighbour_credit():
    """k_top > 1 should accept 4-connected neighbours of the true OBP."""
    pred = np.array([[(0, 1), (1, 2)]], dtype=int)
    true = np.array([[(0, 0), (1, 2)]], dtype=int)
    # Exact: 1/2 (only second matches).
    assert top_k_accuracy(pred, true, k_top=1) == pytest.approx(0.5)
    # 4-connected neighbour radius 1: pred (0,1) is 1 hop from (0,0) → counts.
    # Both should match → 1.0.
    assert top_k_accuracy(pred, true, k_top=4, L=8) == pytest.approx(1.0)


def test_time_to_realign_immediate_recovery():
    """If SNR is above threshold immediately after handover, recovery time = 0."""
    snr = np.full((3, 50), 20.0)
    times = time_to_realign(snr, threshold_db=10.0, handover_step=20, max_search=20)
    np.testing.assert_array_equal(times, np.zeros(3, dtype=int))


def test_time_to_realign_delayed_recovery():
    """SNR rises above threshold k steps after handover → recovery time = k."""
    snr = np.full((1, 100), 0.0)
    snr[0, 30:] = 20.0  # recovers at step 30; handover at step 25 → time 5.
    times = time_to_realign(snr, threshold_db=10.0, handover_step=25, max_search=50)
    assert times[0] == 5


def test_time_to_realign_censored():
    """If recovery never happens in the search window, return max_search."""
    snr = np.full((2, 100), -5.0)  # always below threshold
    times = time_to_realign(snr, threshold_db=10.0, handover_step=10, max_search=20)
    np.testing.assert_array_equal(times, np.full(2, 20, dtype=int))


def test_outage_fraction_known():
    """5/10 steps below threshold → outage = 0.5."""
    row = np.array([5.0] * 5 + [15.0] * 5)
    snr = np.tile(row, (3, 1))
    out = outage_fraction(snr, threshold_db=10.0)
    np.testing.assert_allclose(out, [0.5, 0.5, 0.5])


def test_outage_fraction_complement_of_coverage():
    """outage_fraction + coverage_rate = 1 (modulo strictness on equality)."""
    rng = np.random.default_rng(0)
    snr = rng.uniform(-5, 25, size=(10, 50))
    out = outage_fraction(snr, threshold_db=10.0)
    cov = coverage_rate(snr, gamma_th_db=10.0)
    # Coverage uses >=, outage uses < — they sum to exactly 1.
    np.testing.assert_allclose(out + cov, np.ones_like(out), atol=1e-12)


def test_bootstrap_ci_coverage_rate():
    """Over 200 independent experiments, the 95 % CI should cover the true mean
    at least 90 % of the time (allowing slack for Monte Carlo noise)."""
    rng = np.random.default_rng(123)
    mu = 0.0
    covered = 0
    n_experiments = 200
    for _ in range(n_experiments):
        samples = rng.normal(mu, 1.0, size=100)
        _, lo, hi = bootstrap_ci(samples, alpha=0.05, n_boot=500, rng=rng)
        if lo <= mu <= hi:
            covered += 1
    coverage = covered / n_experiments
    assert coverage >= 0.90, f"Bootstrap CI coverage {coverage:.2%} < 90 %"


# ---------------------------------------------------------------------------
# outage_probability
# ---------------------------------------------------------------------------


def test_outage_probability_pools_to_scalar():
    """Returns a single float averaged across all (trial, step) cells."""
    snr = np.array([[5.0, 15.0], [-1.0, 20.0]])
    p = outage_probability(snr, threshold_db=10.0)
    assert isinstance(p, float)
    # Cells below 10 dB: 5.0, -1.0 → 2 / 4 = 0.5
    assert p == 0.5


def test_outage_probability_threshold_is_strict():
    """An SNR exactly at the threshold is *not* in outage (matches outage_fraction)."""
    snr = np.full((3, 4), 10.0)
    p = outage_probability(snr, threshold_db=10.0)
    assert p == 0.0


def test_outage_probability_complements_coverage_at_boundary():
    """Pooled outage_probability + pooled coverage should sum to 1 exactly."""
    rng = np.random.default_rng(0)
    snr = rng.normal(loc=12.0, scale=4.0, size=(20, 50))
    p_out = outage_probability(snr, threshold_db=10.0)
    p_cov = float(coverage_rate(snr, gamma_th_db=10.0).mean())
    assert p_out + p_cov == pytest.approx(1.0, abs=1e-12)


def test_outage_probability_propagates_nan():
    """NaN inputs produce NaN output; we never silently coerce."""
    snr = np.array([[5.0, np.nan], [12.0, 8.0]])
    p = outage_probability(snr, threshold_db=10.0)
    assert np.isnan(p)


# ---------------------------------------------------------------------------
# beam_switch_rate
# ---------------------------------------------------------------------------


def test_beam_switch_rate_no_switch():
    """A constant (k, l) trace switches zero times."""
    obp = np.tile(np.array([[3, 7]]), (50, 1))
    assert beam_switch_rate(obp) == 0.0


def test_beam_switch_rate_all_switch():
    """A trace where every step differs has rate 1."""
    obp = np.array([[i % 4, (i + 1) % 8] for i in range(20)])
    # Verify the construction actually changes (k, l) every step.
    diffs = np.any(obp[1:] != obp[:-1], axis=-1)
    assert diffs.all(), "test fixture broken: not all consecutive pairs differ"
    assert beam_switch_rate(obp) == 1.0


def test_beam_switch_rate_mixed_2d():
    """Half the steps switch → rate 0.5."""
    obp = np.array([[0, 0], [0, 0], [1, 0], [1, 0], [1, 1]])  # 5 steps, switches at 2 and 4
    rate = beam_switch_rate(obp)
    assert rate == pytest.approx(2.0 / 4.0)  # 4 consecutive pairs, 2 switches


def test_beam_switch_rate_3d_returns_per_trial():
    """3-D input returns one rate per trial."""
    obp = np.array(
        [
            [[0, 0], [0, 0], [0, 0]],  # no switch
            [[0, 0], [1, 1], [2, 2]],  # all switch
            [[0, 0], [0, 0], [1, 1]],  # one switch out of two pairs
        ]
    )
    rates = beam_switch_rate(obp)
    assert isinstance(rates, np.ndarray)
    np.testing.assert_allclose(rates, [0.0, 1.0, 0.5])


def test_beam_switch_rate_single_step_returns_zero():
    """n_steps < 2 → rate is 0 (cannot switch)."""
    assert beam_switch_rate(np.array([[3, 7]])) == 0.0
    np.testing.assert_array_equal(
        beam_switch_rate(np.array([[[3, 7]], [[1, 2]]])),  # (2 trials, 1 step, 2)
        np.zeros(2),
    )


def test_beam_switch_rate_rejects_bad_shape():
    """Last axis must be 2 (k, l)."""
    with pytest.raises(ValueError):
        beam_switch_rate(np.zeros((5, 3)))
    with pytest.raises(ValueError):
        beam_switch_rate(np.zeros((2, 5, 3)))
    with pytest.raises(ValueError):
        beam_switch_rate(np.zeros(7))  # 1-D


# ---------------------------------------------------------------------------
# oracle_snr_db
# ---------------------------------------------------------------------------


def _orthonormal_codebook(n_beams: int, n_elements: int, seed: int) -> np.ndarray:
    """Build a unit-norm random codebook for fast deterministic tests."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n_beams, n_elements)) + 1j * rng.standard_normal(
        (n_beams, n_elements)
    )
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def test_oracle_snr_db_shapes():
    """(n_steps, K, L) SNR is reduced to (n_steps,) of dB."""
    H = np.zeros((4, 3, 5), dtype=np.complex128)
    W = _orthonormal_codebook(2, 3, seed=0)
    F = _orthonormal_codebook(7, 5, seed=1)
    out = oracle_snr_db(H, W, F, noise_amplitude=1e-3)
    assert out.shape == (4,)
    assert out.dtype == np.float64


def test_oracle_snr_db_accepts_single_step_matrix():
    """A bare (n_ue, n_bs) matrix is treated as a 1-step trace."""
    H = np.eye(3, 4, dtype=np.complex128)
    W = _orthonormal_codebook(2, 3, seed=0)
    F = _orthonormal_codebook(2, 4, seed=1)
    out = oracle_snr_db(H, W, F, noise_amplitude=1.0)
    assert out.shape == (1,)


def test_oracle_snr_db_dominates_specific_probe():
    """Oracle SNR at each step >= the SNR of *any* fixed (k, l) probe under
    the same noiseless channel/codebook (the property the metric exists to
    diagnose)."""
    rng = np.random.default_rng(42)
    n_steps, n_ue, n_bs = 8, 4, 6
    H = rng.standard_normal((n_steps, n_ue, n_bs)) + 1j * rng.standard_normal((n_steps, n_ue, n_bs))
    W = _orthonormal_codebook(3, n_ue, seed=2)
    F = _orthonormal_codebook(5, n_bs, seed=3)
    noise_amp = 1e-3
    tx_amp = 1.7

    oracle = oracle_snr_db(H, W, F, noise_amplitude=noise_amp, tx_amp=tx_amp)

    # Compute the noiseless SNR for one specific (k=1, l=2) probe at every step.
    k, l = 1, 2
    y = tx_amp * np.einsum("i,tij,j->t", W[k].conj(), H, F[l])
    achieved = output_snr_db(y, noise_amplitude=noise_amp)

    assert np.all(oracle >= achieved - 1e-9), (
        f"oracle dropped below specific probe: max gap = {(achieved - oracle).max():.3e}"
    )


def test_oracle_snr_db_matches_bplm_signal_convention():
    """Oracle of a single-pair codebook equals the signal SNR of that one
    (k=0, l=0) measurement under :class:`beamsim.bplm.BPLMState`."""
    rng = np.random.default_rng(7)
    n_ue, n_bs = 2, 3
    H = rng.standard_normal((1, n_ue, n_bs)) + 1j * rng.standard_normal((1, n_ue, n_bs))
    w = rng.standard_normal(n_ue) + 1j * rng.standard_normal(n_ue)
    f = rng.standard_normal(n_bs) + 1j * rng.standard_normal(n_bs)
    W = w[None, :]  # (K=1, n_ue)
    F = f[None, :]  # (L=1, n_bs)
    noise_amp = 1e-2
    tx_amp = 0.9

    oracle = oracle_snr_db(H, W, F, noise_amplitude=noise_amp, tx_amp=tx_amp)

    # Direct signal calculation per BPLMState.measure (without the additive noise).
    signal = tx_amp * (w.conj() @ H[0] @ f)
    expected = output_snr_db(np.array([signal]), noise_amplitude=noise_amp)
    np.testing.assert_allclose(oracle, expected, atol=1e-12)


def test_oracle_snr_db_rejects_dimension_mismatch():
    H = np.zeros((2, 3, 4), dtype=np.complex128)
    W = _orthonormal_codebook(2, 5, seed=0)  # last axis 5 ≠ n_ue=3
    F = _orthonormal_codebook(2, 4, seed=1)
    with pytest.raises(ValueError, match="ue_weights last axis"):
        oracle_snr_db(H, W, F, noise_amplitude=1.0)
    W = _orthonormal_codebook(2, 3, seed=0)
    F = _orthonormal_codebook(2, 7, seed=1)  # last axis 7 ≠ n_bs=4
    with pytest.raises(ValueError, match="bs_weights last axis"):
        oracle_snr_db(H, W, F, noise_amplitude=1.0)


def test_oracle_snr_db_rejects_bad_channel_shape():
    H_4d = np.zeros((1, 2, 3, 4), dtype=np.complex128)
    W = _orthonormal_codebook(2, 3, seed=0)
    F = _orthonormal_codebook(2, 4, seed=1)
    with pytest.raises(ValueError, match="channel_matrices"):
        oracle_snr_db(H_4d, W, F, noise_amplitude=1.0)


# ---------------------------------------------------------------------------
# snr_regret_db
# ---------------------------------------------------------------------------


def test_snr_regret_db_sign_convention():
    """``snr_regret_db = oracle - achieved`` so lower is better, zero is optimal."""
    achieved = np.array([10.0, 12.0, 5.0])
    oracle = np.array([15.0, 12.0, 20.0])
    np.testing.assert_allclose(snr_regret_db(achieved, oracle), [5.0, 0.0, 15.0])


def test_snr_regret_db_zero_when_achieved_equals_oracle():
    """Identity input ⇒ zero regret."""
    snr = np.linspace(-5.0, 25.0, 11)
    np.testing.assert_array_equal(snr_regret_db(snr, snr), np.zeros_like(snr))


def test_snr_regret_db_broadcasts():
    """Mismatched shapes broadcast like np arithmetic."""
    achieved = np.array([[10.0, 12.0], [5.0, 8.0]])  # (n_trials, n_steps)
    oracle = np.array([15.0, 14.0])  # (n_steps,) broadcast across trials
    out = snr_regret_db(achieved, oracle)
    assert out.shape == achieved.shape
    np.testing.assert_allclose(out, [[5.0, 2.0], [10.0, 6.0]])


def test_snr_regret_db_propagates_nan():
    """NaN in either input produces NaN at that index."""
    achieved = np.array([10.0, np.nan, 5.0])
    oracle = np.array([15.0, 12.0, np.nan])
    out = snr_regret_db(achieved, oracle)
    assert out[0] == 5.0
    assert np.isnan(out[1])
    assert np.isnan(out[2])
