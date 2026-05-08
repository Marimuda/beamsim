"""Unit tests for beamsim.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from beamsim.metrics import (
    bootstrap_ci,
    bs_selection_loss,
    coverage_rate,
    mean_snr_db,
    output_snr_db,
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
    snr = np.full((4, 10), 20.0)   # (n_trials, n_steps)
    rate = coverage_rate(snr, gamma_th_db=10.0)
    np.testing.assert_array_equal(rate, np.ones(4))


def test_coverage_rate_all_below():
    snr = np.full((3, 8), 0.0)
    rate = coverage_rate(snr, gamma_th_db=10.0)
    np.testing.assert_array_equal(rate, np.zeros(3))


def test_coverage_rate_known_fraction():
    """5 out of 10 steps above threshold → rate = 0.5."""
    row = np.array([15.0] * 5 + [5.0] * 5)      # 5 above, 5 below
    snr = np.tile(row, (3, 1))                   # 3 identical trials
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
            f"{thresholds[i]:.1f} dB to {rates[i+1]:.4f} at {thresholds[i+1]:.1f} dB"
        )


# ---------------------------------------------------------------------------
# bs_selection_loss
# ---------------------------------------------------------------------------

def test_bs_selection_loss_zero_when_best_selected():
    """If selected BS always equals best BS, L_BS = 0."""
    rng = np.random.default_rng(1)
    snr0 = rng.uniform(5, 15, (10, 20))
    snr1 = rng.uniform(-5, 5, (10, 20))   # always lower than BS 0
    per_bs = {0: snr0, 1: snr1}
    selected = np.zeros((10, 20), dtype=int)   # always pick BS 0 (the best)
    loss = bs_selection_loss(per_bs, selected)
    assert loss == pytest.approx(0.0, abs=1e-9)


def test_bs_selection_loss_positive_when_suboptimal():
    """If selected BS is never the best, L_BS > 0."""
    snr0 = np.full((5, 10), 10.0)  # BS 0 is always 10 dB
    snr1 = np.full((5, 10), 0.0)   # BS 1 is always 0 dB
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
