"""Unit tests for beamsim.link_budget."""

from __future__ import annotations

import math

import pytest

from beamsim.channel import umi_path_loss_db
from beamsim.link_budget import tx_amp_for_snr_db


def test_tx_amp_known_value():
    """Verify tx_amp_for_snr_db against a hand-computed reference.

    Parameters: target=10 dB, distance=100 m, fc=28 GHz, h_bs=10 m,
    h_ut=1.5 m, noise_amp=1e-3, n_ue=4, n_bs=16, LOS.

    Expected: sigma_n * sqrt(n_ue * n_bs * target_lin) / pl_lin
    """
    target_db = 10.0
    distance_m = 100.0
    fc_hz = 28e9
    h_bs = 10.0
    h_ut = 1.5
    noise_amp = 1e-3
    n_ue = 4
    n_bs = 16

    pl_db = umi_path_loss_db(distance_m, fc_hz, h_bs, h_ut, los=True)
    pl_lin = 10.0 ** (-pl_db / 20.0)
    target_lin = 10.0 ** (target_db / 10.0)
    expected = float(noise_amp * math.sqrt(n_ue * n_bs * target_lin) / pl_lin)

    result = tx_amp_for_snr_db(target_db, distance_m, fc_hz, h_bs, h_ut,
                                noise_amp, n_ue, n_bs)

    assert result == pytest.approx(expected, rel=1e-9), (
        f"tx_amp_for_snr_db returned {result:.6e}, expected {expected:.6e}"
    )


def test_tx_amp_increases_with_target_snr():
    """Higher target SNR must require higher tx amplitude."""
    kwargs = dict(distance_m=100.0, fc_hz=28e9, h_bs=10.0, h_ut=1.5,
                  noise_amp=1e-3, n_ue=4, n_bs=16)
    amp_low = tx_amp_for_snr_db(0.0, **kwargs)
    amp_high = tx_amp_for_snr_db(20.0, **kwargs)
    assert amp_high > amp_low


def test_tx_amp_increases_with_distance():
    """Greater path loss at longer distance must require larger tx amplitude."""
    kwargs = dict(target_db=10.0, fc_hz=28e9, h_bs=10.0, h_ut=1.5,
                  noise_amp=1e-3, n_ue=4, n_bs=16)
    amp_near = tx_amp_for_snr_db(distance_m=50.0, **kwargs)
    amp_far = tx_amp_for_snr_db(distance_m=200.0, **kwargs)
    assert amp_far > amp_near
