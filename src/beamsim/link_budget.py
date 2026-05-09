"""Shared link-budget utilities for tx-amplitude calibration.

Single source-of-truth for the post-calibration formula used across all
experiment scripts.  Eliminates the four-site duplication identified in the
audit (exp_snr_sweep, exp_alpha_sweep, exp_handover, run.py).
"""

from __future__ import annotations

import math

from beamsim.channel import umi_path_loss_db


def tx_amp_for_snr_db(
    target_db: float,
    distance_m: float,
    fc_hz: float,
    h_bs: float,
    h_ut: float,
    noise_amp: float,
    n_ue: int,
    n_bs: int,
    *,
    los: bool = True,
) -> float:
    """Return tx_amp such that the per-element input SNR equals *target_db*.

    Derivation (Eq 5.9 of predecessor report, Sec 5.3.3):

        SNR_in = (tx_amp * pl_lin)^2 * N_r * N_t / sigma_n^2 = target_lin

    Solving for tx_amp::

        tx_amp = sigma_n * sqrt(N_r * N_t * target_lin) / pl_lin

    where ``pl_lin = 10^(-pl_db/20)`` is the one-way voltage path-loss factor
    and ``sqrt(N_r * N_t)`` accounts for the array gain baked into unit-norm
    codewords.

    Args:
        target_db: Desired per-element input SNR in dB.
        distance_m: Reference distance (m) for path-loss evaluation.
        fc_hz: Carrier frequency (Hz).
        h_bs: BS antenna height (m).
        h_ut: UE antenna height (m).
        noise_amp: Noise amplitude sigma_n.
        n_ue: Number of UE antenna elements (N_r).
        n_bs: Number of BS antenna elements (N_t).
        los: Whether to use the LOS path-loss model (default True).

    Returns:
        Transmit amplitude scalar (float).
    """
    pl_db = umi_path_loss_db(distance_m, fc_hz, h_bs, h_ut, los=los)
    pl_lin = 10.0 ** (-pl_db / 20.0)
    target_lin = 10.0 ** (target_db / 10.0)
    return float(noise_amp * math.sqrt(n_ue * n_bs * target_lin) / pl_lin)
