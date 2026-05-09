"""Pure-numpy metric functions over per-trial output-SNR traces.

All functions operate on arrays of shape (n_trials, n_steps) or 1-D slices
thereof.  Nothing here imports beamsim internals — the module is intentionally
self-contained so it can be tested and reused independently of the runner.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.stats import bootstrap as _scipy_bootstrap


def output_snr_db(
    trace_complex_y: NDArray[np.complex128],
    noise_amplitude: float,
) -> NDArray[np.float64]:
    """Convert |y|^2 / sigma^2 to dB over a 1-D trace of occasions.

    Parameters
    ----------
    trace_complex_y:
        Complex received samples, shape ``(n_occasions,)``.
    noise_amplitude:
        ``sigma_n`` (amplitude, NOT power).  The noise power is
        ``sigma_n^2``.

    Returns
    -------
    SNR in dB, same shape as *trace_complex_y*.
    """
    sigma_sq = noise_amplitude**2
    snr_lin = (np.abs(trace_complex_y) ** 2) / sigma_sq
    # Clip to avoid log(0); anything below -100 dB is below noise floor.
    return 10.0 * np.log10(np.maximum(snr_lin, 1e-10))


def mean_snr_db(snr_db_per_trial: NDArray[np.float64]) -> float:
    """Mean over trials of mean over occasions, returned in dB.

    Parameters
    ----------
    snr_db_per_trial:
        Shape ``(n_trials, n_steps)``.
    """
    return float(np.mean(snr_db_per_trial))


def coverage_rate(
    snr_db_per_trial: NDArray[np.float64],
    gamma_th_db: float,
) -> NDArray[np.float64]:
    """Per-trial fraction of occasions where SNR_dB >= gamma_th_db.

    Parameters
    ----------
    snr_db_per_trial:
        Shape ``(n_trials, n_steps)``.
    gamma_th_db:
        Coverage threshold in dB.

    Returns
    -------
    Shape ``(n_trials,)`` with values in [0, 1].
    """
    covered = snr_db_per_trial >= gamma_th_db
    return covered.mean(axis=1)


def bs_selection_loss(
    per_bs_snr_db: dict[int, NDArray[np.float64]],
    selected_bs: NDArray[np.int_],
) -> float:
    """L_BS in dB: E[10 * log10(P_best / P_selected)] over (trials, occasions).

    Parameters
    ----------
    per_bs_snr_db:
        Mapping ``bs_index -> (n_trials, n_steps)`` SNR arrays in dB.
    selected_bs:
        Integer array of shape ``(n_trials, n_steps)`` with the BS index
        chosen by the algorithm at each occasion.

    Returns
    -------
    Mean selection loss in dB (>= 0 by construction; 0 means always optimal).
    """
    bs_indices = list(per_bs_snr_db.keys())
    # Stack into (n_bs, n_trials, n_steps)
    stacked = np.stack([per_bs_snr_db[b] for b in bs_indices], axis=0)
    best_snr_db = stacked.max(axis=0)  # (n_trials, n_steps)

    # Gather the SNR of the selected BS at each occasion.
    # Convert selected_bs values to positions in bs_indices list.
    idx_map = {b: i for i, b in enumerate(bs_indices)}
    sel_pos = np.vectorize(idx_map.__getitem__)(selected_bs)  # (n_trials, n_steps)
    sel_snr_db = stacked[
        sel_pos, np.arange(sel_pos.shape[0])[:, None], np.arange(sel_pos.shape[1])[None, :]
    ]

    loss_db = best_snr_db - sel_snr_db  # always >= 0 (dB difference)
    return float(np.mean(loss_db))


def bootstrap_ci(
    samples: NDArray[np.float64],
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for the mean of *samples*.

    Parameters
    ----------
    samples:
        1-D array of observations.
    alpha:
        Coverage miss rate; the CI covers (1 - alpha) * 100 % of the
        bootstrap distribution.
    n_boot:
        Number of bootstrap resamples.
    rng:
        Optional random generator for reproducibility.  A fresh one is
        created when *None*.

    Returns
    -------
    ``(mean, lo, hi)`` where *lo* and *hi* are the ``alpha/2`` and
    ``1 - alpha/2`` percentiles of the bootstrap mean distribution.
    """
    if rng is None:
        rng = np.random.default_rng()
    mean = float(samples.mean())
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _scipy_bootstrap(
                (samples,),
                statistic=np.mean,
                n_resamples=n_boot,
                confidence_level=1.0 - alpha,
                method="BCa",
                random_state=rng,
            )
        lo = float(res.confidence_interval.low)
        hi = float(res.confidence_interval.high)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError("BCa degenerate")
    except Exception:
        lo = hi = mean
    return mean, lo, hi
