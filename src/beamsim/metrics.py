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


def probing_overhead(
    obp_history: NDArray[np.int_],
    n_arms: int | None = None,
) -> float:
    """Distinct-arm probing overhead per trial, normalised to [0, 1].

    The 3GPP TR 38.843 evaluation framework asks for an "overhead" metric
    expressed as the fraction of beams probed relative to the full
    codebook.  Algorithms that exhaustively scan have overhead ≈ 1; an
    oracle that picks the right beam every step has overhead ≈ 1 / n_arms.

    Parameters
    ----------
    obp_history:
        Either a (n_steps, 2) array of (k, l) OBP indices for one trial,
        OR a (n_trials, n_steps, 2) array — the function squeezes the
        leading dimension if present and returns the *per-trial* mean
        overhead.
    n_arms:
        Total number of beam pairs (K * L).  If None, inferred from
        the maximum (k, l) pair seen in the history.
    """
    arr = np.asarray(obp_history)
    if arr.ndim == 3:
        return float(np.mean([probing_overhead(arr[t], n_arms) for t in range(arr.shape[0])]))
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected (n_steps, 2) or (n_trials, n_steps, 2), got shape {arr.shape}")
    flat_pairs = arr[:, 0] * (arr[:, 1].max() + 1) + arr[:, 1]
    distinct = len(np.unique(flat_pairs))
    if n_arms is None:
        n_arms = (int(arr[:, 0].max()) + 1) * (int(arr[:, 1].max()) + 1)
    return distinct / max(n_arms, 1)


def top_k_accuracy(
    obp_pred: NDArray[np.int_],
    obp_true: NDArray[np.int_],
    k_top: int = 1,
    L: int | None = None,
) -> float:
    """Top-k OBP-match accuracy across (trials, occasions).

    Parameters
    ----------
    obp_pred:
        (n_trials, n_steps, 2) OBP indices produced by the algorithm.
    obp_true:
        (n_trials, n_steps, 2) ground-truth OBP from a Perfect / oracle
        baseline (or Exhaustive on a noiseless channel).
    k_top:
        ``k_top=1`` is exact match.  ``k_top=4`` accepts any 4-connected
        neighbour of the true OBP — useful when one beam off is acceptable.
    L:
        BS codebook size, used to compute neighbour membership.  If None,
        falls back to exact match (k_top is ignored).

    Returns
    -------
    Fraction in [0, 1].
    """
    pred = np.asarray(obp_pred)
    true = np.asarray(obp_true)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs true {true.shape}")
    if pred.ndim == 2:
        pred = pred[None, :, :]
        true = true[None, :, :]
    if k_top == 1 or L is None:
        return float(np.mean(np.all(pred == true, axis=-1)))
    # Top-k for k>1: accept any 4-connected neighbour at Manhattan <= radius.
    diff = np.abs(pred - true)
    return float(np.mean(diff.sum(axis=-1) <= 1))


def time_to_realign(
    snr_db: NDArray[np.float64],
    threshold_db: float,
    handover_step: int,
    max_search: int = 200,
) -> NDArray[np.int_]:
    """Steps until SNR exceeds ``threshold_db`` after a handover trigger.

    Parameters
    ----------
    snr_db:
        (n_trials, n_steps) SNR-in-dB trace.
    threshold_db:
        Realignment threshold in dB; the recovery is considered complete
        the first step at which ``snr_db >= threshold_db`` after the
        handover.
    handover_step:
        Index of the handover event.  Steps before this are ignored.
    max_search:
        If recovery never occurs within ``handover_step + max_search``
        steps, the trial is recorded as ``max_search`` (capped, censored).

    Returns
    -------
    (n_trials,) integer array of recovery times in steps.
    """
    snr = np.asarray(snr_db)
    if snr.ndim != 2:
        raise ValueError(f"expected 2-D snr_db, got shape {snr.shape}")
    n_trials = snr.shape[0]
    out = np.full(n_trials, max_search, dtype=np.int_)
    end = min(handover_step + max_search, snr.shape[1])
    for t in range(n_trials):
        post = snr[t, handover_step:end]
        idx = np.argmax(post >= threshold_db)
        # argmax returns 0 if no True; check whether [0] is actually >= threshold.
        if post.size > 0 and post[idx] >= threshold_db:
            out[t] = int(idx)
        else:
            out[t] = max_search
    return out


def outage_fraction(
    snr_db: NDArray[np.float64],
    threshold_db: float,
) -> NDArray[np.float64]:
    """Per-trial fraction of occasions with SNR below ``threshold_db``.

    Parameters
    ----------
    snr_db:
        (n_trials, n_steps) SNR-in-dB trace.
    threshold_db:
        Outage threshold in dB.  Anything strictly below this is in outage.

    Returns
    -------
    Shape (n_trials,), values in [0, 1].
    """
    snr = np.asarray(snr_db)
    return (snr < threshold_db).mean(axis=1)


def outage_probability(
    snr_db: NDArray[np.float64],
    threshold_db: float,
) -> float:
    """Population outage probability ``Pr(SNR_dB < threshold_db)``.

    Pools across every trial and step in *snr_db* and returns a single
    scalar in ``[0, 1]``.  For the per-trial breakdown, use
    :func:`outage_fraction` and reduce at the call site.

    The threshold is **strict**: a sample exactly at ``threshold_db`` is
    *not* in outage, matching the convention of :func:`outage_fraction`
    and :func:`coverage_rate` (which are complements at the boundary).

    NaN samples propagate: if any element of *snr_db* is NaN the result
    is NaN, on the principle that a population statistic over partially
    missing data is itself undefined.  Use ``np.nan_to_num`` or filter
    explicitly at the call site if you want to skip NaN.
    """
    snr = np.asarray(snr_db, dtype=np.float64)
    if np.isnan(snr).any():
        return float("nan")
    return float(np.mean(snr < threshold_db))


def beam_switch_rate(
    obp_history: NDArray[np.int_],
) -> NDArray[np.float64] | float:
    """Fraction of consecutive step pairs at which the chosen beam pair changes.

    Parameters
    ----------
    obp_history:
        Either ``(n_steps, 2)`` for a single trial — returns a scalar — or
        ``(n_trials, n_steps, 2)`` — returns a per-trial array of shape
        ``(n_trials,)``.  Indices along the last axis are ``(k, l)``.

    Returns
    -------
    Switch rate(s) in ``[0, 1]``: ``0`` means the algorithm never changed
    its (k, l) selection, ``1`` means every step pair differed.  When
    ``n_steps < 2`` the rate is defined as ``0`` (the algorithm could
    not have switched).

    Notes
    -----
    A "switch" is any change in *either* the UE index ``k`` or the BS
    index ``l`` between consecutive steps.  Pool the per-trial array
    with ``.mean()`` at the call site for the cross-trial mean.
    """
    arr = np.asarray(obp_history)
    if arr.ndim == 2:
        if arr.shape[1] != 2:
            raise ValueError(
                f"expected (n_steps, 2) or (n_trials, n_steps, 2), got shape {arr.shape}"
            )
        if arr.shape[0] < 2:
            return 0.0
        diffs = np.any(arr[1:] != arr[:-1], axis=-1)
        return float(diffs.mean())
    if arr.ndim == 3:
        if arr.shape[2] != 2:
            raise ValueError(
                f"expected (n_steps, 2) or (n_trials, n_steps, 2), got shape {arr.shape}"
            )
        if arr.shape[1] < 2:
            return np.zeros(arr.shape[0], dtype=np.float64)
        diffs = np.any(arr[:, 1:] != arr[:, :-1], axis=-1)
        return diffs.mean(axis=1).astype(np.float64)
    raise ValueError(f"expected 2-D or 3-D obp_history, got shape {arr.shape}")


def oracle_snr_db(
    channel_matrices: NDArray[np.complex128],
    ue_weights: NDArray[np.complex128],
    bs_weights: NDArray[np.complex128],
    noise_amplitude: float,
    tx_amp: float = 1.0,
) -> NDArray[np.float64]:
    """Best achievable SNR (dB) over the *simulated codebook* at each step.

    For each step ``t``, returns

    .. math::

        \\max_{k,l}\\;
            10\\log_{10}\\!\\left(
                \\frac{|\\,\\text{tx\\_amp}\\;\\bm w_k^H\\,\\bm H_t\\,\\bm f_l\\,|^2}
                       {\\sigma_n^2}
            \\right),

    where ``w_k = ue_weights[k]`` is the UE combining vector and
    ``f_l = bs_weights[l]`` is the BS precoding vector.  The combiner is
    applied as ``w.conj() @ H @ f`` so the convention matches
    :class:`beamsim.bplm.BPLMState.measure`.

    Parameters
    ----------
    channel_matrices:
        Per-step channel matrices, shape ``(n_steps, n_ue_elements,
        n_bs_elements)``.  May also be a single ``(n_ue_elements,
        n_bs_elements)`` matrix — the function adds a leading axis.
    ue_weights:
        UE codebook entries stacked as rows, shape ``(K, n_ue_elements)``.
        For :class:`beamsim.codebook.Codebook` instances this is the
        ``.matrix`` attribute.
    bs_weights:
        BS codebook entries stacked as rows, shape ``(L, n_bs_elements)``.
    noise_amplitude:
        ``sigma_n`` (amplitude, NOT power); the noise power is
        ``sigma_n ** 2``.
    tx_amp:
        Transmit-amplitude calibration applied by the runner (defaults to
        ``1.0`` to match :class:`beamsim.bplm.BPLMState`).

    Returns
    -------
    Shape ``(n_steps,)`` of oracle SNR in dB.

    Notes
    -----
    This is the *codebook* oracle: the strongest SNR a measurement-policy
    algorithm could ever report given the same codebook and the same
    channel realisation, evaluated **noiselessly**.  It is *not* a
    Shannon-capacity oracle and *not* a deployable policy — it requires
    measuring every (k, l) pair at every step, which defeats the point
    of beam alignment.  Use it as the comparator in
    :func:`snr_regret_db`.
    """
    H = np.asarray(channel_matrices, dtype=np.complex128)
    if H.ndim == 2:
        H = H[None, :, :]
    if H.ndim != 3:
        raise ValueError(
            "expected channel_matrices of shape (n_steps, n_ue, n_bs) or (n_ue, n_bs); "
            f"got shape {H.shape}"
        )

    W = np.asarray(ue_weights, dtype=np.complex128)
    F = np.asarray(bs_weights, dtype=np.complex128)
    if W.ndim != 2 or F.ndim != 2:
        raise ValueError(
            f"expected 2-D weight matrices; got ue_weights {W.shape}, bs_weights {F.shape}"
        )
    if W.shape[1] != H.shape[1]:
        raise ValueError(
            f"ue_weights last axis ({W.shape[1]}) must match n_ue_elements ({H.shape[1]})"
        )
    if F.shape[1] != H.shape[2]:
        raise ValueError(
            f"bs_weights last axis ({F.shape[1]}) must match n_bs_elements ({H.shape[2]})"
        )

    # Y[t, k, l] = tx_amp * conj(W[k]) @ H[t] @ F[l]
    Y = tx_amp * np.einsum("ki,tij,lj->tkl", W.conj(), H, F)
    sigma_sq = noise_amplitude**2
    snr_lin = (np.abs(Y) ** 2) / sigma_sq
    # Reduce over (k, l), then convert to dB with the same floor as output_snr_db.
    best_per_step = snr_lin.reshape(snr_lin.shape[0], -1).max(axis=1)
    return 10.0 * np.log10(np.maximum(best_per_step, 1e-10))


def snr_regret_db(
    achieved_snr_db: NDArray[np.float64],
    oracle_snr_db: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-step gap between codebook oracle SNR and achieved SNR (dB).

    Sign convention::

        snr_regret_db = oracle_snr_db - achieved_snr_db

    so **lower is better and zero is optimal under the simulated codebook**.

    By construction this is non-negative when both inputs come from the
    same channel realisation and *achieved* uses the noiseless ideal
    measurement; tiny negative values can appear once *achieved* is the
    noisy SNR returned by :func:`output_snr_db`, because a favourable
    noise realisation at the measured ``(k, l)`` can momentarily exceed
    the noiseless oracle at the same step.  Treat negative values as
    floor noise rather than as a bug.

    Parameters
    ----------
    achieved_snr_db:
        SNR-in-dB trace produced by an algorithm under test, any shape.
    oracle_snr_db:
        Oracle SNR trace, broadcastable to *achieved_snr_db*.

    Returns
    -------
    Same shape as the broadcast of the two inputs.
    """
    a = np.asarray(achieved_snr_db, dtype=np.float64)
    o = np.asarray(oracle_snr_db, dtype=np.float64)
    return o - a


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
