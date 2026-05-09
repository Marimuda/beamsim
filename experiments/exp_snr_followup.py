"""Follow-up figures from existing SNR-sweep data (no new simulation runs).

Generates:
  - Fig 6.5: Mean output SNR vs input SNR, 95% BCa CI ribbon
  - Fig 6.6: Empirical CDF of output SNR at input SNR closest to 0 dB
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bootstrap

from beamsim.plotting import (
    ALGORITHM_LABELS,
    ALGORITHM_PALETTE,
    fig_single_column,
    save_figure,
    set_publication_style,
)

ALGORITHMS = ["exhaustive", "nns", "tabu", "angular_prediction", "ci", "mcmd"]
RESULTS_DIR = Path(__file__).parent.parent / "results"


def _load_per_snr_files() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (snr_input_db sorted array, algo -> (n_snr, n_trials) mean-per-trial array)."""

    def _snr_val(f: str) -> float:
        stem = os.path.basename(f).replace("snr_", "").replace(".npz", "")
        return float(stem)

    raw = [
        f
        for f in glob.glob(str(RESULTS_DIR / "snr_*.npz"))
        if os.path.basename(f) != "snr_aggregate.npz"
    ]
    files = sorted(raw, key=_snr_val)

    # Belt-and-suspenders: catch duplicate grid points (e.g. stale pre-calibration
    # files interleaved with post-calibration files at the same input-SNR value).
    all_vals = [_snr_val(f) for f in files]
    seen: set[float] = set()
    dupes = [v for v in all_vals if v in seen or seen.add(v)]  # type: ignore[func-returns-value]
    if dupes:
        raise ValueError(
            f"Duplicate input-SNR grid points detected in {RESULTS_DIR}: {sorted(set(dupes))}. "
            "Delete stale pre-calibration snr_*.npz files before re-running."
        )

    snr_values: list[float] = []
    trial_means: dict[str, list[np.ndarray]] = {a: [] for a in ALGORITHMS}

    for fpath in files:
        snr_val = float(os.path.basename(fpath).replace("snr_", "").replace(".npz", ""))
        d = np.load(fpath)
        snr_values.append(snr_val)
        for a in ALGORITHMS:
            arr = d[f"snr_db/{a}"]  # (n_trials, n_steps)
            # Average in linear domain, then convert to dB (convention fix).
            lin = np.power(10.0, arr / 10.0)
            trial_means[a].append(10.0 * np.log10(lin.mean(axis=1)))  # (n_trials,)

    x = np.array(snr_values)
    # shape: (n_snr, n_trials)
    traces = {a: np.stack(trial_means[a], axis=0) for a in ALGORITHMS}
    return x, traces


def _bca_ci(samples_1d: np.ndarray) -> tuple[float, float]:
    """95% BCa CI for the mean of samples_1d."""
    result = bootstrap(
        (samples_1d,),
        statistic=np.mean,
        n_resamples=1000,
        confidence_level=0.95,
        method="BCa",
        random_state=42,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def plot_mean_snr(
    x: np.ndarray,
    traces: dict[str, np.ndarray],
    n_trials: int,
    output_path: Path,
) -> None:
    """Fig 6.5 — Mean output SNR vs input SNR with 95% BCa CI ribbon."""
    set_publication_style()
    fig = fig_single_column()
    ax = fig.add_subplot(111)

    for a in ALGORITHMS:
        data = traces[a]  # (n_snr, n_trials)
        mean_curve = data.mean(axis=1)  # (n_snr,)
        lo = np.empty_like(mean_curve)
        hi = np.empty_like(mean_curve)
        for i, samples in enumerate(data):
            lo[i], hi[i] = _bca_ci(samples)

        color = ALGORITHM_PALETTE[a]
        ax.plot(x, mean_curve, color=color, label=ALGORITHM_LABELS[a])
        ax.fill_between(x, lo, hi, color=color, alpha=0.20, linewidth=0)

    ax.set_xlabel("Input single-antenna SNR (dB)")
    ax.set_ylabel("Mean output SNR (dB)")
    ax.set_title(f"Case A: UMi, 10 m/s, n_trials={n_trials}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)

    save_figure(fig, output_path)
    plt.close(fig)


def plot_cdf(output_path: Path) -> None:
    """Fig 6.6 — Empirical CDF of output SNR at input SNR closest to 0 dB."""
    files = [
        f
        for f in glob.glob(str(RESULTS_DIR / "snr_*.npz"))
        if os.path.basename(f) != "snr_aggregate.npz"
    ]
    closest = min(
        files,
        key=lambda f: abs(float(os.path.basename(f).replace("snr_", "").replace(".npz", ""))),
    )
    snr_label = float(os.path.basename(closest).replace("snr_", "").replace(".npz", ""))
    d = np.load(closest)

    set_publication_style()
    fig = fig_single_column()
    ax = fig.add_subplot(111)

    for a in ALGORITHMS:
        arr = d[f"snr_db/{a}"]  # (n_trials, n_steps)
        n_trials = arr.shape[0]
        # Plot faint per-trial CDFs
        for t in range(n_trials):
            trial_sorted = np.sort(arr[t])
            cdf = np.arange(1, len(trial_sorted) + 1) / len(trial_sorted)
            ax.plot(trial_sorted, cdf, color=ALGORITHM_PALETTE[a], alpha=0.06, linewidth=0.4)
        # Pooled CDF
        flat = arr.ravel()
        flat_sorted = np.sort(flat)
        cdf_pool = np.arange(1, len(flat_sorted) + 1) / len(flat_sorted)
        ax.plot(
            flat_sorted,
            cdf_pool,
            color=ALGORITHM_PALETTE[a],
            label=ALGORITHM_LABELS[a],
            linewidth=1.0,
        )

    n_trials_total = np.load(closest)["n_trials"].item()
    ax.set_xlabel("Output SNR (dB)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(
        f"Case A: UMi, 10 m/s, output SNR CDF at input SNR = {snr_label:+.1f} dB,"
        f" n_trials={n_trials_total}"
    )
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    ax.set_ylim(0, 1)

    save_figure(fig, output_path)
    plt.close(fig)


def main() -> None:
    x, traces = _load_per_snr_files()
    n_trials = traces["exhaustive"].shape[1]

    out_mean = RESULTS_DIR / "SNR_sweep_MeanSNR_10_mps_umi_with_ci.pdf"
    out_cdf = RESULTS_DIR / "SNR_sweep_CDF_at0dB_10_mps_umi_with_ci.pdf"

    print("Generating Fig 6.5 (mean SNR)...")
    plot_mean_snr(x, traces, n_trials, out_mean)
    print(f"  -> {out_mean}  ({os.path.getsize(out_mean) / 1024:.1f} KB)")

    print("Generating Fig 6.6 (CDF at ~0 dB)...")
    plot_cdf(out_cdf)
    print(f"  -> {out_cdf}  ({os.path.getsize(out_cdf) / 1024:.1f} KB)")

    # Sanity-check numbers for Exhaustive
    # traces values are already per-trial linear-averaged then dB-converted;
    # average across trials in linear domain for the grand mean.
    ex = traces["exhaustive"]
    idx_0 = int(np.argmin(np.abs(x)))
    idx_30 = int(np.argmin(np.abs(x - 30.0)))

    def _lin_mean_db(samples: np.ndarray) -> float:
        return float(10.0 * np.log10(np.power(10.0, samples / 10.0).mean()))

    print("\nSanity check — Exhaustive:")
    print(f"  Mean output SNR at input ~{x[idx_0]:+.1f} dB : {_lin_mean_db(ex[idx_0]):.2f} dB")
    print(f"  Mean output SNR at input ~{x[idx_30]:+.1f} dB : {_lin_mean_db(ex[idx_30]):.2f} dB")


if __name__ == "__main__":
    main()
