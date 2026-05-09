"""Experiment 4 -- Case D: Multi-BS handover -- wrong-BS-selection loss L_BS.

Faithfully reproduces predecessor MSc thesis Case D (Section 5.2.2 and
Section 6.6 / Fig 6.8 / Fig 6.9).

Case D definition (predecessor Section 5.2.2):
  - Single UE, 4 BSs in a center-excited hexagonal tessellation (UMi, IBS=200 m).
  - UE path is a straight line from x=0 to x=2*IBS+IBS/2 = 500 m at y=0,
    passing through all four BS cell areas (predecessor: "goes from zero to
    2IBS + IBS/2 in order to go through all the cell areas").
  - UE speed: 10 m/s (predecessor Fig 6.8/6.9 caption: "10 m/s Case D scenario").
  - Trial duration: 50 seconds = 50 000 occasions at dt = 1 ms.
  - Metric: L_BS (dB) = mean power loss from wrong-BS selection (Eq. 6.2).

BS layout (center-excited hex, IBS = 200 m):
  - BS_0 at (  0,    0)
  - BS_1 at (200,    0)
  - BS_2 at (100,  +100*sqrt(3))  ~= (100, +173.2)
  - BS_3 at (100,  -100*sqrt(3))  ~= (100, -173.2)
  All 4 BS-to-BS distances are exactly IBS = 200 m.

UE path: straight line from x=0 to x=500 m at y=0 (heading=+x).
  Start is perturbed by small Gaussian noise per trial for Monte Carlo diversity.

Reflectors (specular):
  TODO: Implement deterministic per-BS specular reflectors as described in the
  predecessor (Section 3.2; reflectors placed in a box of IBS/2 from BS, below
  BS height). The current implementation uses random NLOS clusters via
  ChannelRealisation as a proxy. When the channel model exposes a deterministic
  reflector placement API, replace _channel_factory with a version that places
  4 reflectors per BS at fixed angles (0, 90, 180, 270 deg) at distance IBS/4.

Number of trials: 30.
"""

from __future__ import annotations

import argparse
import math
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beamsim.channel import ChannelParams, ChannelRealisation
from beamsim.geometry import straight_line_track
from beamsim.link_budget import tx_amp_for_snr_db
from beamsim.plotting import (
    ALGORITHM_LABELS,
    ALGORITHM_PALETTE,
    bootstrap_ci,
    fig_single_column,
    save_figure,
    set_publication_style,
)
from beamsim.runner import Experiment, run_experiment, save_experiment

# Case D: 4 BSs in center-excited hexagonal tessellation, IBS = 200 m.
# (predecessor Section 5.2.2 and 3GPP UMi Table in Section 3.2)
IBS = 200.0
BS_POSITIONS: list[tuple[float, float]] = [
    (0.0, 0.0),
    (IBS, 0.0),
    (IBS / 2.0, IBS / 2.0 * math.sqrt(3)),  # (100, ~173.2)
    (IBS / 2.0, -IBS / 2.0 * math.sqrt(3)),  # (100, ~-173.2)
]

# UE path: from x=0 to x=2*IBS+IBS/2 = 500 m at y=0.
# At 10 m/s that is 50 seconds = 50 000 steps at dt=1 ms.
_UE_SPEED_MPS = 10.0
_PATH_LENGTH_M = 2.0 * IBS + IBS / 2.0  # = 500 m
_TRIAL_DURATION_S = _PATH_LENGTH_M / _UE_SPEED_MPS  # = 50 s
_N_STEPS_DEFAULT = int(_TRIAL_DURATION_S / 1e-3)  # = 50 000

# tx_amp calibrated to per-element input SNR = 10 dB at IBS/2 = 100 m (Case A reference).
# See Sec 5.3.3 Eq 5.9 of the report.
_NOISE_AMP = 1e-3
_N_UE = 4
_N_BS = 16
_TX_AMP = tx_amp_for_snr_db(10.0, 100.0, 28e9, 10.0, 1.5, _NOISE_AMP, _N_UE, _N_BS)


def _track_factory(n_steps: int, dt: float, rng: np.random.Generator):
    # UE starts at x=0 with small Gaussian perturbation for Monte Carlo diversity.
    start_x = float(rng.normal(0.0, 2.0))
    return straight_line_track(
        start_xy=(start_x, 0.0),
        heading=0.0,  # +x direction
        speed_mps=_UE_SPEED_MPS,
        n_steps=n_steps,
        dt=dt,
    )


def _channel_factory(rng: np.random.Generator, bs_index: int, disable_clusters: bool = False):
    # TODO: Replace with deterministic specular reflectors per BS when the
    # channel model exposes a placement API.  Currently uses random NLOS
    # clusters as a proxy (predecessor "with reflectors" approximation).
    params = ChannelParams(ue_speed_mps=_UE_SPEED_MPS, disable_clusters=disable_clusters)
    return ChannelRealisation(
        params=params,
        bs_xy=np.array(BS_POSITIONS[bs_index]),
        bs_yaw=0.0,
        n_bs_elements=16,
        n_ue_elements=4,
        rng=rng,
    )


def _lbs_per_trial(
    snr_db_best: np.ndarray,  # (n_trials, n_steps)
    snr_db_obtained: np.ndarray,  # (n_trials, n_steps)
) -> np.ndarray:
    """Compute scalar L_BS per trial using linear-domain averaging (Eq 6.2).

    L_BS = 10*log10(mean_lin(gamma_best) / mean_lin(gamma_obtained))
    Returns shape (n_trials,).
    """
    gamma_best = 10.0 ** (snr_db_best / 10.0)  # (n_trials, n_steps)
    gamma_obtained = 10.0 ** (snr_db_obtained / 10.0)
    mean_best = gamma_best.mean(axis=1)  # (n_trials,)
    mean_obtained = gamma_obtained.mean(axis=1)
    return 10.0 * np.log10(mean_best / np.maximum(mean_obtained, 1e-30))


def _bar_chart(
    algorithms: list[str],
    lbs_mean: dict[str, float],
    lbs_ci: dict[str, tuple[float, float]],
    title: str,
    output_path: Path,
) -> None:
    """Plot a bar chart of mean L_BS per algorithm with 95% BCa CI error bars."""
    set_publication_style()
    fig = fig_single_column()
    ax = fig.add_subplot(111)

    x = np.arange(len(algorithms))
    means = np.array([lbs_mean[a] for a in algorithms])
    lo = np.maximum(np.array([lbs_mean[a] - lbs_ci[a][0] for a in algorithms]), 0.0)
    hi = np.maximum(np.array([lbs_ci[a][1] - lbs_mean[a] for a in algorithms]), 0.0)
    colors = [ALGORITHM_PALETTE[a] for a in algorithms]
    labels = [ALGORITHM_LABELS[a] for a in algorithms]

    ax.bar(x, means, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.errorbar(x, means, yerr=[lo, hi], fmt="none", color="black", capsize=3, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(r"$L_{BS}$ (dB)")
    ax.set_title(title, fontsize=8)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", which="both", linewidth=0.3, alpha=0.4)

    save_figure(fig, output_path)
    plt.close(fig)


def run(n_trials: int, n_steps: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    dt = 1e-3
    algorithms = ["exhaustive", "nns", "tabu", "angular_prediction", "ci", "mcmd"]

    # -----------------------------------------------------------------------
    # Run both variants
    # -----------------------------------------------------------------------
    results = {}
    for variant, disable_clusters in [("with_reflectors", False), ("no_reflectors", True)]:
        ch_factory = partial(_channel_factory, disable_clusters=disable_clusters)
        exp = Experiment(
            name=f"handover_4bs_10mps_{variant}",
            n_steps=n_steps,
            dt=dt,
            n_trials=n_trials,
            algorithms=algorithms,
            bs_positions=BS_POSITIONS,
            bs_yaws=[0.0, 0.0, 0.0, 0.0],
            track_factory=partial(_track_factory, n_steps, dt),
            channel_factory=ch_factory,
            noise_amplitude=_NOISE_AMP,
            tx_amp=_TX_AMP,
            seed=44444,
        )
        result = run_experiment(exp, progress=True)
        save_experiment(result, output_dir / f"handover_{variant}.npz")
        results[variant] = result

    # -----------------------------------------------------------------------
    # Time-series plot (with reflectors only, for debugging)
    # -----------------------------------------------------------------------
    result_wr = results["with_reflectors"]
    snr_db = result_wr["snr_db"]
    # gamma_best = true oracle: max over all (BS, k, l) noiseless SNR.
    # Consistent with bar-chart definition (Eq 6.2) so Exhaustive is no longer
    # flat at L_BS=0 by construction.
    snr_best_per_occasion = result_wr["snr_oracle"]  # (n_trials, n_steps)

    l_bs_timeseries: dict[str, np.ndarray] = {}
    for a in algorithms:
        gap = snr_best_per_occasion - snr_db[a]
        l_bs_timeseries[a] = np.maximum(gap, 0.0)

    set_publication_style()
    fig = fig_single_column()
    ax = fig.add_subplot(111)
    times = np.arange(n_steps) * dt
    rng = np.random.default_rng(3)
    n_boot = 200
    win = max(1, n_steps // 200)
    for a in algorithms:
        traces = l_bs_timeseries[a]
        mean_curve = traces.mean(axis=0)
        kernel = np.ones(win) / win
        mean_smooth = np.convolve(mean_curve, kernel, mode="same")
        lo, hi = bootstrap_ci(traces, n_boot=n_boot, ci_alpha=0.05, rng=rng)
        lo_smooth = np.convolve(lo, kernel, mode="same")
        hi_smooth = np.convolve(hi, kernel, mode="same")
        color = ALGORITHM_PALETTE[a]
        ax.plot(times, mean_smooth, color=color, label=ALGORITHM_LABELS[a])
        ax.fill_between(times, lo_smooth, hi_smooth, color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Wrong-BS selection gap $L_{BS}$ (dB)")
    ax.set_title(f"Case D: 4-BS UMi handover, 10 m/s, n_trials={n_trials}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)

    timeseries_pdf = output_dir / "handover_lbs_timeseries.pdf"
    save_figure(fig, timeseries_pdf)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Bar charts: Fig 6.8 (with reflectors) and Fig 6.9 (without reflectors)
    # -----------------------------------------------------------------------
    output_pdfs = {}
    for variant, label in [
        ("with_reflectors", "With reflectors"),
        ("no_reflectors", "Without reflectors"),
    ]:
        res = results[variant]
        snr_obt = res["snr_db"]

        # gamma_best = true oracle: max over all (BS, k, l) noiseless SNR.
        # Shape (n_trials, n_steps); algorithm-independent upper bound (Eq 6.2).
        gamma_best_arr = res["snr_oracle"]  # (n_trials, n_steps)

        lbs_mean: dict[str, float] = {}
        lbs_ci: dict[str, tuple[float, float]] = {}

        for a in algorithms:
            best_arr = gamma_best_arr

            lbs_trials = _lbs_per_trial(best_arr, snr_obt[a])  # (n_trials,)
            lbs_trials = np.maximum(lbs_trials, 0.0)
            lbs_mean[a] = float(lbs_trials.mean())

            # 95% BCa bootstrap CI via shared bootstrap_ci helper
            lo_arr, hi_arr = bootstrap_ci(
                lbs_trials.reshape(1, -1).T,
                n_boot=2000,
                ci_alpha=0.05,
                rng=np.random.default_rng(7),
            )
            lbs_ci[a] = (float(lo_arr[0]), float(hi_arr[0]))

        # Sanity check: warn if values are out of report's expected range
        max_lbs = max(lbs_mean.values())
        if max_lbs > 20.0 or max_lbs < 0.01:
            print(
                f"WARNING [{variant}]: max L_BS={max_lbs:.2f} dB is outside "
                f"expected report range (0-7 dB). Continuing."
            )

        pdf_name = (
            "handover_lbs_with_reflectors.pdf"
            if variant == "with_reflectors"
            else "handover_lbs_no_reflectors.pdf"
        )
        pdf_path = output_dir / pdf_name
        _bar_chart(
            algorithms,
            lbs_mean,
            lbs_ci,
            title=(
                f"Case D: 4-BS handover, 10 m/s — {label}\n(n_trials={n_trials}, n_steps={n_steps})"
            ),
            output_path=pdf_path,
        )
        output_pdfs[variant] = pdf_path

        print(f"\n--- L_BS summary [{variant}] ---")
        for a in algorithms:
            lo_ci, hi_ci = lbs_ci[a]
            print(
                f"  {ALGORITHM_LABELS[a]:25s}: {lbs_mean[a]:.3f} dB  "
                f"[{lo_ci:.3f}, {hi_ci:.3f}] 95% BCa CI"
            )

    # Legacy aggregate (for backward compat)
    np.savez_compressed(
        output_dir / "handover_aggregate.npz",
        **{f"l_bs_db/{a}": l_bs_timeseries[a] for a in algorithms},
    )

    return output_pdfs


def main():
    parser = argparse.ArgumentParser(
        description="Case D: 4-BS hexagonal handover scenario (predecessor Fig 6.8)."
    )
    parser.add_argument("--n-trials", type=int, default=30)
    # 50 s at 1 ms = 50 000 steps (10 m/s over 500 m path).
    # Use a smaller value (e.g. 5000) for quick smoke-tests.
    parser.add_argument("--n-steps", type=int, default=_N_STEPS_DEFAULT)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    pdfs = run(args.n_trials, args.n_steps, args.output)
    for variant, path in pdfs.items():
        print(f"Wrote [{variant}]: {path}")


if __name__ == "__main__":
    main()
