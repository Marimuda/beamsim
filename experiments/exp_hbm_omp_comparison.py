"""Case A UMi 10 m/s SNR sweep: Exhaustive vs HBM vs OMP compressive.

Adds HBM (Alkhateeb et al. 2014) and OMP compressive (Marzi et al. 2016) to
the standard Case A SNR sweep for comparison with Exhaustive search.

Usage:
    python experiments/exp_hbm_omp_comparison.py [--n-trials 5] [--n-snr-points 11]

Output:
    results/hbm_omp_comparison_with_ci.pdf
"""

from __future__ import annotations

import argparse
import math
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beamsim.channel import ChannelParams, ChannelRealisation, umi_path_loss_db
from beamsim.geometry import straight_line_track
from beamsim.metrics import coverage_rate
from beamsim.plotting import (
    ALGORITHM_LABELS,
    ALGORITHM_PALETTE,
    bootstrap_ci,
    fig_single_column,
    save_figure,
    set_publication_style,
)
from beamsim.runner import Experiment, run_experiment

# Case A constants (matching exp_snr_sweep.py)
GAMMA_TH_DB = -9.5335
DISTANCE_M = 100.0
NOISE_AMP = 1e-3
_BS_XY = np.array([0.0, 0.0])
_UE_PATH_Y = 150.0
_UE_PATH_HALF_LEN = 100.0


def _track_factory(n_steps: int, dt: float, rng: np.random.Generator):
    start_x = float(rng.uniform(-_UE_PATH_HALF_LEN, _UE_PATH_HALF_LEN))
    return straight_line_track(
        start_xy=(start_x, _UE_PATH_Y),
        heading=0.0,
        speed_mps=10.0,
        n_steps=n_steps,
        dt=dt,
    )


def _channel_factory(rng: np.random.Generator, bs_index: int):
    params = ChannelParams(ue_speed_mps=10.0)
    return ChannelRealisation(
        params=params,
        bs_xy=_BS_XY,
        bs_yaw=0.0,
        n_bs_elements=16,
        n_ue_elements=4,
        rng=rng,
    )


def _tx_amp_for(target_db: float, n_ue: int = 4, n_bs: int = 16) -> float:
    pl_db = umi_path_loss_db(DISTANCE_M, 28e9, 10.0, 1.5, los=True)
    pl_lin = 10 ** (-pl_db / 20.0)
    target_lin = 10 ** (target_db / 10.0)
    return float(NOISE_AMP * math.sqrt(n_ue * n_bs * target_lin) / pl_lin)


def run(
    n_trials: int,
    n_steps: int,
    output_dir: Path,
    snr_db_values: np.ndarray,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dt = 1e-3
    algorithms = ["exhaustive", "hbm", "omp_compressive"]

    cr_per_snr: dict[str, np.ndarray] = {
        a: np.zeros((len(snr_db_values), n_trials)) for a in algorithms
    }

    for i, snr_db in enumerate(snr_db_values):
        tx_amp = _tx_amp_for(float(snr_db))
        exp = Experiment(
            name=f"hbm_omp_{snr_db:+.1f}",
            n_steps=n_steps,
            dt=dt,
            n_trials=n_trials,
            algorithms=algorithms,
            bs_positions=[tuple(_BS_XY.tolist())],
            bs_yaws=[0.0],
            track_factory=partial(_track_factory, n_steps, dt),
            channel_factory=_channel_factory,
            noise_amplitude=NOISE_AMP,
            tx_amp=tx_amp,
            seed=77777 + i,
        )
        result = run_experiment(exp, progress=False)
        for a in algorithms:
            cr_per_snr[a][i] = coverage_rate(result["snr_db"][a], GAMMA_TH_DB)
        print(f"[hbm_omp] {i + 1}/{len(snr_db_values)} : SNR={snr_db:+.1f} dB done")

    # --- Report final-step mean output SNR at +20 dB ---
    target_snr_db = 20.0
    # Find closest SNR point to +20 dB
    idx_20 = int(np.argmin(np.abs(snr_db_values - target_snr_db)))
    print(f"\nCoverage rate at SNR ~{snr_db_values[idx_20]:+.1f} dB (index {idx_20}):")
    for a in algorithms:
        mean_cr = float(cr_per_snr[a][idx_20].mean())
        print(f"  {a:20s}: {mean_cr:.3f}")

    # --- Plot ---
    set_publication_style()
    fig = fig_single_column()
    ax = fig.add_subplot(111)
    rng = np.random.default_rng(5)
    n_boot = 500

    for a in algorithms:
        traces = cr_per_snr[a]  # (n_snr, n_trials)
        mean_curve = traces.mean(axis=1)
        lo, hi = bootstrap_ci(traces.T, n_boot=n_boot, ci_alpha=0.05, rng=rng)
        color = ALGORITHM_PALETTE.get(a)
        label = ALGORITHM_LABELS.get(a, a)
        ax.plot(snr_db_values, mean_curve, color=color, label=label)
        if n_trials > 1:
            ax.fill_between(snr_db_values, lo, hi, color=color, alpha=0.20, linewidth=0)

    ax.axhline(
        GAMMA_TH_DB,
        color="k",
        linewidth=0.6,
        linestyle="--",
        label=rf"$\Gamma_{{th}}={GAMMA_TH_DB:.2f}$ dB",
    )
    ax.set_xlabel("Input single-antenna SNR (dB)")
    ax.set_ylabel(rf"Coverage rate ($\gamma_{{\mathrm{{th}}}}={GAMMA_TH_DB:.2f}$ dB)")
    ax.set_title(f"Case A: Exhaustive vs HBM vs OMP, n_trials={n_trials}")
    ax.legend(fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.0)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)

    out_pdf = output_dir / "hbm_omp_comparison_with_ci.pdf"
    save_figure(fig, out_pdf)
    plt.close(fig)
    print(f"\nSaved {out_pdf}")
    return out_pdf


def main():
    parser = argparse.ArgumentParser(description="HBM vs OMP vs Exhaustive — Case A SNR sweep.")
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--n-steps", type=int, default=1_000)
    parser.add_argument("--n-snr-points", type=int, default=11)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    snr_grid = np.linspace(-15.0, 30.0, args.n_snr_points)
    run(args.n_trials, args.n_steps, args.output, snr_grid)


if __name__ == "__main__":
    main()
