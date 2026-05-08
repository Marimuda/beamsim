"""Experiment 3: Coverage rate vs. input SNR at 10 m/s UMi.

Sweep single-antenna SNR over a wide range; coverage-rate threshold fixed
at the predecessor's gamma_th = -9.5335 dB. 31 SNR points (matching the
predecessor's "61 steps" with reduced density to keep wall time manageable;
at 30 Monte Carlo trials and ~500 occasions per trial this completes in
under 10 minutes on 7 cores).
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from beamsim.channel import ChannelParams, ChannelRealisation, umi_path_loss_db
from beamsim.geometry import straight_line_track
from beamsim.metrics import coverage_rate
from beamsim.plotting import (
    ALGORITHM_LABELS,
    ALGORITHM_PALETTE,
    fig_single_column,
    save_figure,
    set_publication_style,
)
from beamsim.runner import Experiment, run_experiment, save_experiment


GAMMA_TH_DB = -9.5335
DISTANCE_M = 50.0
NOISE_AMP = 1e-3


def _track_factory(n_steps: int, dt: float, rng: np.random.Generator):
    start_x = float(rng.uniform(-5.0, 5.0))
    start_y = float(rng.uniform(-5.0, 5.0))
    heading = float(rng.uniform(-np.pi, np.pi))
    return straight_line_track(start_xy=(start_x, start_y),
                                heading=heading,
                                speed_mps=10.0,
                                n_steps=n_steps,
                                dt=dt)


def _channel_factory(rng: np.random.Generator, bs_index: int):
    params = ChannelParams(ue_speed_mps=10.0)
    return ChannelRealisation(params=params,
                               bs_xy=np.array([DISTANCE_M, 0.0]),
                               bs_yaw=0.0,
                               n_bs_elements=16,
                               n_ue_elements=4,
                               rng=rng)


def _tx_amp_for(target_db: float) -> float:
    pl_db = umi_path_loss_db(DISTANCE_M, 28e9, 10.0, 1.5, los=True)
    pl_lin = 10 ** (-pl_db / 20.0)
    target_lin = 10 ** (target_db / 10.0)
    return float(NOISE_AMP * np.sqrt(target_lin) / pl_lin)


def run(n_trials: int, n_steps: int, output_dir: Path, snr_db_values: np.ndarray):
    output_dir.mkdir(parents=True, exist_ok=True)
    dt = 1e-3
    algorithms = ["exhaustive", "nns", "tabu", "angular_prediction", "ci", "mcmd"]

    cr_per_snr: dict[str, np.ndarray] = {a: np.zeros((len(snr_db_values), n_trials))
                                           for a in algorithms}

    for i, snr_db in enumerate(snr_db_values):
        tx_amp = _tx_amp_for(float(snr_db))
        exp = Experiment(
            name=f"snr_{snr_db:+.1f}",
            n_steps=n_steps,
            dt=dt,
            n_trials=n_trials,
            algorithms=algorithms,
            bs_positions=[(DISTANCE_M, 0.0)],
            bs_yaws=[0.0],
            track_factory=partial(_track_factory, n_steps, dt),
            channel_factory=_channel_factory,
            noise_amplitude=NOISE_AMP,
            tx_amp=tx_amp,
            seed=33333 + i,
        )
        result = run_experiment(exp, progress=False)
        save_experiment(result, output_dir / f"snr_{snr_db:+.1f}.npz")
        for a in algorithms:
            cr_per_snr[a][i] = coverage_rate(result["snr_db"][a], GAMMA_TH_DB)
        print(f"[snr_sweep] {i+1}/{len(snr_db_values)} : SNR={snr_db:+.1f} dB done")

    # Plot
    set_publication_style()
    fig = fig_single_column()
    ax = fig.add_subplot(111)
    rng = np.random.default_rng(2)
    n_boot = 1000
    for a in algorithms:
        traces = cr_per_snr[a]
        mean_curve = traces.mean(axis=1)
        boot_means = np.empty((n_boot, traces.shape[0]))
        for b in range(n_boot):
            idx = rng.integers(0, traces.shape[1], size=traces.shape[1])
            boot_means[b] = traces[:, idx].mean(axis=1)
        lo = np.percentile(boot_means, 2.5, axis=0)
        hi = np.percentile(boot_means, 97.5, axis=0)
        color = ALGORITHM_PALETTE[a]
        ax.plot(snr_db_values, mean_curve, color=color, label=ALGORITHM_LABELS[a])
        ax.fill_between(snr_db_values, lo, hi, color=color, alpha=0.20, linewidth=0)

    ax.set_xlabel("Input single-antenna SNR (dB)")
    ax.set_ylabel(rf"Coverage rate ($\gamma_{{\mathrm{{th}}}}={GAMMA_TH_DB}$ dB)")
    ax.set_title(f"UMi LOS, 10 m/s, n_trials={n_trials}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    ax.set_ylim(0, 1)

    out_pdf = output_dir / "SNR_sweep_Coverage_Rate_10_mps_umi_with_ci.pdf"
    save_figure(fig, out_pdf)
    plt.close(fig)
    np.savez_compressed(output_dir / "snr_aggregate.npz",
                        snr_db=snr_db_values, gamma_th_db=GAMMA_TH_DB,
                        **{f"coverage_rate/{a}": cr_per_snr[a] for a in algorithms})
    return out_pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--n-snr-points", type=int, default=31)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    snr_grid = np.linspace(-10.0, 30.0, args.n_snr_points)
    out = run(args.n_trials, args.n_steps, args.output, snr_grid)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
