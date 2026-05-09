"""MAB beam-selection comparison: UCB1 and ThompsonGaussian vs. existing algorithms.

Case A (UMi, 10 m/s) SNR sweep with UCB1 and Thompson added alongside the
six predecessor algorithms.  Produces a coverage-rate vs. input-SNR figure.

Usage:
    python experiments/exp_mab_comparison.py --n-trials 5 --n-steps 200

References:
    Auer, Cesa-Bianchi, Fischer (2002). UCB1.
    Hashemi et al. (2018). Contextual bandits for mmWave beam alignment.
"""

from __future__ import annotations

import argparse
import math
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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

GAMMA_TH_DB = -9.5335
NOISE_AMP = 1e-3
DISTANCE_M = 100.0
_BS_XY = np.array([0.0, 0.0])
_UE_PATH_Y = 150.0
_UE_PATH_HALF_LEN = 100.0

ALGORITHMS = [
    "exhaustive",
    "nns",
    "tabu",
    "angular_prediction",
    "ci",
    "mcmd",
    "ucb1",
    "thompson",
]


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


def _tx_amp_for(target_db: float) -> float:
    pl_db = umi_path_loss_db(DISTANCE_M, 28e9, 10.0, 1.5, los=True)
    pl_lin = 10 ** (-pl_db / 20.0)
    target_lin = 10 ** (target_db / 10.0)
    return float(NOISE_AMP * math.sqrt(4 * 16 * target_lin) / pl_lin)


def run(n_trials: int, n_steps: int, output_dir: Path, snr_db_values: np.ndarray) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dt = 1e-3

    cr_per_snr: dict[str, np.ndarray] = {
        a: np.zeros((len(snr_db_values), n_trials)) for a in ALGORITHMS
    }

    for i, snr_db in enumerate(snr_db_values):
        tx_amp = _tx_amp_for(float(snr_db))
        exp = Experiment(
            name=f"mab_{snr_db:+.1f}",
            n_steps=n_steps,
            dt=dt,
            n_trials=n_trials,
            algorithms=ALGORITHMS,
            bs_positions=[tuple(_BS_XY.tolist())],
            bs_yaws=[0.0],
            track_factory=partial(_track_factory, n_steps, dt),
            channel_factory=_channel_factory,
            noise_amplitude=NOISE_AMP,
            tx_amp=tx_amp,
            seed=77777 + i,
        )
        result = run_experiment(exp, progress=False)
        for a in ALGORITHMS:
            cr_per_snr[a][i] = coverage_rate(result["snr_db"][a], GAMMA_TH_DB)
        print(f"[mab_comparison] {i + 1}/{len(snr_db_values)}: SNR={snr_db:+.1f} dB done")

    # Save aggregate data
    np.savez_compressed(
        output_dir / "mab_comparison_aggregate.npz",
        snr_db=snr_db_values,
        gamma_th_db=GAMMA_TH_DB,
        **{f"coverage_rate_{a}": cr_per_snr[a] for a in ALGORITHMS},
    )

    # Report +20 dB mean output SNR for UCB1 and Thompson
    idx_20 = int(np.argmin(np.abs(snr_db_values - 20.0)))
    for a in ["ucb1", "thompson", "exhaustive"]:
        cr = float(cr_per_snr[a][idx_20].mean())
        print(f"  Coverage rate @ +20 dB input: {a} = {cr:.4f}")

    # Plot
    set_publication_style()
    fig = fig_single_column()
    ax = fig.add_subplot(111)
    rng = np.random.default_rng(3)

    for a in ALGORITHMS:
        traces = cr_per_snr[a]  # (n_snr, n_trials)
        mean_curve = traces.mean(axis=1)
        color = ALGORITHM_PALETTE.get(a)
        label = ALGORITHM_LABELS.get(a, a)
        ax.plot(snr_db_values, mean_curve, color=color, label=label)
        if n_trials > 1:
            lo, hi = bootstrap_ci(traces.T, n_boot=500, ci_alpha=0.05, rng=rng)
            ax.fill_between(snr_db_values, lo, hi, color=color, alpha=0.20, linewidth=0)

    ax.axhline(
        GAMMA_TH_DB,
        color="k",
        linewidth=0.5,
        linestyle="--",
        label=rf"$\Gamma_{{\mathrm{{th}}}}={GAMMA_TH_DB:.2f}$ dB",
    )
    ax.set_xlabel("Input single-antenna SNR (dB)")
    ax.set_ylabel(r"Coverage rate")
    ax.set_title(f"MAB comparison: Case A UMi 10 m/s (n_trials={n_trials})")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.0)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)

    out_pdf = output_dir / "mab_comparison_with_ci.pdf"
    save_figure(fig, out_pdf)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    return out_pdf


def main():
    parser = argparse.ArgumentParser(description="MAB beam-selection comparison (Case A).")
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--n-snr-points", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    snr_grid = np.linspace(-15.0, 30.0, args.n_snr_points)
    run(args.n_trials, args.n_steps, args.output, snr_grid)


if __name__ == "__main__":
    main()
