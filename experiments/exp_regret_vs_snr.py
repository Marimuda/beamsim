"""Experiment: Regret to the codebook oracle vs. input SNR.

Reuses Case A geometry (UMi, 10 m/s, single BS at the IBS/2 reference
distance) but reports the per-step SNR gap to the codebook oracle
(``metrics.oracle_snr_db``) for each algorithm, rather than coverage
rate against a fixed threshold.

The codebook oracle is the strongest output SNR achievable on the
simulated finite UE×BS codebook for the same channel realisation. It
is *not* Shannon capacity and *not* a deployable policy. Per-step
regret is computed as

    Delta(m) = SNR_oracle_dB(m) - SNR_out_dB(m)

so lower is better and zero is optimal under the simulated codebook.
The cross-trial mean and 95 % bootstrap CI of this quantity are
plotted against input single-antenna SNR.

Defaults are deliberately lighter than ``exp_snr_sweep.py`` because the
purpose is to demonstrate a metric on a known scenario, not to publish
a new headline figure. Override with ``--n-trials`` and ``--n-steps``
for production runs.
"""

from __future__ import annotations

import argparse
import logging
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
from beamsim.runner import Experiment, run_experiment

logger = logging.getLogger("beamsim.experiments.regret_vs_snr")

# Case A geometry, mirroring ``exp_snr_sweep.py``.
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
    return tx_amp_for_snr_db(target_db, DISTANCE_M, 28e9, 10.0, 1.5, NOISE_AMP, n_ue, n_bs)


def run(n_trials: int, n_steps: int, output_dir: Path, snr_db_values: np.ndarray):
    output_dir.mkdir(parents=True, exist_ok=True)
    dt = 1e-3
    algorithms = ["exhaustive", "nns", "tabu", "angular_prediction", "ci", "mcmd"]

    # Per-trial mean regret (dB), shape (n_snr, n_trials).
    regret_per_snr: dict[str, np.ndarray] = {
        a: np.zeros((len(snr_db_values), n_trials)) for a in algorithms
    }

    for i, snr_db in enumerate(snr_db_values):
        tx_amp = _tx_amp_for(float(snr_db))
        exp = Experiment(
            name=f"regret_snr_{snr_db:+.1f}",
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
            seed=44444 + i,
        )
        result = run_experiment(exp, progress=False)

        snr_oracle_db = result["snr_oracle"]  # (n_trials, n_steps), v0.2.1 single-BS
        if snr_oracle_db is None:
            raise RuntimeError(
                "snr_oracle is None — this experiment requires beamsim >= 0.2.1 "
                "(single-BS oracle support). Run `pip install -e .` from a fresh "
                "checkout."
            )
        for a in algorithms:
            achieved_db = result["snr_db"][a]  # (n_trials, n_steps)
            regret_db = snr_oracle_db - achieved_db  # (n_trials, n_steps)
            regret_per_snr[a][i] = regret_db.mean(axis=1)  # per-trial mean
        logger.info("[regret_vs_snr] %d/%d : SNR=%+.1f dB done", i + 1, len(snr_db_values), snr_db)

    # ---- Plot ---------------------------------------------------------------
    set_publication_style()
    fig = fig_single_column()
    ax = fig.add_subplot(111)
    rng = np.random.default_rng(2)
    n_boot = 1000
    for a in algorithms:
        traces = regret_per_snr[a]
        mean_curve = traces.mean(axis=1)
        lo, hi = bootstrap_ci(traces.T, n_boot=n_boot, ci_alpha=0.05, rng=rng)
        color = ALGORITHM_PALETTE[a]
        ax.plot(snr_db_values, mean_curve, color=color, label=ALGORITHM_LABELS[a])
        ax.fill_between(snr_db_values, lo, hi, color=color, alpha=0.20, linewidth=0)

    ax.set_xlabel("Input single-antenna SNR (dB)")
    ax.set_ylabel(r"Mean regret to codebook oracle, $\overline{\Delta}$ (dB)")
    ax.set_title(f"Case A: UMi, 10 m/s, n_trials={n_trials} (lower is better)")
    ax.legend(fontsize=7, ncol=2)
    ax.invert_yaxis()  # smaller regret at the top of the plot
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)

    out_pdf = output_dir / "regret_vs_snr_10mps_umi_with_ci.pdf"
    save_figure(fig, out_pdf)
    plt.close(fig)
    np.savez_compressed(
        output_dir / "regret_vs_snr_aggregate.npz",
        snr_db=snr_db_values,
        **{f"regret_db_mean/{a}": regret_per_snr[a] for a in algorithms},
    )
    return out_pdf


def main():
    parser = argparse.ArgumentParser(
        description="Regret-to-codebook-oracle sweep on Case A UMi 10 m/s."
    )
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--n-snr-points", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/regret_vs_snr"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    snr_grid = np.linspace(-10.0, 20.0, args.n_snr_points)
    out = run(args.n_trials, args.n_steps, args.output, snr_grid)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
