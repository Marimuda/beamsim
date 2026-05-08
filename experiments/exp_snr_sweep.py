"""Experiment 3 — Case A (SNR sweep): Coverage rate vs. input single-antenna SNR.

Faithfully reproduces predecessor MSc thesis Case A with SNR sweep (Section 6.4
/ Fig 6.4 and Fig 6.5).

Case A definition (predecessor Section 5.2.2) — same geometry as exp_alpha_sweep:
  - Single UE, single BS, straight-line path — 3GPP UMi channel model.
  - IBS = 200 m; BS at origin, UE path at y = 150 m (3/4 IBS lateral offset).
  - Reference distance for tx_amp calibration: IBS/2 = 100 m.
  - UE speed: 10 m/s (Fig 6.4 caption: "UE speed = 10 m/s").

SNR sweep:
  - x-axis range: -15 to +30 dB (matching predecessor Fig 6.4 x-axis extent).
  - 46 points (1-dB steps across -15 to +30 dB); may be reduced to 31 points
    (1.5-dB steps) via --n-snr-points for faster wall time with explicit
    documentation trade-off.

Coverage threshold: -9.53 dB (predecessor Fig 6.4 caption: "SNR threshold
-9.53 dB", corresponding to the minimum SINR at MCS-1, Section 5.3.4
Equation 5.21).

Trial duration: 1 second = 1 000 steps at dt = 1 ms (predecessor default for
Case A, consistent with Section 6 simulation parameters).

Number of trials: 30 per SNR point.
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


# Predecessor Fig 6.4 caption: "coverage rate as defined in subsection 5.3.4,
# with SNR threshold -9.53 dB" (minimum SINR at MCS-1, Equation 5.21).
GAMMA_TH_DB = -9.5335

# Case A reference distance: IBS/2 = 100 m (predecessor link budget Section 3.2.6).
DISTANCE_M = 100.0
NOISE_AMP = 1e-3

# Case A geometry: BS at origin, UE path at y = 3/4 * IBS = 150 m.
_BS_XY = np.array([0.0, 0.0])
_UE_PATH_Y = 150.0
_UE_PATH_HALF_LEN = 100.0   # half of IBS = 100 m; UE traverses ±100 m along x


def _track_factory(n_steps: int, dt: float, rng: np.random.Generator):
    # UE starts at a random position along the Case A path (+x direction).
    start_x = float(rng.uniform(-_UE_PATH_HALF_LEN, _UE_PATH_HALF_LEN))
    return straight_line_track(start_xy=(start_x, _UE_PATH_Y),
                                heading=0.0,       # +x direction
                                speed_mps=10.0,    # 10 m/s per Fig 6.4 caption
                                n_steps=n_steps,
                                dt=dt)


def _channel_factory(rng: np.random.Generator, bs_index: int):
    params = ChannelParams(ue_speed_mps=10.0)
    return ChannelRealisation(params=params,
                               bs_xy=_BS_XY,
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
            bs_positions=[tuple(_BS_XY.tolist())],
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
    ax.set_ylabel(rf"Coverage rate ($\gamma_{{\mathrm{{th}}}}={GAMMA_TH_DB:.2f}$ dB)")
    ax.set_title(f"Case A: UMi, 10 m/s, n_trials={n_trials}")
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
    parser = argparse.ArgumentParser(
        description="Case A SNR sweep (predecessor Fig 6.4).")
    parser.add_argument("--n-trials", type=int, default=30)
    # 1 second at 1 ms = 1 000 steps (predecessor default, Section 6).
    parser.add_argument("--n-steps", type=int, default=1_000)
    # 46 points = 1-dB steps over -15 to +30 dB (predecessor Fig 6.4 range).
    # Use --n-snr-points 31 for faster runs (1.5-dB steps); note in caption.
    parser.add_argument("--n-snr-points", type=int, default=46)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    # x-axis spans -15 to +30 dB matching predecessor Fig 6.4.
    snr_grid = np.linspace(-15.0, 30.0, args.n_snr_points)
    out = run(args.n_trials, args.n_steps, args.output, snr_grid)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
