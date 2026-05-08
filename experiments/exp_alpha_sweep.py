"""Experiment 2: Coverage rate vs. measurement-rate factor alpha.

UMi LOS at 1 m/s, fixed BS at d_2D = 50 m, single-antenna SNR target = 10 dB.
Sweep alpha in {0.5, 1, 2, 4, 8} corresponding to measurement rates of
500 Hz to 8 kHz. Trial duration is held at 1 second so n_steps scales with
alpha; the channel evolves at fixed wall-clock rate while measurements
become more frequent.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from beamsim.channel import ChannelParams, ChannelRealisation
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


GAMMA_TH_DB = -9.5335   # predecessor convention (lowest E-UTRA CQI)


def _track_factory(n_steps: int, dt: float, rng: np.random.Generator):
    start_x = float(rng.uniform(-2.0, 2.0))
    start_y = float(rng.uniform(-2.0, 2.0))
    heading = float(rng.uniform(-np.pi, np.pi))
    return straight_line_track(start_xy=(start_x, start_y),
                                heading=heading,
                                speed_mps=1.0,
                                n_steps=n_steps,
                                dt=dt)


def _channel_factory(rng: np.random.Generator, bs_index: int):
    params = ChannelParams(ue_speed_mps=1.0)
    return ChannelRealisation(params=params,
                               bs_xy=np.array([50.0, 0.0]),
                               bs_yaw=0.0,
                               n_bs_elements=16,
                               n_ue_elements=4,
                               rng=rng)


def _tx_amp_for_target_input_snr_db(target_db: float, distance_m: float = 50.0,
                                      noise_amplitude: float = 1e-3) -> float:
    """Pick tx_amp so the post-path-loss single-antenna SNR meets the target.

    Single-antenna SNR (linear) = (tx_amp * 10**(-PL/20))**2 / sigma_n^2.
    """
    from beamsim.channel import umi_path_loss_db
    pl_db = umi_path_loss_db(distance_m, 28e9, 10.0, 1.5, los=True)
    pl_lin = 10 ** (-pl_db / 20.0)
    target_lin = 10 ** (target_db / 10.0)
    return float(noise_amplitude * np.sqrt(target_lin) / pl_lin)


def run(n_trials: int, output_dir: Path, alpha_values: list[float]):
    output_dir.mkdir(parents=True, exist_ok=True)
    base_rate_hz = 1000.0
    duration_s = 1.0
    algorithms = ["exhaustive", "nns", "tabu", "angular_prediction", "ci", "mcmd"]

    cr_per_alpha: dict[str, np.ndarray] = {a: np.zeros((len(alpha_values), n_trials)) for a in algorithms}
    tx_amp = _tx_amp_for_target_input_snr_db(10.0)

    for i, alpha in enumerate(alpha_values):
        rate = alpha * base_rate_hz
        n_steps = int(round(duration_s * rate))
        dt = 1.0 / rate
        exp = Experiment(
            name=f"alpha_{alpha:g}",
            n_steps=n_steps,
            dt=dt,
            n_trials=n_trials,
            algorithms=algorithms,
            bs_positions=[(50.0, 0.0)],
            bs_yaws=[0.0],
            track_factory=partial(_track_factory, n_steps, dt),
            channel_factory=_channel_factory,
            noise_amplitude=1e-3,
            tx_amp=tx_amp,
            seed=22222 + i,
        )
        result = run_experiment(exp, progress=True)
        save_experiment(result, output_dir / f"alpha_{alpha:g}.npz")
        for a in algorithms:
            cr_per_alpha[a][i] = coverage_rate(result["snr_db"][a], GAMMA_TH_DB)

    # Plot
    set_publication_style()
    fig = fig_single_column()
    ax = fig.add_subplot(111)
    alpha_arr = np.array(alpha_values, dtype=float)
    rng = np.random.default_rng(1)
    n_boot = 1000
    for a in algorithms:
        traces = cr_per_alpha[a]
        mean_curve = traces.mean(axis=1)
        boot_means = np.empty((n_boot, traces.shape[0]))
        for b in range(n_boot):
            idx = rng.integers(0, traces.shape[1], size=traces.shape[1])
            boot_means[b] = traces[:, idx].mean(axis=1)
        lo = np.percentile(boot_means, 2.5, axis=0)
        hi = np.percentile(boot_means, 97.5, axis=0)
        color = ALGORITHM_PALETTE[a]
        ax.plot(alpha_arr, mean_curve, color=color, label=ALGORITHM_LABELS[a])
        ax.fill_between(alpha_arr, lo, hi, color=color, alpha=0.20, linewidth=0)

    ax.set_xscale("log")
    ax.set_xlabel(r"Measurement-rate factor $\alpha$")
    ax.set_ylabel(rf"Coverage rate ($\gamma_{{\mathrm{{th}}}}={GAMMA_TH_DB}$ dB)")
    ax.set_title(f"UMi LOS, 1 m/s, 10 dB single-antenna SNR, n={n_trials} trials")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    ax.set_ylim(0, 1)

    out_pdf = output_dir / "alphasw_coveragerate_1mps_with_ci.pdf"
    save_figure(fig, out_pdf)
    plt.close(fig)
    np.savez_compressed(output_dir / "alpha_aggregate.npz",
                        alpha=alpha_arr, gamma_th_db=GAMMA_TH_DB,
                        **{f"coverage_rate/{a}": cr_per_alpha[a] for a in algorithms})
    return out_pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    out = run(args.n_trials, args.output, [0.5, 1.0, 2.0, 4.0, 8.0])
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
