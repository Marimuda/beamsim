"""Experiment 1 — Case C: Mean received power vs. UE rotational velocity.

Faithfully reproduces predecessor MSc thesis Case C (Section 5.2.2 and
Section 6.2 / Fig 6.2).

Case C definition (predecessor Section 5.2.2):
  - Single UE, single BS, rotation only — Free Space (LOS only, no angular
    spread or fading).
  - UE and BS are co-located at a "short spacing" of 10 metres.
  - Both UE and BS rotate at the same angular velocity (same rpm).
  - Start angle is randomised per iteration for convergence (Monte Carlo).
  - Duration: 10 seconds per iteration (Section 5.5: "10 seconds for each
    iteration"), i.e. n_steps = 10 000 at dt = 1 ms.
  - Metric: mean post-combining received power in dB (noiseless scenario).

rpm sweep matches Fig 6.2 x-axis tick marks: {10, 20, 40, 60, 80, 120, 180}.
Fig 5.24 shows 5 rpm as the leftmost point; we therefore prepend 5 to the
sweep giving {5, 10, 20, 40, 60, 80, 120, 180} — 8 log-spaced values that
cover the range of both predecessor figures (Figs 5.24 and 6.2).

Number of trials: 10, matching the original "_10_itr" filename convention.
Users may override with --n-trials.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from beamsim.channel import FreeSpaceLosChannel
from beamsim.geometry import rotation_track
from beamsim.plotting import (
    ALGORITHM_LABELS,
    ALGORITHM_PALETTE,
    fig_single_column,
    save_figure,
    set_publication_style,
)
from beamsim.runner import Experiment, run_experiment


# Module-level factories (must be picklable for ProcessPoolExecutor).

def _track_factory(rpm: float, n_steps: int, dt: float, rng: np.random.Generator):
    initial_yaw = float(rng.uniform(-np.pi, np.pi))
    return rotation_track(position_xy=(0.0, 0.0),
                           rpm=rpm,
                           n_steps=n_steps,
                           dt=dt,
                           initial_orientation=initial_yaw)


def _channel_factory(rng: np.random.Generator, bs_index: int):
    # Case C: BS at 10 m (predecessor "short spacing", Section 5.2.2).
    return FreeSpaceLosChannel(bs_xy=np.array([10.0, 0.0]),
                                bs_yaw=0.0,
                                n_bs_elements=16,
                                n_ue_elements=4)


def run(n_trials: int, n_steps: int, output_dir: Path, rpm_values: list[float]):
    output_dir.mkdir(parents=True, exist_ok=True)
    dt = 1e-3   # 1 kHz

    # Mean output power per algorithm per rpm: shape (n_rpm, n_trials)
    n_rpm = len(rpm_values)
    algorithms = ["exhaustive", "nns", "tabu", "angular_prediction", "ci", "mcmd"]

    # Collect mean output SNR (dB) per (rpm, trial, algo).
    mean_snr_per_rpm: dict[str, np.ndarray] = {a: np.zeros((n_rpm, n_trials)) for a in algorithms}

    for i, rpm in enumerate(rpm_values):
        exp = Experiment(
            name=f"rotational_rpm_{rpm:g}",
            n_steps=n_steps,
            dt=dt,
            n_trials=n_trials,
            algorithms=algorithms,
            bs_positions=[(10.0, 0.0)],   # Case C: BS at 10 m
            bs_yaws=[0.0],
            track_factory=partial(_track_factory, rpm, n_steps, dt),
            channel_factory=_channel_factory,
            noise_amplitude=1e-3,    # very low noise (rotation test is "noiseless")
            tx_amp=1.0,
            seed=12345 + i,
        )
        result = run_experiment(exp, progress=True)
        # Save per-rpm raw traces
        from beamsim.runner import save_experiment
        save_experiment(result, output_dir / f"rotational_rpm_{rpm:g}.npz")
        # Aggregate: mean SNR over occasions per trial
        for a in algorithms:
            mean_snr_per_rpm[a][i] = result["snr_db"][a].mean(axis=1)

    # Plot
    set_publication_style()
    fig = fig_single_column()
    ax = fig.add_subplot(111)
    rpm_arr = np.array(rpm_values, dtype=float)
    rng = np.random.default_rng(0)
    n_boot = 1000
    for a in algorithms:
        traces = mean_snr_per_rpm[a]   # (n_rpm, n_trials)
        mean_curve = traces.mean(axis=1)
        boot_means = np.empty((n_boot, traces.shape[0]))
        for b in range(n_boot):
            idx = rng.integers(0, traces.shape[1], size=traces.shape[1])
            boot_means[b] = traces[:, idx].mean(axis=1)
        lo = np.percentile(boot_means, 2.5, axis=0)
        hi = np.percentile(boot_means, 97.5, axis=0)
        color = ALGORITHM_PALETTE[a]
        ax.plot(rpm_arr, mean_curve, color=color, label=ALGORITHM_LABELS[a])
        ax.fill_between(rpm_arr, lo, hi, color=color, alpha=0.20, linewidth=0)

    ax.set_xscale("log")
    ax.set_xlabel("Rotational velocity (rpm)")
    ax.set_ylabel("Mean output SNR (dB)")
    ax.set_title(f"Case C: free-space LOS rotation, BS=10 m, n_trials={n_trials}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)

    out_pdf = output_dir / "rotational_velocity_with_ci.pdf"
    save_figure(fig, out_pdf)
    plt.close(fig)
    # Also save aggregated data for the paper
    np.savez_compressed(output_dir / "rotational_aggregate.npz",
                        rpm=rpm_arr,
                        **{f"mean_snr_db/{a}": mean_snr_per_rpm[a] for a in algorithms})
    return out_pdf


def main():
    parser = argparse.ArgumentParser(
        description="Case C: free-space rotation sweep (predecessor Fig 6.2).")
    # Default 10 trials matches predecessor "_10_itr" naming convention.
    parser.add_argument("--n-trials", type=int, default=10)
    # 10 seconds at 1 ms = 10 000 steps (predecessor Section 5.5).
    parser.add_argument("--n-steps", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    # rpm values matching predecessor Fig 5.24 and Fig 6.2 x-axis ticks.
    # Leftmost point in Fig 5.24 is 5 rpm; Fig 6.2 starts at 10 rpm.
    # We include both: {5, 10, 20, 40, 60, 80, 120, 180}.
    rpm_values = [5, 10, 20, 40, 60, 80, 120, 180]
    out = run(args.n_trials, args.n_steps, args.output, rpm_values)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
