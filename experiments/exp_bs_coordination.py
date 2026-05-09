"""Experiment: BS/UE coordination — Fig. 6.7 (report Sec. 6.5).

Case C (free-space LOS rotation, single BS at 10 m) with four algorithms:
  - Exhaustive
  - NNS
  - Perfect knowledge
  - NNS-BS-sequential (BS beam follows fixed stride-7 round-robin)

Qualitatively, NNS-BS-sequential should sit between Exhaustive (lowest SNR)
and NNS (best), converging toward Exhaustive at high rpm where losing BS-beam
coordination costs the most (~2 dB advantage vs ~10 dB for full NNS).

Outputs:
  results/bs_coordination_aggregate.npz
  results/bs_coordination_with_ci.pdf
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beamsim.channel import FreeSpaceLosChannel
from beamsim.geometry import rotation_track
from beamsim.plotting import (
    ALGORITHM_LABELS,
    ALGORITHM_PALETTE,
    bootstrap_ci,
    fig_single_column,
    save_figure,
    set_publication_style,
)
from beamsim.runner import Experiment, run_experiment

# ---------------------------------------------------------------------------
# Picklable factories (shared with exp_rotational — must stay module-level)
# ---------------------------------------------------------------------------


def _track_factory(rpm: float, n_steps: int, dt: float, rng: np.random.Generator):
    initial_yaw = float(rng.uniform(-np.pi, np.pi))
    return rotation_track(
        position_xy=(0.0, 0.0), rpm=rpm, n_steps=n_steps, dt=dt, initial_orientation=initial_yaw
    )


def _channel_factory(rng: np.random.Generator, bs_index: int):
    return FreeSpaceLosChannel(
        bs_xy=np.array([10.0, 0.0]), bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4
    )


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------


def run(n_trials: int, n_steps: int, output_dir: Path, rpm_values: list[float]):
    output_dir.mkdir(parents=True, exist_ok=True)
    dt = 1e-3  # 1 kHz

    algorithms = ["exhaustive", "nns", "perfect", "nns_bs_sequential"]
    n_rpm = len(rpm_values)
    mean_snr_per_rpm: dict[str, np.ndarray] = {a: np.zeros((n_rpm, n_trials)) for a in algorithms}

    for i, rpm in enumerate(rpm_values):
        exp = Experiment(
            name=f"bs_coordination_rpm_{rpm:g}",
            n_steps=n_steps,
            dt=dt,
            n_trials=n_trials,
            algorithms=algorithms,
            bs_positions=[(10.0, 0.0)],
            bs_yaws=[0.0],
            track_factory=partial(_track_factory, rpm, n_steps, dt),
            channel_factory=_channel_factory,
            noise_amplitude=1e-3,
            tx_amp=1.0,
            seed=99000 + i,
        )
        result = run_experiment(exp, progress=True)
        for a in algorithms:
            mean_snr_per_rpm[a][i] = result["snr_db"][a].mean(axis=1)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    set_publication_style()
    fig = fig_single_column()
    ax = fig.add_subplot(111)
    rpm_arr = np.array(rpm_values, dtype=float)
    rng = np.random.default_rng(0)
    n_boot = 1000

    for a in algorithms:
        traces = mean_snr_per_rpm[a]  # (n_rpm, n_trials)
        mean_curve = traces.mean(axis=1)
        lo, hi = bootstrap_ci(traces.T, n_boot=n_boot, ci_alpha=0.05, rng=rng)
        color = ALGORITHM_PALETTE[a]
        ax.plot(rpm_arr, mean_curve, color=color, label=ALGORITHM_LABELS[a])
        ax.fill_between(rpm_arr, lo, hi, color=color, alpha=0.20, linewidth=0)

    ax.set_xscale("log")
    ax.set_xlabel("Rotational velocity (rpm)")
    ax.set_ylabel("Mean output SNR (dB)")
    ax.set_title(
        f"Fig. 6.7: BS/UE coordination — Case C, n_trials={n_trials}",
        fontsize=7,
    )
    ax.legend(fontsize=7, ncol=1)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)

    out_pdf = output_dir / "bs_coordination_with_ci.pdf"
    save_figure(fig, out_pdf)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Aggregate NPZ
    # -----------------------------------------------------------------------
    np.savez_compressed(
        output_dir / "bs_coordination_aggregate.npz",
        rpm=rpm_arr,
        **{f"mean_snr_db/{a}": mean_snr_per_rpm[a] for a in algorithms},
    )

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    print("\nPer-rpm mean SNR summary (mean across trials):")
    print(f"{'rpm':>6}  " + "  ".join(f"{a:>20}" for a in algorithms))
    for i, rpm in enumerate(rpm_values):
        vals = "  ".join(f"{mean_snr_per_rpm[a][i].mean():>20.2f}" for a in algorithms)
        print(f"{rpm:>6g}  {vals}")

    return out_pdf


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fig. 6.7: BS/UE coordination experiment (Case C)."
    )
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    rpm_values = [5, 10, 20, 40, 60, 80, 120, 180]
    out = run(args.n_trials, args.n_steps, args.output, rpm_values)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
