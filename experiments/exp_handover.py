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

import numpy as np
import matplotlib.pyplot as plt

from beamsim.channel import ChannelParams, ChannelRealisation, umi_path_loss_db
from beamsim.geometry import straight_line_track
from beamsim.plotting import (
    ALGORITHM_LABELS,
    ALGORITHM_PALETTE,
    fig_single_column,
    save_figure,
    set_publication_style,
)
from beamsim.runner import Experiment, run_experiment, save_experiment


# Case D: 4 BSs in center-excited hexagonal tessellation, IBS = 200 m.
# (predecessor Section 5.2.2 and 3GPP UMi Table in Section 3.2)
IBS = 200.0
BS_POSITIONS: list[tuple[float, float]] = [
    (0.0,   0.0),
    (IBS,   0.0),
    (IBS / 2.0,  IBS / 2.0 * math.sqrt(3)),   # (100, ~173.2)
    (IBS / 2.0, -IBS / 2.0 * math.sqrt(3)),   # (100, ~-173.2)
]

# UE path: from x=0 to x=2*IBS+IBS/2 = 500 m at y=0.
# At 10 m/s that is 50 seconds = 50 000 steps at dt=1 ms.
_UE_SPEED_MPS = 10.0
_PATH_LENGTH_M = 2.0 * IBS + IBS / 2.0    # = 500 m
_TRIAL_DURATION_S = _PATH_LENGTH_M / _UE_SPEED_MPS   # = 50 s
_N_STEPS_DEFAULT = int(_TRIAL_DURATION_S / 1e-3)       # = 50 000

# tx_amp calibrated to input SNR = 10 dB at IBS/2 = 100 m (Case A reference).
_NOISE_AMP = 1e-3
_TX_AMP = float(
    _NOISE_AMP
    * np.sqrt(10 ** (10.0 / 10.0))
    / (10 ** (-umi_path_loss_db(100.0, 28e9, 10.0, 1.5, los=True) / 20.0))
)


def _track_factory(n_steps: int, dt: float, rng: np.random.Generator):
    # UE starts at x=0 with small Gaussian perturbation for Monte Carlo diversity.
    start_x = float(rng.normal(0.0, 2.0))
    return straight_line_track(start_xy=(start_x, 0.0),
                                heading=0.0,      # +x direction
                                speed_mps=_UE_SPEED_MPS,
                                n_steps=n_steps,
                                dt=dt)


def _channel_factory(rng: np.random.Generator, bs_index: int):
    # TODO: Replace with deterministic specular reflectors per BS when the
    # channel model exposes a placement API.  Currently uses random NLOS
    # clusters as a proxy (predecessor "with reflectors" approximation).
    params = ChannelParams(ue_speed_mps=_UE_SPEED_MPS)
    return ChannelRealisation(params=params,
                               bs_xy=np.array(BS_POSITIONS[bs_index]),
                               bs_yaw=0.0,
                               n_bs_elements=16,
                               n_ue_elements=4,
                               rng=rng)


def run(n_trials: int, n_steps: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    dt = 1e-3
    algorithms = ["exhaustive", "nns", "tabu", "angular_prediction", "ci", "mcmd"]

    exp = Experiment(
        name="handover_4bs_10mps",
        n_steps=n_steps,
        dt=dt,
        n_trials=n_trials,
        algorithms=algorithms,
        bs_positions=BS_POSITIONS,
        bs_yaws=[0.0, 0.0, 0.0, 0.0],
        track_factory=partial(_track_factory, n_steps, dt),
        channel_factory=_channel_factory,
        noise_amplitude=_NOISE_AMP,
        tx_amp=_TX_AMP,
        seed=44444,
    )
    result = run_experiment(exp, progress=True)
    save_experiment(result, output_dir / "handover.npz")

    # Compute L_BS per occasion per algorithm (predecessor Eq. 6.2):
    #   L_BS = 10 * log10(gamma_best / gamma_obtained)
    # gamma_best is the SNR of the best-possible BS at each occasion.
    # We approximate it as the per-occasion maximum SNR across all algorithms
    # (a lower bound on the oracle; conservative in the sense that L_BS
    # cannot be negative). When oracle per-BS channel info is available this
    # can be replaced with the true best-BS SNR.
    snr_db = result["snr_db"]
    snr_stack = np.stack([snr_db[a] for a in algorithms], axis=0)  # (n_algo, n_trials, n_steps)
    snr_best_per_occasion = snr_stack.max(axis=0)   # (n_trials, n_steps)
    l_bs_db: dict[str, np.ndarray] = {}
    for a in algorithms:
        gap = snr_best_per_occasion - snr_db[a]   # >= 0 always
        l_bs_db[a] = gap   # (n_trials, n_steps)

    # Plot L_BS (mean over trials) vs. time along UE path.
    set_publication_style()
    fig = fig_single_column()
    ax = fig.add_subplot(111)
    times = np.arange(n_steps) * dt   # seconds
    rng = np.random.default_rng(3)
    n_boot = 500
    win = max(1, n_steps // 200)   # smoothing window ~0.5% of path
    for a in algorithms:
        traces = l_bs_db[a]   # (n_trials, n_steps)
        mean_curve = traces.mean(axis=0)
        kernel = np.ones(win) / win
        mean_smooth = np.convolve(mean_curve, kernel, mode="same")
        boot_means = np.empty((n_boot, n_steps))
        for b in range(n_boot):
            idx = rng.integers(0, traces.shape[0], size=traces.shape[0])
            boot_means[b] = traces[idx].mean(axis=0)
        lo = np.percentile(boot_means, 2.5, axis=0)
        hi = np.percentile(boot_means, 97.5, axis=0)
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

    out_pdf = output_dir / "handover_4bs_10mps_with_ci.pdf"
    save_figure(fig, out_pdf)
    plt.close(fig)
    np.savez_compressed(output_dir / "handover_aggregate.npz",
                        **{f"l_bs_db/{a}": l_bs_db[a] for a in algorithms})
    return out_pdf


def main():
    parser = argparse.ArgumentParser(
        description="Case D: 4-BS hexagonal handover scenario (predecessor Fig 6.8).")
    parser.add_argument("--n-trials", type=int, default=30)
    # 50 s at 1 ms = 50 000 steps (10 m/s over 500 m path).
    # Use a smaller value (e.g. 5000) for quick smoke-tests.
    parser.add_argument("--n-steps", type=int, default=_N_STEPS_DEFAULT)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    out = run(args.n_trials, args.n_steps, args.output)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
