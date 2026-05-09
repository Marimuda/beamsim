"""Experiment 2 — Case A: Coverage rate vs. measurement-rate factor alpha.

.. deprecated::
    Prefer the Hydra entry point::

        beamsim-run --config-name alpha [run.n_trials=30]

    This script is kept as a backward-compatible shim.

Faithfully reproduces predecessor MSc thesis Case A (Section 5.2.2 and
Section 6.3 / Fig 6.3).

Case A definition (predecessor Section 5.2.2):
  - Single UE, Single BS, straight-line path — 3GPP UMi channel model.
  - UE moves a length corresponding to IBS = 200 m at a 3/4 IBS lateral
    offset.  The largest UE-to-BS distance is therefore IBS/2 = 100 m.
    BS is placed at (100, 0) m and the UE path runs parallel at y = 150 m.
  - UE speed: 1 m/s (Fig 6.3 caption: "1 m/s UE speed").
  - Single-antenna input SNR target: 10 dB (Fig 6.3 caption).

Fig 6.3 x-axis spans alpha = 0 to 12 (alpha=0 is "no measurements" and is
not a meaningful operating point, so we start at alpha=1).
Alpha sweep: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}.

Coverage threshold: output SNR > −9.53 dB (CQI-1 floor, Sec 5.3.4 of the
report).  The report's Fig 6.3 caption states "Coverage rate here defined
as when output SNR is above 10 dB" — that value was tied to the input SNR
target and differs from the CQI-1 floor used here.  This is a documented
divergence; the qualitative shape is preserved but absolute coverage values
will differ slightly from Fig 6.3.  See commit 2cc1437 ("alpha-sweep label:
gamma_th displayed as -9.53 (2 d.p.) not -10 (rounded)") for history.

Number of trials: 30, matching the predecessor's "_30_itr" naming convention
(consistent with default for Case A simulations in the report).
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beamsim.channel import ChannelParams, ChannelRealisation
from beamsim.geometry import straight_line_track
from beamsim.link_budget import tx_amp_for_snr_db
from beamsim.metrics import coverage_rate
from beamsim.plotting import (
    ALGORITHM_LABELS,
    ALGORITHM_PALETTE,
    bootstrap_ci,
    fig_single_column,
    save_figure,
    set_publication_style,
)
from beamsim.runner import Experiment, run_experiment, save_experiment

# CQI-1 floor (lowest E-UTRA CQI), report Sec 5.3.4.  The report's Fig 6.3
# used 10 dB (tied to input SNR target) — see module docstring for rationale.
GAMMA_TH_DB = -9.5335  # commit 2cc1437: -9.53 dB is correct, not rounded -10 dB


# Case A geometry constants (predecessor Section 5.2.2):
#   IBS = 200 m (UMi inter-BS spacing, Table 3GPP parameters in Section 3.2).
#   UE path runs at a 3/4 IBS lateral offset (y = 150 m).
#   BS is at the origin of its cell; UE closest approach is at x = 0.
#   Largest BS-to-UE distance along path ≈ IBS/2 = 100 m (used for
#   link-budget in Section 3.2.6 and for tx_amp calibration below).
_BS_XY_CASE_A = np.array([0.0, 0.0])
_UE_PATH_Y = 150.0  # 3/4 * IBS = 3/4 * 200 m
_UE_PATH_HALF_LEN = 100.0  # UE traverses IBS = 200 m, centred at x=0


def _track_factory(n_steps: int, dt: float, rng: np.random.Generator):
    # UE starts at a random position along the Case A path (x in [-100, 100])
    # and walks in the +x direction.  Random start gives Monte Carlo diversity
    # over angle-of-arrival profiles within the cell area.
    start_x = float(rng.uniform(-_UE_PATH_HALF_LEN, _UE_PATH_HALF_LEN))
    return straight_line_track(
        start_xy=(start_x, _UE_PATH_Y),
        heading=0.0,  # +x direction
        speed_mps=1.0,
        n_steps=n_steps,
        dt=dt,
    )


def _channel_factory(rng: np.random.Generator, bs_index: int):
    params = ChannelParams(ue_speed_mps=1.0)
    return ChannelRealisation(
        params=params, bs_xy=_BS_XY_CASE_A, bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4, rng=rng
    )


def _tx_amp_for_target_input_snr_db(
    target_db: float, noise_amplitude: float = 1e-3, n_ue: int = 4, n_bs: int = 16
) -> float:
    """Tx amplitude such that per-element SNR = target_db at IBS/2 = 100 m."""
    return tx_amp_for_snr_db(target_db, 100.0, 28e9, 10.0, 1.5, noise_amplitude, n_ue, n_bs)


def run(n_trials: int, output_dir: Path, alpha_values: list[float]):
    output_dir.mkdir(parents=True, exist_ok=True)
    base_rate_hz = 1000.0
    duration_s = 1.0
    algorithms = ["exhaustive", "nns", "tabu", "angular_prediction", "ci", "mcmd"]

    cr_per_alpha: dict[str, np.ndarray] = {
        a: np.zeros((len(alpha_values), n_trials)) for a in algorithms
    }
    tx_amp = _tx_amp_for_target_input_snr_db(10.0)

    for i, alpha in enumerate(alpha_values):
        rate = alpha * base_rate_hz
        n_steps = round(duration_s * rate)
        dt = 1.0 / rate
        exp = Experiment(
            name=f"alpha_{alpha:g}",
            n_steps=n_steps,
            dt=dt,
            n_trials=n_trials,
            algorithms=algorithms,
            bs_positions=[tuple(_BS_XY_CASE_A.tolist())],
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

    # Plot — "ci" is excluded from the figure (it stays in the simulation data
    # but is not plotted because Context Information is not a meaningful
    # comparator for Case-A coverage-rate vs alpha; its plateau near 0.9
    # distorts the y-axis range and misleads comparisons).
    plotted_algorithms = [a for a in algorithms if a != "ci"]

    set_publication_style()
    fig = fig_single_column()
    ax = fig.add_subplot(111)
    alpha_arr = np.array(alpha_values, dtype=float)
    rng = np.random.default_rng(1)
    n_boot = 1000
    for a in plotted_algorithms:
        traces = cr_per_alpha[a]
        mean_curve = traces.mean(axis=1)
        lo, hi = bootstrap_ci(traces.T, n_boot=n_boot, ci_alpha=0.05, rng=rng)
        color = ALGORITHM_PALETTE[a]
        ax.plot(alpha_arr, mean_curve, color=color, label=ALGORITHM_LABELS[a])
        ax.fill_between(alpha_arr, lo, hi, color=color, alpha=0.20, linewidth=0)

    ax.set_xlabel(r"Measurement-rate factor $\alpha$")
    ax.set_ylabel(rf"Coverage rate (SNR$_{{out}}$ > {GAMMA_TH_DB:.2f} dB)")
    ax.set_title(
        f"Case A: UMi, 1 m/s, input SNR 10 dB, n={n_trials} trials\n"
        r"$\gamma_{\mathrm{th}}=-9.53$ dB (CQI-1, Sec 5.3.4); report Fig 6.3 used $\gamma_{\mathrm{th}}=10$ dB"
    )
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    ax.set_ylim(0, 1)

    out_pdf = output_dir / "alphasw_coveragerate_1mps_with_ci.pdf"
    save_figure(fig, out_pdf)
    plt.close(fig)
    np.savez_compressed(
        output_dir / "alpha_aggregate.npz",
        alpha=alpha_arr,
        gamma_th_db=GAMMA_TH_DB,
        **{f"coverage_rate/{a}": cr_per_alpha[a] for a in algorithms},
    )
    return out_pdf


def main():
    parser = argparse.ArgumentParser(description="Case A: alpha sweep (predecessor Fig 6.3).")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    # Integer alpha 1-12 matching predecessor Fig 6.3 x-axis (Section 6.3).
    # alpha=0 is excluded (no measurements is not a meaningful data point).
    alpha_values = list(range(1, 13))
    out = run(args.n_trials, args.output, alpha_values)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
