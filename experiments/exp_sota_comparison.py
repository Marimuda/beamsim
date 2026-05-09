"""SOTA comparison experiment — Case A UMi 10 m/s SNR sweep.

Extends the SNR-sweep baseline to include DL predictor alongside the other
SOTA algorithms (HBM, OMP, UCB1, Thompson) and the predecessor baselines.

Run a quick smoke sweep (n_trials=5, n_steps=500) with a single SNR point
(+20 dB) to confirm the DL predictor pipeline produces sensible output.

Usage
-----
    python experiments/exp_sota_comparison.py [--full] [--output results/sota]
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np

from beamsim.channel import ChannelParams, ChannelRealisation, umi_path_loss_db
from beamsim.geometry import straight_line_track
from beamsim.runner import Experiment, run_experiment, save_experiment

_BS_XY = np.array([0.0, 0.0])
_UE_PATH_Y = 150.0
_UE_PATH_HALF_LEN = 100.0
NOISE_AMP = 1e-3
DISTANCE_M = 100.0


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
    import math

    pl_db = umi_path_loss_db(DISTANCE_M, 28e9, 10.0, 1.5, los=True)
    pl_lin = 10 ** (-pl_db / 20.0)
    target_lin = 10 ** (target_db / 10.0)
    return float(NOISE_AMP * math.sqrt(n_ue * n_bs * target_lin) / pl_lin)


def run(
    n_trials: int,
    n_steps: int,
    output_dir: Path,
    snr_db_values: list[float],
    algorithms: list[str],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    dt = 1e-3

    results_by_snr = {}
    for snr_db in snr_db_values:
        tx_amp = _tx_amp_for(snr_db)
        exp = Experiment(
            name=f"sota_snr_{snr_db:+.0f}",
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
            seed=99999,
        )
        result = run_experiment(exp, progress=True)
        save_experiment(result, output_dir / f"sota_snr_{snr_db:+.0f}.npz")
        results_by_snr[snr_db] = result

        # Print mean SNR summary
        print(f"\nSNR input = {snr_db:+.0f} dB  (n_trials={n_trials}, n_steps={n_steps})")
        print(f"  {'Algorithm':<26}  Mean output SNR (dB)")
        print(f"  {'-' * 26}  {'-' * 20}")
        for a in algorithms:
            mean_snr = float(result["snr_db"][a].mean())
            print(f"  {a:<26}  {mean_snr:+.2f}")

    return results_by_snr


def main() -> None:
    parser = argparse.ArgumentParser(description="SOTA DL-predictor comparison (Case A)")
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("results/sota"))
    parser.add_argument(
        "--full", action="store_true", help="Full 31-point SNR sweep instead of single +20 dB point"
    )
    args = parser.parse_args()

    algorithms = [
        "exhaustive",
        "nns",
        "angular_prediction",
        "ci",
        "mcmd",
        "ucb1",
        "thompson",
        "hbm",
        "omp_compressive",
        "dl_predictor",
    ]

    snr_values = list(np.arange(-15, 31, 1.5).tolist()) if args.full else [20.0]
    run(args.n_trials, args.n_steps, args.output, snr_values, algorithms)


if __name__ == "__main__":
    main()
