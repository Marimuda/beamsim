"""Minimal end-to-end beamsim example.

Runs a tiny rotational experiment with three algorithms and prints the mean
post-combining SNR for each. Designed to finish in well under 30 seconds on
a laptop, with no figure rendering and no external assets.

Usage:

    python examples/minimal_example.py
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import numpy as np

from beamsim import (
    Experiment,
    FreeSpaceLosChannel,
    run_experiment,
    save_experiment,
)
from beamsim.geometry import rotation_track

logger = logging.getLogger("beamsim.examples.minimal")


def _rotation_track(rpm: float, n_steps: int, dt: float, rng: np.random.Generator):
    initial_yaw = float(rng.uniform(-np.pi, np.pi))
    return rotation_track(
        position_xy=(0.0, 0.0),
        rpm=rpm,
        n_steps=n_steps,
        dt=dt,
        initial_orientation=initial_yaw,
    )


def _freespace_channel(rng: np.random.Generator, bs_index: int) -> FreeSpaceLosChannel:
    return FreeSpaceLosChannel(
        bs_xy=np.array([100.0, 0.0]),
        bs_yaw=0.0,
        n_bs_elements=16,
        n_ue_elements=4,
    )


def main(output_path: Path | None = None) -> dict[str, float]:
    """Run a minimal Monte Carlo experiment and return mean per-algo SNR (dB).

    Parameters
    ----------
    output_path:
        If provided, also writes the full result archive to this path.

    Returns
    -------
    Mapping ``{algorithm_name -> mean SNR in dB across trials and steps}``.
    """
    n_steps = 200
    dt = 1e-3
    n_trials = 4
    algorithms = ["exhaustive", "nns", "mcmd"]

    exp = Experiment(
        name="minimal_example",
        n_steps=n_steps,
        dt=dt,
        n_trials=n_trials,
        algorithms=algorithms,
        bs_positions=[(100.0, 0.0)],
        bs_yaws=[0.0],
        track_factory=partial(_rotation_track, 60.0, n_steps, dt),
        channel_factory=_freespace_channel,
        noise_amplitude=1e-3,
        tx_amp=1.0,
        seed=0,
    )

    result = run_experiment(exp, n_workers=1, progress=False)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_experiment(result, output_path)

    snr_db_per_algo = result["snr_db"]
    mean_snr = {a: float(np.nanmean(snr_db_per_algo[a])) for a in algorithms}
    return mean_snr


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    means = main()
    for algo, snr in means.items():
        logger.info("%-12s mean SNR = %+.2f dB", algo, snr)
