"""Unified Hydra entry point: ``python -m beamsim.run`` or ``beamsim-run``.

Each experiment config composes:
  - configs/scenario/<name>.yaml
  - configs/sweep/<name>.yaml
  - configs/algo/<name>.yaml  (optional)
  - run.* overrides

Usage examples::

    beamsim-run --config-name experiment/rotational
    beamsim-run --config-name experiment/rotational run.n_trials=2 run.n_steps=100
    beamsim-run --config-name experiment/alpha     run.n_trials=5

The ``--config-path`` defaults to ``../../configs`` relative to this file,
which resolves to the top-level ``configs/`` directory whether the package is
installed in editable mode or from a wheel.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from beamsim.channel import (
    ChannelParams,
    ChannelRealisation,
    FreeSpaceLosChannel,
)
from beamsim.geometry import Track, rotation_track, straight_line_track
from beamsim.link_budget import tx_amp_for_snr_db
from beamsim.runner import Experiment, run_experiment, save_experiment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Picklable track factories (module-level so ProcessPoolExecutor can pickle)
# ---------------------------------------------------------------------------


def _rotation_track_factory(
    rpm: float,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
) -> Track:
    initial_yaw = float(rng.uniform(-np.pi, np.pi))
    return rotation_track(
        position_xy=(0.0, 0.0),
        rpm=rpm,
        n_steps=n_steps,
        dt=dt,
        initial_orientation=initial_yaw,
    )


def _straight_line_track_factory(
    ue_speed_mps: float,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
) -> Track:
    """Straight-line track along +x with random start in [-100, 100] m."""
    start_x = float(rng.uniform(-100.0, 100.0))
    return straight_line_track(
        start_xy=(start_x, 150.0),
        heading=0.0,
        speed_mps=ue_speed_mps,
        n_steps=n_steps,
        dt=dt,
    )


def _case_d_track_factory(
    ue_speed_mps: float,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
) -> Track:
    """Case D: straight line from x~0 with small Gaussian perturbation."""
    start_x = float(rng.normal(0.0, 2.0))
    return straight_line_track(
        start_xy=(start_x, 0.0),
        heading=0.0,
        speed_mps=ue_speed_mps,
        n_steps=n_steps,
        dt=dt,
    )


# ---------------------------------------------------------------------------
# Picklable channel factories
# ---------------------------------------------------------------------------


def _freespace_los_channel_factory(
    bs_positions: list[tuple[float, float]],
    bs_yaws: list[float],
    rng: np.random.Generator,
    bs_index: int,
) -> FreeSpaceLosChannel:
    return FreeSpaceLosChannel(
        bs_xy=np.array(bs_positions[bs_index]),
        bs_yaw=bs_yaws[bs_index],
        n_bs_elements=16,
        n_ue_elements=4,
    )


def _geometric_umi_channel_factory(
    bs_positions: list[tuple[float, float]],
    bs_yaws: list[float],
    ue_speed_mps: float,
    disable_clusters: bool,
    rng: np.random.Generator,
    bs_index: int,
) -> ChannelRealisation:
    params = ChannelParams(
        ue_speed_mps=ue_speed_mps,
        disable_clusters=disable_clusters,
    )
    return ChannelRealisation(
        params=params,
        bs_xy=np.array(bs_positions[bs_index]),
        bs_yaw=bs_yaws[bs_index],
        n_bs_elements=16,
        n_ue_elements=4,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# tx_amp helper (calibrate to target per-element input SNR at 100 m)
# ---------------------------------------------------------------------------

_NOISE_AMP = 1e-3


def _tx_amp_for_snr(target_db: float, n_ue: int = 4, n_bs: int = 16) -> float:
    """Return tx_amp such that per-element SNR == target_db at 100 m (IBS/2)."""
    return tx_amp_for_snr_db(target_db, 100.0, 28e9, 10.0, 1.5, _NOISE_AMP, n_ue, n_bs)


# ---------------------------------------------------------------------------
# Factory builders from config
# ---------------------------------------------------------------------------


def _build_track_factory(
    scenario: Any,
    n_steps: int,
    dt: float,
    rpm: float | None = None,
) -> Any:
    """Return a picklable (rng) -> Track callable for the given scenario."""
    track_kind: str = scenario.ue_track_kind
    if track_kind == "rotation":
        _rpm = rpm if rpm is not None else 0.0
        return partial(_rotation_track_factory, _rpm, n_steps, dt)
    if track_kind == "straight_line":
        case: str = scenario.case
        if case == "case_d":
            return partial(_case_d_track_factory, float(scenario.ue_speed_mps), n_steps, dt)
        return partial(_straight_line_track_factory, float(scenario.ue_speed_mps), n_steps, dt)
    raise ValueError(f"Unknown ue_track_kind: {track_kind!r}")


def _build_channel_factory(scenario: Any) -> Any:
    """Return a picklable (rng, bs_index) -> channel callable."""
    bs_positions = [list(p) for p in scenario.bs_positions]
    bs_yaws = list(scenario.bs_yaws)
    pos_tuples = [tuple(p) for p in bs_positions]

    if scenario.channel_kind == "freespace_los":
        return partial(_freespace_los_channel_factory, pos_tuples, bs_yaws)

    if scenario.channel_kind == "geometric_umi":
        opts = OmegaConf.to_container(scenario.channel_options, resolve=True) or {}
        disable_clusters = bool(opts.get("disable_clusters", False))  # type: ignore[union-attr]  # OmegaConf.to_container returns dict at runtime
        return partial(
            _geometric_umi_channel_factory,
            pos_tuples,
            bs_yaws,
            float(scenario.ue_speed_mps),
            disable_clusters,
        )
    raise ValueError(f"Unknown channel_kind: {scenario.channel_kind!r}")


def _build_tx_amp(scenario: Any, sweep: Any, sweep_point: float | None) -> float:
    """Compute tx_amp: uses sweep.target_snr_db for calibration, or defaults."""
    variable: str = sweep.variable
    if variable == "snr_db" and sweep_point is not None:
        return _tx_amp_for_snr(float(sweep_point))
    target = sweep.target_snr_db
    if target is not None:
        return _tx_amp_for_snr(float(target))
    if scenario.channel_kind == "freespace_los":
        return 1.0
    return _tx_amp_for_snr(10.0)


# ---------------------------------------------------------------------------
# Sweep runner: dispatches on sweep.variable
# ---------------------------------------------------------------------------


def run_from_config(cfg: DictConfig) -> None:
    """Build and run an experiment (or sweep) from a resolved Hydra config."""
    scenario = cfg.scenario
    sweep = cfg.sweep
    run_cfg = cfg.run

    output_dir = Path(run_cfg.output_path) / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)

    algorithms = list(run_cfg.algorithms)
    n_trials = int(run_cfg.n_trials)
    seed = int(run_cfg.seed)
    n_workers = run_cfg.n_workers  # None or int

    channel_factory = _build_channel_factory(scenario)

    sweep_var: str = sweep.variable

    sweep_vals = list(sweep.sweep_values)

    if sweep_var == "none" or not sweep_vals:
        # Single-point experiment
        n_steps = int(run_cfg.n_steps)
        dt = float(scenario.dt)
        track_factory = _build_track_factory(scenario, n_steps, dt)
        tx_amp = _build_tx_amp(scenario, sweep, None)
        exp = Experiment(
            name=cfg.name,
            n_steps=n_steps,
            dt=dt,
            n_trials=n_trials,
            algorithms=algorithms,
            bs_positions=[tuple(p) for p in scenario.bs_positions],
            bs_yaws=list(scenario.bs_yaws),
            track_factory=track_factory,
            channel_factory=channel_factory,
            noise_amplitude=_NOISE_AMP,
            tx_amp=tx_amp,
            seed=seed,
        )
        result = run_experiment(exp, n_workers=n_workers, progress=True)
        save_experiment(result, output_dir / f"{cfg.name}.npz")

    elif sweep_var == "rpm":
        for i, rpm in enumerate(sweep_vals):
            n_steps = int(run_cfg.n_steps)
            dt = float(scenario.dt)
            track_factory = _build_track_factory(scenario, n_steps, dt, rpm=float(rpm))
            tx_amp = _build_tx_amp(scenario, sweep, None)
            exp = Experiment(
                name=f"{cfg.name}_rpm_{rpm:g}",
                n_steps=n_steps,
                dt=dt,
                n_trials=n_trials,
                algorithms=algorithms,
                bs_positions=[tuple(p) for p in scenario.bs_positions],
                bs_yaws=list(scenario.bs_yaws),
                track_factory=track_factory,
                channel_factory=channel_factory,
                noise_amplitude=_NOISE_AMP,
                tx_amp=tx_amp,
                seed=seed + i,
            )
            result = run_experiment(exp, n_workers=n_workers, progress=True)
            save_experiment(result, output_dir / f"rpm_{rpm:g}.npz")

    elif sweep_var == "alpha":
        base_rate_hz = 1000.0
        duration_s = 1.0
        for i, alpha in enumerate(sweep_vals):
            rate = float(alpha) * base_rate_hz
            n_steps = round(duration_s * rate)
            dt = 1.0 / rate
            track_factory = _build_track_factory(scenario, n_steps, dt)
            tx_amp = _build_tx_amp(scenario, sweep, None)
            exp = Experiment(
                name=f"{cfg.name}_alpha_{alpha:g}",
                n_steps=n_steps,
                dt=dt,
                n_trials=n_trials,
                algorithms=algorithms,
                bs_positions=[tuple(p) for p in scenario.bs_positions],
                bs_yaws=list(scenario.bs_yaws),
                track_factory=track_factory,
                channel_factory=channel_factory,
                noise_amplitude=_NOISE_AMP,
                tx_amp=tx_amp,
                seed=seed + i,
            )
            result = run_experiment(exp, n_workers=n_workers, progress=True)
            save_experiment(result, output_dir / f"alpha_{alpha:g}.npz")

    elif sweep_var == "snr_db":
        n_steps = int(run_cfg.n_steps)
        dt = float(scenario.dt)
        track_factory = _build_track_factory(scenario, n_steps, dt)
        for i, snr_db in enumerate(sweep_vals):
            tx_amp = _build_tx_amp(scenario, sweep, float(snr_db))
            exp = Experiment(
                name=f"{cfg.name}_snr_{snr_db:+.1f}",
                n_steps=n_steps,
                dt=dt,
                n_trials=n_trials,
                algorithms=algorithms,
                bs_positions=[tuple(p) for p in scenario.bs_positions],
                bs_yaws=list(scenario.bs_yaws),
                track_factory=track_factory,
                channel_factory=channel_factory,
                noise_amplitude=_NOISE_AMP,
                tx_amp=tx_amp,
                seed=seed + i,
            )
            result = run_experiment(exp, n_workers=n_workers, progress=False)
            save_experiment(result, output_dir / f"snr_{snr_db:+.1f}.npz")
            logger.info("[snr_sweep] %d/%d : SNR=%+.1f dB done", i + 1, len(sweep_vals), snr_db)

    else:
        raise ValueError(f"Unknown sweep variable: {sweep_var!r}")

    logger.info("Results written to: %s", output_dir)


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

_CONFIGS_DIR = str(Path(__file__).parent.parent.parent / "configs")


@hydra.main(config_path=_CONFIGS_DIR, config_name="rotational", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Hydra-managed entry point — override any field on the CLI.

    The config_path points to the top-level ``configs/`` directory so
    config names map directly to YAML files there::

        beamsim-run                                  # uses default: rotational
        beamsim-run --config-name alpha              # alpha sweep
        beamsim-run --config-name snr                # SNR sweep
        beamsim-run run.n_trials=2 run.n_steps=100   # quick smoke run

    Groups (scenario/, sweep/, algo/) are subdirectories of the same root,
    so absolute references ``/scenario``, ``/sweep``, ``/algo`` in defaults
    lists resolve correctly.
    """
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    run_from_config(cfg)


if __name__ == "__main__":
    main()
