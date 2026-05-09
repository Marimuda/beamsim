"""Structured config dataclasses for OmegaConf / Hydra.

All public fields are typed so OmegaConf can validate them at load time.
Use ``OmegaConf.structured(ExperimentConfig)`` or rely on Hydra's
``@hydra.main`` decorator to instantiate the composed config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScenarioConfig:
    """Geometry and channel parameters for one experiment scenario."""

    case: str = "case_c"
    # UE mobility
    ue_speed_mps: float = 0.0
    ue_track_kind: str = "rotation"  # "rotation" | "straight_line"
    # BS layout — list of [x, y] pairs
    bs_positions: list[list[float]] = field(default_factory=lambda: [[10.0, 0.0]])
    bs_yaws: list[float] = field(default_factory=lambda: [0.0])
    # Timing
    dt: float = 1e-3
    n_steps: int = 10_000
    # Channel
    channel_kind: str = "freespace_los"  # "freespace_los" | "geometric_umi"
    channel_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class SweepConfig:
    """Description of the independent variable swept in this experiment."""

    variable: str = "none"  # "rpm" | "alpha" | "snr_db" | "none"
    sweep_values: list[float] = field(default_factory=list)
    target_snr_db: float | None = None  # used for tx_amp calibration


@dataclass
class RunConfig:
    """Monte Carlo execution parameters."""

    algorithms: list[str] = field(
        default_factory=lambda: [
            "exhaustive",
            "nns",
            "tabu",
            "angular_prediction",
            "ci",
            "mcmd",
            "perfect",
        ]
    )
    n_trials: int = 10
    n_steps: int = 10_000  # overrides scenario.n_steps when set
    seed: int = 12345
    n_workers: int | None = None
    output_path: str = "results"


@dataclass
class ExperimentConfig:
    """Top-level composed config; built by Hydra from the defaults list."""

    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    run: RunConfig = field(default_factory=RunConfig)
    name: str = "unnamed"
