"""UE/BS positions and mobility tracks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Track:
    """Sampled UE pose over a sequence of measurement occasions.

    Positions are 2-D (x, y) in metres, orientations in radians (yaw, with
    0 pointing along +x). Each array has length ``n_steps`` matching the
    Monte Carlo trial duration. The simulator is azimuth-only, so a 2-D
    track is sufficient.
    """

    positions: NDArray[np.float64]   # (n_steps, 2)
    orientations: NDArray[np.float64]  # (n_steps,)
    dt: float                           # seconds per measurement occasion

    @property
    def n_steps(self) -> int:
        return self.positions.shape[0]


def straight_line_track(start_xy: tuple[float, float],
                         heading: float,
                         speed_mps: float,
                         n_steps: int,
                         dt: float,
                         orientation: float | None = None) -> Track:
    """Constant-velocity straight-line track at ``speed_mps`` m/s.

    ``heading`` is the direction of motion in radians. If ``orientation``
    is None, the UE faces along the heading.
    """
    t = np.arange(n_steps) * dt
    direction = np.array([np.cos(heading), np.sin(heading)])
    positions = np.array(start_xy) + speed_mps * t[:, None] * direction
    yaw = np.full(n_steps, heading if orientation is None else orientation)
    return Track(positions=positions, orientations=yaw, dt=dt)


def rotation_track(position_xy: tuple[float, float],
                    rpm: float,
                    n_steps: int,
                    dt: float,
                    initial_orientation: float = 0.0) -> Track:
    """Stationary UE rotating about its own axis at ``rpm`` revolutions/min."""
    t = np.arange(n_steps) * dt
    omega = rpm * 2 * np.pi / 60.0  # rad/s
    yaw = initial_orientation + omega * t
    positions = np.tile(np.array(position_xy), (n_steps, 1))
    return Track(positions=positions, orientations=yaw, dt=dt)


def relative_aoa(ue_xy: NDArray[np.float64],
                  ue_yaw: float,
                  source_xy: NDArray[np.float64]) -> float:
    """AoA at the UE for a source at ``source_xy``, expressed in the UE body frame."""
    delta = np.asarray(source_xy) - np.asarray(ue_xy)
    world_aoa = np.arctan2(delta[1], delta[0])
    return _wrap_pi(world_aoa - ue_yaw)


def relative_aod(bs_xy: NDArray[np.float64],
                  bs_yaw: float,
                  ue_xy: NDArray[np.float64]) -> float:
    """AoD at the BS toward the UE, in the BS body frame."""
    delta = np.asarray(ue_xy) - np.asarray(bs_xy)
    world = np.arctan2(delta[1], delta[0])
    return _wrap_pi(world - bs_yaw)


def _wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi
