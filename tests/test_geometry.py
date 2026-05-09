"""Geometry/mobility sanity tests."""

import numpy as np

from beamsim.geometry import (
    relative_aoa,
    relative_aod,
    rotation_track,
    straight_line_track,
)


def test_straight_line_track_distance():
    track = straight_line_track((0.0, 0.0), heading=0.0, speed_mps=1.0, n_steps=1001, dt=1e-3)
    travelled = np.linalg.norm(track.positions[-1] - track.positions[0])
    np.testing.assert_allclose(travelled, 1.0, atol=1e-9)
    assert track.n_steps == 1001


def test_rotation_track_orientation():
    track = rotation_track((10.0, 0.0), rpm=60.0, n_steps=1001, dt=1e-3)
    # 60 rpm = 1 Hz = 2*pi rad/s. After 1 second we should be back to start.
    np.testing.assert_allclose(
        np.cos(track.orientations[-1]), np.cos(track.orientations[0]), atol=1e-9
    )


def test_relative_aoa_aod_round_trip():
    ue_xy = np.array([0.0, 0.0])
    bs_xy = np.array([10.0, 10.0])
    ue_yaw = 0.0
    bs_yaw = 0.0
    aoa = relative_aoa(ue_xy, ue_yaw, bs_xy)
    aod = relative_aod(bs_xy, bs_yaw, ue_xy)
    # AoA at UE points to BS at 45 deg; AoD at BS points to UE at 45+180 deg = -135 deg.
    np.testing.assert_allclose(aoa, np.deg2rad(45.0), atol=1e-9)
    np.testing.assert_allclose(aod, np.deg2rad(-135.0), atol=1e-9)


def test_rotation_changes_relative_aoa():
    track = rotation_track((0.0, 0.0), rpm=60.0, n_steps=11, dt=0.1)
    bs = np.array([10.0, 0.0])
    aoa0 = relative_aoa(track.positions[0], track.orientations[0], bs)
    aoa5 = relative_aoa(track.positions[5], track.orientations[5], bs)
    assert not np.isclose(aoa0, aoa5)
