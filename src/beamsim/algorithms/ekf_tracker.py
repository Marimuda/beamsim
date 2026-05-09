"""Extended Kalman Filter beam-angle tracker.

Reference:
    Jayaprakasam, S., Ma, X., Choi, J. W., Kim, S. (2017). "Robust Beam-
    Tracking for mmWave Mobile Communications." IEEE Communications
    Letters 21(12), 2654-2657.  DOI: 10.1109/LCOMM.2017.2748140
    Burghal, D., Abbasi, A. A., Molisch, A. F. (2019). "Extended Kalman
    Filter Beam Tracking for Millimeter Wave Communications." arXiv:
    1911.01638 (also IEEE GlobalSIP 2019).

State and motion model
----------------------
The EKF tracks a 4-D state ``x = [theta_AoA, theta_AoD, omega_AoA,
omega_AoD]`` with a constant-angular-velocity motion model:

    theta_{t+1} = theta_t + omega_t * dt + w_theta
    omega_{t+1} = omega_t + w_omega

Both AoA (UE-side) and AoD (BS-side) are modelled jointly because for a
LoS path on a moving UE both angles drift in time (the AoD drift is
typically slow and small; the AoA drift is dominated by UE rotation).

Observation model
-----------------
At each step we measure one beam-pair and update the BPLM observations
matrix.  The observation we feed into the EKF is the *currently observed
best beam pair* (state.obp()) converted back to angles via the codebook
``theta`` arrays:

    z = [theta_UE[k_obp], theta_BS[l_obp]]

This is a noisy, codebook-quantised estimate of the true angles; we model
the observation noise as Gaussian with std-dev equal to half a beam
spacing.  H (the linearised observation Jacobian) is therefore
``[[1,0,0,0],[0,1,0,0]]``.

Beam selection
--------------
After the EKF predict/update cycle, the next beam pair is the codebook
index closest to the predicted next-step angles:

    k = argmin_k |theta_UE[k] - (theta_AoA + omega_AoA * dt)|
    l = argmin_l |theta_BS[l] - (theta_AoD + omega_AoD * dt)|

A pure greedy lookahead — no exploration, no fallback search.  This is
appropriate for a low-noise tracker: under constant-rate motion a single
measurement per step is enough to maintain alignment; under high noise or
abrupt motion changes the EKF is expected to fail (this is the classic
limitation of model-based trackers and is documented in the references).

Cold-start
----------
The first few steps run an Exhaustive-style full sweep so the BPLM has
populated entries the EKF can read.  We require a minimum of
``warmup`` measurements before switching to EKF prediction.  During
warmup the algorithm cycles through (k, l) pairs in row-major order
identical to ``Exhaustive``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState


class EKFTracker(Algorithm):
    """Constant-angular-velocity EKF tracker on AoA/AoD.

    Parameters
    ----------
    warmup:
        Number of cold-start sweep steps before EKF takes over.
    dt:
        Step interval (seconds).  Must match the runner's ``dt`` for
        the angular-rate prediction to be calibrated.
    sigma_proc_angle:
        Process noise std-dev on the angle component (rad/sqrt(s)).
    sigma_proc_rate:
        Process noise std-dev on the rate component (rad/s/sqrt(s)).
    obs_noise_floor:
        Observation noise std-dev expressed as a fraction of the
        codebook beam spacing.  ``0.5`` means we trust the codebook
        argmax to ±half a beam.
    """

    name = "ekf_tracker"

    def __init__(
        self,
        warmup: int = 12,
        dt: float = 1e-3,
        sigma_proc_angle: float = 1e-3,
        sigma_proc_rate: float = 5e-2,
        obs_noise_floor: float = 0.5,
    ) -> None:
        self._warmup = max(int(warmup), 2)
        self._dt = float(dt)
        self._sigma_proc_angle = float(sigma_proc_angle)
        self._sigma_proc_rate = float(sigma_proc_rate)
        self._obs_noise_floor = float(obs_noise_floor)

    def reset(self, state: BPLMState, context: dict) -> None:
        # State: [theta_AoA, theta_AoD, omega_AoA, omega_AoD].
        self._x: NDArray[np.float64] = np.zeros(4, dtype=np.float64)
        # Initial covariance: large angle uncertainty (we have no prior),
        # moderate rate uncertainty.
        self._P: NDArray[np.float64] = np.diag([1.0, 1.0, 0.5, 0.5])
        # Process-noise covariance (constant-rate model).
        sa = self._sigma_proc_angle
        sr = self._sigma_proc_rate
        dt = self._dt
        self._Q = np.diag([sa**2 * dt, sa**2 * dt, sr**2 * dt, sr**2 * dt])

        # Observation matrix H: angle observation only.
        self._H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

        # Observation noise: scale codebook spacing by floor factor.
        ue_theta = state.ue_codebook.theta
        bs_theta = state.bs_codebook.theta
        ue_spacing = float(np.median(np.diff(ue_theta)))
        bs_spacing = float(np.median(np.diff(bs_theta)))
        sigma_obs_ue = abs(ue_spacing) * self._obs_noise_floor
        sigma_obs_bs = abs(bs_spacing) * self._obs_noise_floor
        self._R = np.diag([sigma_obs_ue**2, sigma_obs_bs**2])

        self._cold_index: int = 0
        self._n_meas: int = 0
        self._initialised: bool = False

    def _f_predict(self) -> None:
        """Predict step: x <- F x; P <- F P F^T + Q."""
        dt = self._dt
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self._Q

    def _h_update(self, z: NDArray[np.float64]) -> None:
        """Update step: y = z - H x; S = H P H^T + R; K = P H^T S^-1."""
        H = self._H
        innov = z - H @ self._x
        S = H @ self._P @ H.T + self._R
        K = self._P @ H.T @ np.linalg.inv(S)
        self._x = self._x + K @ innov
        self._P = (np.eye(4) - K @ H) @ self._P

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        K, L = state.K, state.L
        ue_theta = state.ue_codebook.theta
        bs_theta = state.bs_codebook.theta

        # Cold-start: Latin-square-like diagonal sweep so each UE row and
        # several BS columns are sampled within ``warmup`` steps.  This
        # matters because a row-major sweep at warmup << K*L stays in UE
        # row 0 only, biasing the EKF initialisation; the diagonal pattern
        # spreads probes across the (k, l) grid.
        if self._cold_index < self._warmup:
            k = self._cold_index % K
            stride = max(L // max(K, 1), 1)
            offset = self._cold_index // K
            l = (self._cold_index * stride + offset) % L
            self._cold_index += 1
            self._n_meas += 1
            return k, l

        # Initialise EKF state from first OBP if we haven't yet.
        if not self._initialised and np.any(state.measured_at >= 0):
            k_obp, l_obp = state.obp()
            self._x[0] = float(ue_theta[k_obp])
            self._x[1] = float(bs_theta[l_obp])
            # Rates initialised to zero; the Q covariance lets them drift.
            self._x[2] = 0.0
            self._x[3] = 0.0
            self._initialised = True

        if not self._initialised:
            # Still no measurements — fall back to (0, 0) deterministically.
            self._n_meas += 1
            return 0, 0

        # Predict next angles.
        self._f_predict()

        # Update EKF with current OBP (a noisy angle measurement).
        k_obp, l_obp = state.obp()
        z = np.array([float(ue_theta[k_obp]), float(bs_theta[l_obp])])
        self._h_update(z)

        # Choose the next beam pair = codebook index closest to predicted
        # angle one step ahead.  We have already predicted *into* this step,
        # so x[0:2] is the current best estimate; for a one-step-ahead probe
        # we project forward by another dt's worth of rate.
        ahead_ue = self._x[0] + self._x[2] * self._dt
        ahead_bs = self._x[1] + self._x[3] * self._dt
        k = int(np.argmin(np.abs(ue_theta - ahead_ue)))
        l = int(np.argmin(np.abs(bs_theta - ahead_bs)))
        self._n_meas += 1
        return k, l
