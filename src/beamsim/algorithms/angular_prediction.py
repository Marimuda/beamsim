"""Angular prediction: Kalman-filter AoA/AoD tracking in angle space.

Implements the angular tracking algorithm described in the predecessor MSc
thesis (Kristmundsson & Syberg, 2018), Section 5.4.3, Algorithm 3 (p. 63),
upgraded with a linear Kalman filter (constant-velocity dynamics) in place of
the naive index-space gradient predictor.

Report description (Sec. 5.4.3):
  Step 1: h(m) <- a(k_hat, l_hat)   [OBP index -> angle]
  Step 2: (k,l) <- cb(h(m) + 1/z * sum_i F(i) * (h(m-i+1) - h(m-i)))
  where a() is the angle of the codebook entry (arcsin of the cosine-spaced
  spatial frequency), and cb() is the nearest-codebook-entry quantiser.

Upgrade: replace the report's gradient-sum predictor with a 4-state linear
Kalman filter (LKF) tracking [theta_AoA, AoA_rate, theta_AoD, AoD_rate].
This is the constant-velocity model widely used in beam-angle tracking
(Va et al. 2016, ref [39] in the thesis) and is equivalent to the report's
Algorithm 3 with F chosen as the Kalman gain sequence.

State vector:  x = [theta_AoA, dot_theta_AoA, theta_AoD, dot_theta_AoD]
Observation:   z_m = [theta_AoA_codebook(k), theta_AoD_codebook(l)]
                  (noisy angle read from the OBP codebook beam centre)

Process model (constant velocity, dt = 1 occasion):
  F = [[1 dt 0  0 ]
       [0  1 0  0 ]
       [0  0 1 dt ]
       [0  0 0  1 ]]

Process noise:
  Q = diag(q_angle, q_rate, q_angle, q_rate)
  q_angle tuned so the filter tracks ~1 rad/s angular drift (about 1 beam
  width per 4-5 occasions at 8 UE beams); q_rate slightly larger to allow
  quick velocity adaptation.

Observation model:
  H_obs = [[1 0 0 0],
           [0 0 1 0]]
  R_obs = diag(sigma_AoA^2, sigma_AoD^2) — codebook quantisation noise
  (half-beamwidth standard deviation).

Cold start: fall back to exhaustive round-robin for the first `warmup`
occasions (default 4) so the Kalman filter has enough angle samples to
initialise its state estimate.

Reference:
  Va, V. et al. (2016). "Inverse multisine excitation for fast beam
  alignment in millimeter wave systems". [39] in the thesis.
  Kristmundsson & Syberg (2018), WCS10-951, Section 5.4.3, Algorithm 3,
  Figure 5.19 (pp. 63-64).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from beamsim.algorithms.base import Algorithm
from beamsim.bplm import BPLMState

# State dimension and observation dimension
_NX = 4
_NZ = 2


class AngularPrediction(Algorithm):
    """Kalman-filter AoA/AoD tracker with codebook-beam quantisation.

    Parameters
    ----------
    warmup : int
        Number of OBP samples required before switching from exhaustive
        cold-start to Kalman prediction.  Default 4 (report recommends 3-5).
    q_angle : float
        Process noise variance on the angle states (rad^2 per occasion).
        Tuned for ~1 rad/s tracking at 1 kHz sampling.  Default 1e-3.
    q_rate : float
        Process noise variance on the angular-rate states.  Default 1e-2.
    sigma_obs_factor : float
        Observation noise std dev as a fraction of the mean half-beamwidth.
        Default 1.0 (one half-beamwidth of quantisation uncertainty).
    """

    name = "angular_prediction"

    def __init__(self,
                 warmup: int = 4,
                 q_angle: float = 1e-3,
                 q_rate: float = 1e-2,
                 sigma_obs_factor: float = 1.0):
        self.warmup = warmup
        self.q_angle = q_angle
        self.q_rate = q_rate
        self.sigma_obs_factor = sigma_obs_factor

    # ------------------------------------------------------------------
    # Algorithm interface
    # ------------------------------------------------------------------

    def reset(self, state: BPLMState, context: dict) -> None:
        # Kalman state estimate and covariance
        self._x: NDArray[np.float64] = np.zeros(_NX)   # [aoa, aoa_dot, aod, aod_dot]
        self._P: NDArray[np.float64] = np.eye(_NX) * 1.0

        # Build system matrices once (dt = 1 occasion)
        dt = 1.0
        self._F_kf = np.array([[1, dt, 0, 0],
                                [0, 1,  0, 0],
                                [0, 0,  1, dt],
                                [0, 0,  0, 1]], dtype=float)
        self._H_kf = np.array([[1, 0, 0, 0],
                                [0, 0, 1, 0]], dtype=float)

        Q_diag = [self.q_angle, self.q_rate, self.q_angle, self.q_rate]
        self._Q = np.diag(Q_diag)

        # Observation noise: half-beamwidth of each codebook
        bw_ue = self._mean_half_beamwidth(state.ue_codebook.theta)
        bw_bs = self._mean_half_beamwidth(state.bs_codebook.theta)
        sigma_aoa = self.sigma_obs_factor * bw_ue
        sigma_aod = self.sigma_obs_factor * bw_bs
        self._R = np.diag([sigma_aoa ** 2, sigma_aod ** 2])

        self._obp_count = 0
        self._initialised = False
        self._sweep_idx = 0

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        # Update filter if we have a new OBP observation
        if np.any(state.measured_at >= 0):
            self._update_filter(state)

        # Cold start: exhaustive sweep until warmup samples collected
        if self._obp_count < self.warmup:
            return self._cold_start_step(state)

        # Predict one step ahead
        self._predict()
        pred_aoa = float(self._x[0])
        pred_aod = float(self._x[2])

        k = _nearest_beam(pred_aoa, state.ue_codebook.theta)
        l = _nearest_beam(pred_aod, state.bs_codebook.theta)
        return k, l

    # ------------------------------------------------------------------
    # Kalman filter internals
    # ------------------------------------------------------------------

    def _update_filter(self, state: BPLMState) -> None:
        """Incorporate the current OBP beam indices as an angle observation."""
        ok, ol = state.obp()
        z = np.array([
            float(state.ue_codebook.theta[ok]),
            float(state.bs_codebook.theta[ol])
        ])

        if not self._initialised:
            # Initialise state from first observation
            self._x = np.array([z[0], 0.0, z[1], 0.0])
            self._P = np.eye(_NX) * 1.0
            self._initialised = True
            self._obp_count += 1
            return

        # Standard Kalman update on the predicted state
        y = z - self._H_kf @ self._x                        # innovation
        S = self._H_kf @ self._P @ self._H_kf.T + self._R   # innovation covariance
        K_gain = self._P @ self._H_kf.T @ np.linalg.inv(S)  # Kalman gain
        self._x = self._x + K_gain @ y
        self._P = (np.eye(_NX) - K_gain @ self._H_kf) @ self._P
        self._obp_count += 1

    def _predict(self) -> None:
        """Advance the Kalman state one occasion forward."""
        self._x = self._F_kf @ self._x
        self._P = self._F_kf @ self._P @ self._F_kf.T + self._Q

    def _cold_start_step(self, state: BPLMState) -> tuple[int, int]:
        """Round-robin sweep during warmup phase."""
        total = state.K * state.L
        idx = self._sweep_idx % total
        self._sweep_idx += 1
        return idx // state.L, idx % state.L

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_half_beamwidth(theta: NDArray[np.float64]) -> float:
        """Mean half-beamwidth (radians) from the spacing of codebook angles."""
        if len(theta) < 2:
            return 0.1
        spacings = np.diff(np.sort(theta))
        return float(np.mean(spacings) / 2.0)


def _nearest_beam(angle: float, theta: NDArray[np.float64]) -> int:
    """Index of the codebook beam whose steering angle is closest to `angle`."""
    diffs = np.abs(_wrap_pi(theta - angle))
    return int(np.argmin(diffs))


def _wrap_pi(a: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
