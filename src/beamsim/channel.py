"""Simplified 3GPP TR 38.901-inspired cluster-delay-line channel.

Scope and simplifications (also stated in the paper's reproducibility section):

- Azimuth-only (no zenith); single polarisation; no random ray-coupling.
- 12 NLOS clusters with 20 sub-rays each (per TR 38.901 Table 7.5-6 LOS).
- Per-cluster intra-spread uses cluster ASA/ASD means without LSP cross-correlation.
- Path loss for UMi LOS / NLOS per TR 38.901 Section 7.4.1.
- K-factor for LOS scenarios drawn from a Gaussian (mean 9 dB, std 5 dB), clipped to [-3, 20] dB.
- Spatial consistency across the trajectory is approximated by anchoring cluster
  scatterer positions at the start of the trial; the UE-to-scatterer geometry
  evolves per occasion as the UE moves, so AoA at the UE updates naturally.
- Blockage is supported as a probabilistic LOS-blockage event with configurable
  rate; when blocked, the LOS K-factor is set to 0 for the remainder of the trial.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from beamsim.codebook import steering_vector


SPEED_OF_LIGHT = 2.998e8


@dataclass(frozen=True)
class ChannelParams:
    fc_hz: float = 28e9
    bandwidth_hz: float = 100e6
    h_bs: float = 10.0
    h_ut: float = 1.5
    n_clusters: int = 12
    n_rays_per_cluster: int = 20
    cluster_asa_deg: float = 17.0     # TR 38.901 Table 7.5-6 UMi LOS
    cluster_asd_deg: float = 3.0
    k_factor_mean_db: float = 9.0
    k_factor_std_db: float = 5.0
    cluster_shadow_std_db: float = 3.0
    los_probability: float = 1.0       # set <1.0 for stochastic LOS state
    blockage_rate_per_sec: float = 0.0  # 0 = no blockage events
    scenario: str = "umi"              # "umi" or "uma" (path-loss only)


def umi_path_loss_db(d_2d_m: float, fc_hz: float, h_bs: float, h_ut: float, los: bool) -> float:
    """3GPP TR 38.901 Section 7.4.1, UMi-Street-Canyon path loss in dB."""
    fc_ghz = fc_hz / 1e9
    h_e = 1.0  # effective height for breakpoint
    d_bp = 4 * (h_bs - h_e) * (h_ut - h_e) * fc_hz / SPEED_OF_LIGHT
    d_3d = np.sqrt(d_2d_m ** 2 + (h_bs - h_ut) ** 2)
    pl_los_close = 32.4 + 21 * np.log10(d_3d) + 20 * np.log10(fc_ghz)
    if d_2d_m <= d_bp:
        pl_los = pl_los_close
    else:
        pl_los = 32.4 + 40 * np.log10(d_3d) + 20 * np.log10(fc_ghz) \
                 - 9.5 * np.log10(d_bp ** 2 + (h_bs - h_ut) ** 2)
    if los:
        return pl_los
    pl_nlos = 35.3 * np.log10(d_3d) + 22.4 + 21.3 * np.log10(fc_ghz) - 0.3 * (h_ut - 1.5)
    return max(pl_los, pl_nlos)


@dataclass
class ChannelRealisation:
    """One Monte Carlo realisation: scatterer geometry fixed at trial start."""
    params: ChannelParams
    bs_xy: NDArray[np.float64]
    bs_yaw: float
    n_bs_elements: int
    n_ue_elements: int
    rng: np.random.Generator
    is_los: bool = True
    # Geometry-anchored scatterer positions (NLOS clusters)
    scatterer_xy: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 2)))
    cluster_powers: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    # Sub-ray offsets per cluster: (n_clusters, n_rays_per_cluster) at AoA/AoD level
    sub_ray_aoa_offsets: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    sub_ray_aod_offsets: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    sub_ray_phases: NDArray[np.complex128] = field(default_factory=lambda: np.zeros((0, 0), dtype=np.complex128))
    # LOS K-factor (linear)
    k_lin: float = 1.0
    los_blocked: bool = False

    def __post_init__(self):
        rng = self.rng
        p = self.params

        if self.params.los_probability < 1.0:
            self.is_los = rng.random() < self.params.los_probability

        # Sample LOS K-factor (linear)
        if self.is_los:
            k_db = np.clip(rng.normal(p.k_factor_mean_db, p.k_factor_std_db), -3.0, 20.0)
            self.k_lin = 10 ** (k_db / 10.0)
        else:
            self.k_lin = 0.0

        # Sample NLOS scatterer positions in a 200 m disc around the BS
        radius = 200.0
        n = p.n_clusters
        r = radius * np.sqrt(rng.random(n))
        theta = 2 * np.pi * rng.random(n)
        scatterers = self.bs_xy + np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        self.scatterer_xy = scatterers

        # Cluster powers: exponential decay with random shadowing.
        cluster_idx = np.arange(n)
        decay = np.exp(-cluster_idx / 6.0)
        shadow = 10 ** (rng.normal(0.0, p.cluster_shadow_std_db, size=n) / 10.0)
        powers = decay * shadow
        powers /= powers.sum()  # normalise to total NLOS power = 1
        self.cluster_powers = powers

        # Intra-cluster sub-ray angular offsets (small Laplacian-like spread)
        nr = p.n_rays_per_cluster
        # Use scaled-Laplace-like from TR 38.901 (approximation):
        std_aoa = np.deg2rad(p.cluster_asa_deg) / np.sqrt(2)
        std_aod = np.deg2rad(p.cluster_asd_deg) / np.sqrt(2)
        self.sub_ray_aoa_offsets = rng.laplace(0.0, std_aoa, size=(n, nr))
        self.sub_ray_aod_offsets = rng.laplace(0.0, std_aod, size=(n, nr))
        # Random phases per sub-ray
        self.sub_ray_phases = np.exp(1j * 2 * np.pi * rng.random((n, nr)))

    def channel_matrix(self,
                        ue_xy: NDArray[np.float64],
                        ue_yaw: float,
                        time_s: float = 0.0) -> NDArray[np.complex128]:
        """(n_ue_elements, n_bs_elements) downlink channel at the UE pose.

        Convention y = w_k^H H f_l x with H in C^{N_UE x N_BS}, so the channel
        matrix has UE rows and BS columns. Magnitude is amplitude (path loss
        is included as an amplitude scale), so |y_kl|^2 with unit-variance
        noise gives the receive SNR scaled by the runner's tx_amp.
        """
        # Simple time-driven blockage event: once blocked, stays blocked
        if self.is_los and not self.los_blocked and self.params.blockage_rate_per_sec > 0:
            if self.rng.random() < self.params.blockage_rate_per_sec * 1e-3:  # per-millisecond probability
                self.los_blocked = True

        # Distance and path loss
        d_2d = float(np.linalg.norm(np.asarray(ue_xy) - self.bs_xy))
        pl_db = umi_path_loss_db(d_2d, self.params.fc_hz, self.params.h_bs,
                                  self.params.h_ut, los=self.is_los)
        pl_lin = 10 ** (-pl_db / 20.0)  # amplitude scaling

        h = np.zeros((self.n_ue_elements, self.n_bs_elements), dtype=np.complex128)

        # LOS direct path
        if self.is_los and not self.los_blocked:
            aoa_world_los = np.arctan2(self.bs_xy[1] - ue_xy[1], self.bs_xy[0] - ue_xy[0])
            aod_world_los = np.arctan2(ue_xy[1] - self.bs_xy[1], ue_xy[0] - self.bs_xy[0])
            aoa_rel = _wrap_pi(aoa_world_los - ue_yaw)
            aod_rel = _wrap_pi(aod_world_los - self.bs_yaw)
            a_ue = steering_vector(self.n_ue_elements, aoa_rel)
            a_bs = steering_vector(self.n_bs_elements, aod_rel)
            los_amp = pl_lin * np.sqrt(self.k_lin / (1 + self.k_lin))
            h += los_amp * np.outer(a_ue, a_bs.conj())

        # NLOS cluster contributions
        nlos_total_amp = pl_lin / np.sqrt(1 + self.k_lin) if self.is_los else pl_lin
        for c in range(self.params.n_clusters):
            sc = self.scatterer_xy[c]
            aoa_world_c = np.arctan2(sc[1] - ue_xy[1], sc[0] - ue_xy[0])
            aod_world_c = np.arctan2(sc[1] - self.bs_xy[1], sc[0] - self.bs_xy[0])
            cluster_amp = nlos_total_amp * np.sqrt(self.cluster_powers[c])

            for r in range(self.params.n_rays_per_cluster):
                aoa_rel = _wrap_pi(aoa_world_c + self.sub_ray_aoa_offsets[c, r] - ue_yaw)
                aod_rel = _wrap_pi(aod_world_c + self.sub_ray_aod_offsets[c, r] - self.bs_yaw)
                a_ue = steering_vector(self.n_ue_elements, aoa_rel)
                a_bs = steering_vector(self.n_bs_elements, aod_rel)
                ray_amp = cluster_amp * self.sub_ray_phases[c, r] / np.sqrt(self.params.n_rays_per_cluster)
                h += ray_amp * np.outer(a_ue, a_bs.conj())
        return h

    def los_aoa_world(self, ue_xy: NDArray[np.float64]) -> float:
        return float(np.arctan2(self.bs_xy[1] - ue_xy[1], self.bs_xy[0] - ue_xy[0]))

    def los_aod_world(self, ue_xy: NDArray[np.float64]) -> float:
        return float(np.arctan2(ue_xy[1] - self.bs_xy[1], ue_xy[0] - self.bs_xy[0]))


def _wrap_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class FreeSpaceLosChannel:
    """Single-LOS-component channel for the noiseless rotation experiment.

    No clusters, no path loss (gain = 1), so the rotation test isolates the
    array-response geometry and codebook resolution from the channel.
    """
    bs_xy: NDArray[np.float64]
    bs_yaw: float
    n_bs_elements: int
    n_ue_elements: int

    def channel_matrix(self, ue_xy, ue_yaw, time_s: float = 0.0) -> NDArray[np.complex128]:
        aoa_world = np.arctan2(self.bs_xy[1] - ue_xy[1], self.bs_xy[0] - ue_xy[0])
        aod_world = np.arctan2(ue_xy[1] - self.bs_xy[1], ue_xy[0] - self.bs_xy[0])
        a_ue = steering_vector(self.n_ue_elements, _wrap_pi(aoa_world - ue_yaw))
        a_bs = steering_vector(self.n_bs_elements, _wrap_pi(aod_world - self.bs_yaw))
        return np.outer(a_ue, a_bs.conj())
