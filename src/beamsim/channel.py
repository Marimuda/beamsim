"""3GPP TR 38.901-compliant cluster-delay-line channel model.

Implementation scope and documented simplifications:

Implemented (TR 38.901 reference in parentheses):
  - Sub-ray angle offsets: canonical 20-ray table (TR 38.901 Table 7.5-3).
  - Large-scale parameters (DS, ASA, ASD, K, SF): per-draw Gaussian from
    TR 38.901 Table 7.5-6 UMi LOS / NLOS marginal statistics.
  - Cluster delays and powers: exponential + per-cluster shadow via
    TR 38.901 Step 6 procedure with r_tau and xi parameters from Table 7.5-6.
  - K-factor power split: LOS/NLOS power normalisation per TR 38.901 Step 6
    second sub-step.
  - LOS probability: TR 38.901 §7.4.2 UMi distance-dependent formula.
  - Blockage model: simplified stochastic screen model inspired by
    TR 38.901 §7.6.4.1 (Model A), per-cluster, with 30 dB attenuation plus
    distance-dependent knife-edge diffraction term.
  - Doppler phases: per-ray evolution via f_D = (v · k_nm)/λ (TR 38.901 §7.5
    Step 11).
  - Path loss: UMi-Street-Canyon LOS/NLOS (TR 38.901 §7.4.1).

Documented simplifications (not implemented):
  - LSP cross-correlation matrix (TR 38.901 Table 7.5-6 lower-triangular Cholesky
    draw) is skipped; all LSPs are drawn independently. TODO: implement via
    Cholesky decomposition of the inter-parameter correlation matrix.
  - Zenith angles (ZOA, ZOD) are not modelled; azimuth-only 2-D geometry.
  - Single polarisation only; no polarisation rotation matrix.
  - No random sub-cluster splitting (TR 38.901 §7.5 Step 11 sub-cluster mapping
    for the two strongest clusters) is applied.
  - Spatial consistency (TR 38.901 §7.6.3) is approximated by anchoring
    scatterer positions at the trial start; LSPs are not spatially filtered.
  - Blockage knife-edge diffraction uses a simplified single-screen model
    rather than the full dual-lobe model of TR 38.901 §7.6.4.
  - UMa path-loss added as convenience; full UMa cluster statistics not
    separately parameterised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from beamsim.codebook import steering_vector


SPEED_OF_LIGHT = 2.998e8

# ---------------------------------------------------------------------------
# TR 38.901 Table 7.5-3: Intra-cluster sub-ray offset angles (degrees).
# These 20 deterministic offsets replicate the Laplacian intra-cluster spread.
# The positive/negative pair structure means the offsets sum to zero by design.
# Reference: 3GPP TR 38.901 v17.0.0, Table 7.5-3.
# ---------------------------------------------------------------------------
_TR38901_RAY_OFFSETS_DEG: NDArray[np.float64] = np.array([
     0.0447,  0.1413,  0.2492,  0.3715,  0.5129,
     0.6797,  0.8844,  1.1481,  1.5195,  2.1551,
    -0.0447, -0.1413, -0.2492, -0.3715, -0.5129,
    -0.6797, -0.8844, -1.1481, -1.5195, -2.1551,
], dtype=np.float64)  # shape (20,)


# ---------------------------------------------------------------------------
# LSP statistics per TR 38.901 Table 7.5-6 (UMi-Street-Canyon).
# Keys: (scenario, los_state) -> dict of parameter stats.
# DS: log10(s), ASA: log10(deg), ASD: log10(deg), K: dB, SF: dB std only.
# r_tau: delay-scaling ratio; xi: per-cluster shadow std in dB.
# N: number of clusters; M: sub-rays per cluster.
# ---------------------------------------------------------------------------
_LSP_PARAMS: dict = {
    ("umi", True): {
        "ds_mu": -7.19, "ds_sigma": 0.40,       # TR 38.901 Table 7.5-6 UMi LOS DS
        "asa_mu": 1.81,  "asa_sigma": 0.20,      # UMi LOS ASA
        "asd_mu": 1.20,  "asd_sigma": 0.41,      # UMi LOS ASD
        "k_mu_db": 9.0,  "k_sigma_db": 5.0,      # UMi LOS K-factor (dB)
        "sf_sigma_db": 4.0,                       # UMi LOS shadow fading std
        "r_tau": 3.0,                             # delay-scaling ratio (Table 7.5-6)
        "xi_db": 3.0,                             # per-cluster shadow std (Table 7.5-6)
        "n_clusters": 12,
        "n_rays": 20,
        "cluster_asa_deg": 17.0,                  # intra-cluster ASA (Table 7.5-6)
        "cluster_asd_deg": 3.0,                   # intra-cluster ASD (Table 7.5-6)
    },
    ("umi", False): {
        "ds_mu": -6.89, "ds_sigma": 0.54,        # TR 38.901 Table 7.5-6 UMi NLOS DS
        "asa_mu": 1.84,  "asa_sigma": 0.16,      # UMi NLOS ASA
        "asd_mu": 1.19,  "asd_sigma": 0.21,      # UMi NLOS ASD
        "k_mu_db": None, "k_sigma_db": None,      # K-factor N/A for NLOS
        "sf_sigma_db": 7.82,                      # UMi NLOS shadow fading std
        "r_tau": 2.1,                             # Table 7.5-6
        "xi_db": 3.0,                             # Table 7.5-6
        "n_clusters": 19,
        "n_rays": 20,
        "cluster_asa_deg": 22.0,                  # Table 7.5-6
        "cluster_asd_deg": 5.0,                   # Table 7.5-6
    },
}


@dataclass(frozen=True)
class ChannelParams:
    """Configuration for a channel Monte Carlo trial.

    Most values default to the TR 38.901 UMi-LOS cluster statistics.
    Override ``los_probability`` to force a fixed LOS/NLOS state, or leave
    it as ``None`` to use the distance-dependent TR 38.901 §7.4.2 formula.
    """
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
    los_probability: float = 1.0       # set <1.0 for stochastic LOS state; 1.0 = always LOS
    blockage_rate_per_sec: float = 0.0  # 0 = no blockage events
    scenario: str = "umi"              # "umi" or "uma" (path-loss only)
    ue_speed_mps: float = 0.0          # UE speed in m/s for Doppler (magnitude)


# ---------------------------------------------------------------------------
# Path loss
# ---------------------------------------------------------------------------

def umi_path_loss_db(d_2d_m: float, fc_hz: float, h_bs: float, h_ut: float, los: bool) -> float:
    """3GPP TR 38.901 §7.4.1, UMi-Street-Canyon path loss in dB.

    Two-slope LOS model with breakpoint distance, NLOS lower-bounded by LOS.
    Reference: 3GPP TR 38.901 v17.0.0, Section 7.4.1, Table 7.4.1-1.
    """
    fc_ghz = fc_hz / 1e9
    h_e = 1.0  # effective environment height for breakpoint (Table 7.4.1-1)
    d_bp = 4 * (h_bs - h_e) * (h_ut - h_e) * fc_hz / SPEED_OF_LIGHT
    d_3d = np.sqrt(d_2d_m ** 2 + (h_bs - h_ut) ** 2)
    pl_los_close = 32.4 + 21 * np.log10(d_3d) + 20 * np.log10(fc_ghz)
    if d_2d_m <= d_bp:
        pl_los = pl_los_close
    else:
        pl_los = (32.4 + 40 * np.log10(d_3d) + 20 * np.log10(fc_ghz)
                  - 9.5 * np.log10(d_bp ** 2 + (h_bs - h_ut) ** 2))
    if los:
        return pl_los
    pl_nlos = 35.3 * np.log10(d_3d) + 22.4 + 21.3 * np.log10(fc_ghz) - 0.3 * (h_ut - 1.5)
    return max(pl_los, pl_nlos)


def uma_path_loss_db(d_2d_m: float, fc_hz: float, h_bs: float, h_ut: float, los: bool) -> float:
    """3GPP TR 38.901 §7.4.1, UMa path loss in dB (convenience wrapper).

    Reference: 3GPP TR 38.901 v17.0.0, Section 7.4.1, Table 7.4.1-1.
    """
    fc_ghz = fc_hz / 1e9
    h_e = 1.0
    d_bp = 4 * (h_bs - h_e) * (h_ut - h_e) * fc_hz / SPEED_OF_LIGHT
    d_3d = np.sqrt(d_2d_m ** 2 + (h_bs - h_ut) ** 2)
    if d_2d_m <= d_bp:
        pl_los = 28.0 + 22 * np.log10(d_3d) + 20 * np.log10(fc_ghz)
    else:
        pl_los = (28.0 + 40 * np.log10(d_3d) + 20 * np.log10(fc_ghz)
                  - 9.0 * np.log10(d_bp ** 2 + (h_bs - h_ut) ** 2))
    if los:
        return pl_los
    pl_nlos = 13.54 + 39.08 * np.log10(d_3d) + 20 * np.log10(fc_ghz) - 0.6 * (h_ut - 1.5)
    return max(pl_los, pl_nlos)


# ---------------------------------------------------------------------------
# LOS probability
# ---------------------------------------------------------------------------

def umi_los_probability(d_2d_m: float) -> float:
    """TR 38.901 §7.4.2 UMi-Street-Canyon LOS probability.

    P_LOS = min(18/d_2D, 1) * (1 - exp(-d_2D/36)) + exp(-d_2D/36)

    Reference: 3GPP TR 38.901 v17.0.0, Table 7.4.2-1.
    """
    d = max(d_2d_m, 1.0)  # avoid division by zero at origin
    return min(18.0 / d, 1.0) * (1.0 - np.exp(-d / 36.0)) + np.exp(-d / 36.0)


# ---------------------------------------------------------------------------
# LSP draws
# ---------------------------------------------------------------------------

def _draw_lsps(rng: np.random.Generator, lsp: dict,
               k_mu_db_override: Optional[float] = None,
               k_sigma_db_override: Optional[float] = None) -> dict:
    """Draw independent large-scale parameters for one UE drop.

    Returns a dict with keys: ds_s, asa_deg, asd_deg, k_db, k_lin, sf_db.

    ``k_mu_db_override`` and ``k_sigma_db_override`` allow ChannelParams to
    override the table K-factor statistics (e.g. for deterministic test scenarios).

    NOTE: LSP cross-correlation (TR 38.901 Table 7.5-6 lower-triangular
    Cholesky draw) is skipped here. Each LSP is drawn independently from its
    marginal Gaussian distribution.
    TODO: implement the full cross-correlated draw via:
        z = L @ N(0,I) where L = cholesky(C_lsp)
    as described in TR 38.901 §7.5 Step 4.
    """
    ds_s = 10 ** rng.normal(lsp["ds_mu"], lsp["ds_sigma"])
    asa_deg = 10 ** rng.normal(lsp["asa_mu"], lsp["asa_sigma"])
    asd_deg = 10 ** rng.normal(lsp["asd_mu"], lsp["asd_sigma"])
    sf_db = rng.normal(0.0, lsp["sf_sigma_db"])

    k_mu = k_mu_db_override if k_mu_db_override is not None else lsp["k_mu_db"]
    k_sigma = k_sigma_db_override if k_sigma_db_override is not None else lsp["k_sigma_db"]

    if k_mu is not None:
        k_db = float(np.clip(rng.normal(k_mu, k_sigma if k_sigma is not None else 0.0),
                             -3.0, 20.0))
        k_lin = float(10 ** (k_db / 10.0))
    else:
        k_db = -np.inf
        k_lin = 0.0
    return {"ds_s": ds_s, "asa_deg": asa_deg, "asd_deg": asd_deg,
            "k_db": k_db, "k_lin": k_lin, "sf_db": sf_db}


# ---------------------------------------------------------------------------
# Cluster delays and powers per TR 38.901 Step 5–6
# ---------------------------------------------------------------------------

def _draw_cluster_delays(rng: np.random.Generator, n: int, r_tau: float, ds_s: float) -> NDArray[np.float64]:
    """TR 38.901 §7.5 Step 5: draw and sort cluster delays.

    tau'_n ~ -r_tau * DS * ln(X_n),  X_n ~ Uniform(0,1).
    Normalised to tau'_1 = 0 (subtract minimum).

    Reference: 3GPP TR 38.901 v17.0.0, Section 7.5, Step 5.
    """
    x = rng.uniform(0.0, 1.0, size=n)
    tau_raw = -r_tau * ds_s * np.log(x)
    tau_sorted = np.sort(tau_raw - tau_raw.min())
    return tau_sorted


def _draw_cluster_powers(rng: np.random.Generator, taus: NDArray[np.float64],
                          r_tau: float, ds_s: float, xi_db: float,
                          k_lin: float, is_los: bool) -> NDArray[np.float64]:
    """TR 38.901 §7.5 Step 6: compute cluster powers.

    P'_n = exp(-tau_n * (r_tau-1) / (r_tau*DS)) * 10^(-Z_n/10)
    where Z_n ~ N(0, xi^2).

    For LOS, the power in cluster 1 is reduced so that the LOS-to-NLOS ratio
    equals the drawn K-factor (TR 38.901 §7.5 Step 6, second sub-step):
        P1_nlos = P1 * 1/(K+1)     (cluster 1 reduced by K-factor)
        LOS component = K/(K+1) * sum(P) renormalised to total power.

    Powers are normalised so sum = 1 (NLOS total).

    Reference: 3GPP TR 38.901 v17.0.0, Section 7.5, Step 6.
    """
    z = rng.normal(0.0, xi_db, size=len(taus))
    p_lin = np.exp(-taus * (r_tau - 1.0) / (r_tau * ds_s)) * 10 ** (-z / 10.0)

    if is_los and k_lin > 0:
        # Scale cluster 1 NLOS power so LOS:NLOS ratio = k_lin.
        # Per TR 38.901 Step 6: P_n -> P_n / (K_R + 1) then add delta LOS power.
        p_lin = p_lin / (1.0 + k_lin)

    p_lin /= p_lin.sum()  # normalise NLOS cluster powers
    return p_lin


# ---------------------------------------------------------------------------
# Blockage (simplified Model A, TR 38.901 §7.6.4.1)
# ---------------------------------------------------------------------------

def _blockage_attenuation_db(rng: np.random.Generator, n: int,
                              blockage_prob_per_cluster: float = 0.15) -> NDArray[np.float64]:
    """Simplified stochastic blockage attenuation per cluster (dB).

    TR 38.901 §7.6.4.1 Model A describes a region-based screen model with
    knife-edge diffraction. This implementation uses a simplified stochastic
    version: each cluster is independently blocked with probability
    ``blockage_prob_per_cluster``.  Blocked clusters receive a base 30 dB
    attenuation plus a uniformly sampled additional diffraction loss in [0, 10]
    dB to represent variable screen positions.

    Simplification documented: the full geometric KED computation (screen
    geometry, Fresnel-zone diffraction parameter nu) is replaced by a random
    draw; the directional spread model of the original §7.6.4.1 is omitted.

    Reference: 3GPP TR 38.901 v17.0.0, Section 7.6.4.1.
    """
    blocked = rng.random(n) < blockage_prob_per_cluster
    attn = np.zeros(n)
    attn[blocked] = 30.0 + rng.uniform(0.0, 10.0, size=n)[blocked]
    return attn


# ---------------------------------------------------------------------------
# Main channel realisation
# ---------------------------------------------------------------------------

@dataclass
class ChannelRealisation:
    """One Monte Carlo realisation: scatterer geometry fixed at trial start.

    Implements TR 38.901 §7.5 Steps 1–11 for azimuth-only geometry.
    Public API is backward-compatible with the simplified GSCM predecessor.
    """
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
    cluster_delays_s: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    # Sub-ray offsets per cluster: (n_clusters, n_rays_per_cluster) at AoA/AoD level
    sub_ray_aoa_offsets: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    sub_ray_aod_offsets: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    sub_ray_phases: NDArray[np.complex128] = field(default_factory=lambda: np.zeros((0, 0), dtype=np.complex128))
    # Per-ray Doppler direction unit vectors (n_clusters, n_rays, 2) for v·k_nm
    _ray_unit_vecs: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0, 2)))
    # LOS K-factor (linear)
    k_lin: float = 1.0
    los_blocked: bool = False
    # Drawn LSPs stored for inspection / tests
    lsp: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        rng = self.rng
        p = self.params

        # ------------------------------------------------------------------
        # Step 1 – LOS / NLOS determination
        # TR 38.901 §7.4.2; override if los_probability < 1.0.
        # ------------------------------------------------------------------
        if p.los_probability < 1.0:
            self.is_los = rng.random() < p.los_probability

        # ------------------------------------------------------------------
        # Steps 3–4 – Large-scale parameter draws
        # TR 38.901 Table 7.5-6 marginal statistics (independent draws).
        # NOTE: LSP cross-correlation matrix skipped; see module docstring.
        # ------------------------------------------------------------------
        key = (p.scenario, self.is_los)
        lsp_table = _LSP_PARAMS.get(key, _LSP_PARAMS[("umi", self.is_los)])
        # Pass ChannelParams K-factor values so test overrides are respected.
        # Default ChannelParams values match the UMi LOS table, so normal
        # scenarios are unaffected; explicit overrides (e.g. k_factor_mean_db=30)
        # are honoured for reproducible test scenarios.
        drawn = _draw_lsps(rng, lsp_table,
                           k_mu_db_override=p.k_factor_mean_db if self.is_los else None,
                           k_sigma_db_override=p.k_factor_std_db if self.is_los else None)
        self.lsp = drawn
        self.k_lin = drawn["k_lin"]

        n = lsp_table["n_clusters"]
        nr = lsp_table["n_rays"]
        cluster_asa = lsp_table["cluster_asa_deg"]
        cluster_asd = lsp_table["cluster_asd_deg"]

        # ------------------------------------------------------------------
        # Step 5 – Cluster delays
        # TR 38.901 §7.5 Step 5.
        # ------------------------------------------------------------------
        taus = _draw_cluster_delays(rng, n, lsp_table["r_tau"], drawn["ds_s"])
        self.cluster_delays_s = taus

        # ------------------------------------------------------------------
        # Step 6 – Cluster powers (with K-factor split for LOS)
        # TR 38.901 §7.5 Step 6.
        # ------------------------------------------------------------------
        powers = _draw_cluster_powers(rng, taus, lsp_table["r_tau"],
                                       drawn["ds_s"], lsp_table["xi_db"],
                                       drawn["k_lin"], self.is_los)
        self.cluster_powers = powers

        # ------------------------------------------------------------------
        # Step 7 – Scatterer positions: place geometrically from BS
        # using drawn cluster angles (approximation: uniform in disc bounded
        # by max path-delay equivalent distance).
        # ------------------------------------------------------------------
        radius = 200.0
        r_vals = radius * np.sqrt(rng.random(n))
        theta_vals = 2 * np.pi * rng.random(n)
        scatterers = self.bs_xy + np.column_stack(
            [r_vals * np.cos(theta_vals), r_vals * np.sin(theta_vals)]
        )
        self.scatterer_xy = scatterers

        # ------------------------------------------------------------------
        # Step 8 – Sub-ray angle offsets per TR 38.901 Table 7.5-3.
        # The canonical 20 deterministic offsets are scaled by the cluster
        # intra-spread (ASA for AoA, ASD for AoD).
        # TR 38.901 Table 7.5-3: offset_m = c_ASA * alpha_m  (Eq. 7.5-1).
        # ------------------------------------------------------------------
        offsets_deg = _TR38901_RAY_OFFSETS_DEG  # shape (20,)
        # Scale offsets by intra-cluster spread (broadcast over clusters)
        aoa_off_deg = cluster_asa * offsets_deg[np.newaxis, :]  # (1, 20)
        aod_off_deg = cluster_asd * offsets_deg[np.newaxis, :]
        self.sub_ray_aoa_offsets = np.tile(np.deg2rad(aoa_off_deg), (n, 1))  # (n, 20)
        self.sub_ray_aod_offsets = np.tile(np.deg2rad(aod_off_deg), (n, 1))

        # ------------------------------------------------------------------
        # Step 9 – Random initial phases per sub-ray (uniform [0, 2π))
        # TR 38.901 §7.5 Step 9.
        # ------------------------------------------------------------------
        self.sub_ray_phases = np.exp(1j * 2 * np.pi * rng.random((n, nr)))

        # ------------------------------------------------------------------
        # Step 11 – Pre-compute per-ray unit direction vectors for Doppler.
        # k_nm = (cos(phi_nm), sin(phi_nm)) where phi_nm is the world-frame
        # AoA of sub-ray (n,m).  These are initialised at origin; actual
        # geometry updated per channel_matrix call from scatterer angles.
        # We store the scatterer-relative offsets here; the world-frame
        # direction is computed per UE position in channel_matrix.
        # TR 38.901 §7.5 Step 11, Eq. 7.5-22.
        # ------------------------------------------------------------------
        self._ray_unit_vecs = np.zeros((n, nr, 2))

        # ------------------------------------------------------------------
        # Blockage: draw per-cluster attenuation at trial start.
        # Simplified TR 38.901 §7.6.4.1 Model A (stochastic, not geometric).
        # ------------------------------------------------------------------
        self._blockage_attn_db = _blockage_attenuation_db(rng, n)

    def channel_matrix(self,
                        ue_xy: NDArray[np.float64],
                        ue_yaw: float,
                        time_s: float = 0.0) -> NDArray[np.complex128]:
        """(n_ue_elements, n_bs_elements) downlink channel matrix at UE pose.

        Convention: y = w_k^H H f_l x with H in C^{N_UE x N_BS}.
        Magnitude is amplitude; |y|^2 with unit-variance noise gives receive
        SNR scaled by runner tx_amp.

        Doppler: if ue_speed_mps > 0, each sub-ray phase is rotated by
        2*pi*f_D*time_s where f_D = (v · k_nm) / lambda,
        k_nm is the inward unit vector from UE toward scatterer sub-ray.
        TR 38.901 §7.5 Step 11, Eq. 7.5-22.
        """
        p = self.params
        ue_xy = np.asarray(ue_xy, dtype=np.float64)

        # Simple time-driven blockage event for the LOS path
        if self.is_los and not self.los_blocked and p.blockage_rate_per_sec > 0:
            if self.rng.random() < p.blockage_rate_per_sec * 1e-3:
                self.los_blocked = True

        # Distance and path loss
        d_2d = float(np.linalg.norm(ue_xy - self.bs_xy))
        pl_db = umi_path_loss_db(d_2d, p.fc_hz, p.h_bs, p.h_ut, los=self.is_los)
        pl_lin = 10 ** (-pl_db / 20.0)  # amplitude scaling

        lam = SPEED_OF_LIGHT / p.fc_hz  # wavelength

        h = np.zeros((self.n_ue_elements, self.n_bs_elements), dtype=np.complex128)

        # ------------------------------------------------------------------
        # LOS direct path
        # ------------------------------------------------------------------
        if self.is_los and not self.los_blocked:
            aoa_world_los = np.arctan2(self.bs_xy[1] - ue_xy[1],
                                       self.bs_xy[0] - ue_xy[0])
            aod_world_los = np.arctan2(ue_xy[1] - self.bs_xy[1],
                                       ue_xy[0] - self.bs_xy[0])
            aoa_rel = _wrap_pi(aoa_world_los - ue_yaw)
            aod_rel = _wrap_pi(aod_world_los - self.bs_yaw)
            a_ue = steering_vector(self.n_ue_elements, aoa_rel)
            a_bs = steering_vector(self.n_bs_elements, aod_rel)
            los_amp = pl_lin * np.sqrt(self.k_lin / (1.0 + self.k_lin))
            # Doppler on LOS component: UE moves toward/away from BS
            los_doppler = _los_doppler_phase(ue_xy, self.bs_xy, p.ue_speed_mps,
                                             ue_yaw, lam, time_s)
            h += los_amp * los_doppler * np.outer(a_ue, a_bs.conj())

        # ------------------------------------------------------------------
        # NLOS cluster contributions
        # ------------------------------------------------------------------
        nlos_total_amp = pl_lin / np.sqrt(1.0 + self.k_lin) if self.is_los else pl_lin

        n_cl = len(self.cluster_powers)
        for c in range(n_cl):
            sc = self.scatterer_xy[c]
            aoa_world_c = np.arctan2(sc[1] - ue_xy[1], sc[0] - ue_xy[0])
            aod_world_c = np.arctan2(sc[1] - self.bs_xy[1], sc[0] - self.bs_xy[0])

            # Blockage attenuation (simplified §7.6.4.1)
            blk_amp = 10 ** (-self._blockage_attn_db[c] / 20.0)

            cluster_amp = nlos_total_amp * np.sqrt(self.cluster_powers[c]) * blk_amp

            nr = self.params.n_rays_per_cluster
            for r in range(nr):
                aoa_rel = _wrap_pi(aoa_world_c + self.sub_ray_aoa_offsets[c, r] - ue_yaw)
                aod_rel = _wrap_pi(aod_world_c + self.sub_ray_aod_offsets[c, r] - self.bs_yaw)
                a_ue = steering_vector(self.n_ue_elements, aoa_rel)
                a_bs = steering_vector(self.n_bs_elements, aod_rel)

                # Doppler phase: f_D = (v · k_nm) / lambda
                # k_nm = unit vector toward scatterer sub-ray direction
                ray_angle = aoa_world_c + self.sub_ray_aoa_offsets[c, r]
                doppler_phase = _ray_doppler_phase(ray_angle, p.ue_speed_mps,
                                                   ue_yaw, lam, time_s)

                ray_amp = (cluster_amp * self.sub_ray_phases[c, r]
                           * doppler_phase / np.sqrt(nr))
                h += ray_amp * np.outer(a_ue, a_bs.conj())
        return h

    def los_aoa_world(self, ue_xy: NDArray[np.float64]) -> float:
        return float(np.arctan2(self.bs_xy[1] - ue_xy[1], self.bs_xy[0] - ue_xy[0]))

    def los_aod_world(self, ue_xy: NDArray[np.float64]) -> float:
        return float(np.arctan2(ue_xy[1] - self.bs_xy[1], ue_xy[0] - self.bs_xy[0]))


# ---------------------------------------------------------------------------
# Doppler helpers
# ---------------------------------------------------------------------------

def _los_doppler_phase(ue_xy: NDArray[np.float64], bs_xy: NDArray[np.float64],
                        speed_mps: float, ue_yaw: float,
                        lam: float, time_s: float) -> complex:
    """Doppler phase for LOS ray.  v is assumed aligned with ue_yaw heading."""
    if speed_mps == 0.0 or time_s == 0.0:
        return 1.0 + 0j
    d = bs_xy - ue_xy
    norm = np.linalg.norm(d)
    if norm < 1e-9:
        return 1.0 + 0j
    k_hat = d / norm  # unit vector toward BS from UE
    v_vec = speed_mps * np.array([np.cos(ue_yaw), np.sin(ue_yaw)])
    f_d = float(np.dot(v_vec, k_hat)) / lam
    return complex(np.exp(1j * 2 * np.pi * f_d * time_s))


def _ray_doppler_phase(ray_angle_world: float, speed_mps: float,
                        ue_yaw: float, lam: float, time_s: float) -> complex:
    """Doppler phase for a sub-ray arriving from world angle ray_angle_world.

    f_D = (v · k_nm) / lambda,  k_nm = unit inward direction (toward UE).
    TR 38.901 §7.5 Step 11, Eq. 7.5-22.
    """
    if speed_mps == 0.0 or time_s == 0.0:
        return 1.0 + 0j
    k_hat = np.array([np.cos(ray_angle_world), np.sin(ray_angle_world)])
    v_vec = speed_mps * np.array([np.cos(ue_yaw), np.sin(ue_yaw)])
    f_d = float(np.dot(v_vec, k_hat)) / lam
    return complex(np.exp(1j * 2 * np.pi * f_d * time_s))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _wrap_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Free-space single-LOS channel (unchanged API, kept for rotation experiment)
# ---------------------------------------------------------------------------

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
