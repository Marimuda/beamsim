"""Geometry-based stochastic cluster-delay-line channel model.

This module implements the mmW propagation model described in the predecessor
MSc report (Sec 3.2, Table 3.1, Eqs 3.13–3.18). It diverges intentionally
from a strict TR 38.901 implementation in several ways described below.

Model summary (matching Sec 3.2 of the predecessor report):
------------------------------------------------------------
* Large-scale parameters (DS, ASA, ASD, K-factor, SF) are always drawn from
  the UMi LOS table (Table 3.1 / TR 38.901 Table 7.5-6 LOS row), regardless
  of whether the trial is LOS or NLOS. The LOS/NLOS distinction is then
  enforced solely through (a) whether the direct ray contributes and (b) the
  blockage model. (Sec 3.2.2: "in this simplified version we generate the
  channel using the LOS parameters and later apply the blockage model".)

* Scatterer positions are drawn uniformly in a disc around the BS. Cluster
  powers are computed geometrically: extra path-length = d(BS,S)+d(S,UE) -
  d(BS,UE) gives additional path loss relative to the LOS distance, so closer
  scatterers (shorter extra path) contribute more power. (Sec 3.2.2: "Cluster
  component powers are calculated by finding effective extra path-length from
  the excess delay and calculating additional path-loss".)

* Sub-ray AoA offsets within each cluster are drawn from a Laplacian
  distribution with scale S_alpha = cluster_asa / sqrt(2), and AoD offsets
  with S_alpha = cluster_asd / sqrt(2). (Eq 3.14, Sec 3.2.3.)
  The first sub-ray of each cluster keeps the cluster mean AoA (large-scale
  component); remaining L-1 rays are Laplacian draws centred on the cluster.

* Random initial phases per sub-ray are NOT included. (Sec 3.2: "Not included
  parts: ... Random initial phases".) Sub-ray phase arises only from steering
  vectors and Doppler.

* Blockage Model A (Sec 3.2.4, Eqs 3.15–3.18): one self-blocker (120–160 deg
  wide, 30 dB attenuation, centred at the back of the UE in body frame) and
  four non-self blockers (width U[5,15] deg, KED-based attenuation per
  Eqs 3.16–3.17, centre angles drifting per the autocorrelation model of
  Eq 3.18). Evaluated per UE pose at each channel_matrix() call.

Preserved from TR 38.901:
  - Path loss: umi_path_loss_db(), uma_path_loss_db() (TR 38.901 §7.4.1).
  - LOS probability: umi_los_probability() (TR 38.901 §7.4.2).
  - Doppler: f_D = (v · k_nm) / lambda per sub-ray.
  - Cluster delay draw: exponential with r_tau and DS from LOS table.
  - K-factor power split (LOS vs NLOS cluster power).

Documented simplifications:
  - LSP cross-correlation matrix skipped; all LSPs drawn independently.
  - Zenith angles not modelled (azimuth-only, 2-D geometry).
  - Single polarisation; no polarisation rotation matrix.
  - No random coupling of rays (TR 38.901 §7.5 Step 9 coupling).
  - Spatial consistency approximated by anchoring scatterer positions at
    trial start; LSPs are not spatially filtered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from beamsim.codebook import steering_vector

SPEED_OF_LIGHT = 2.998e8

# ---------------------------------------------------------------------------
# TR 38.901 Table 7.5-3: kept for import compatibility (tests may still import
# this symbol).  The value is no longer used internally for sub-ray generation.
# ---------------------------------------------------------------------------
_TR38901_RAY_OFFSETS_DEG: NDArray[np.float64] = np.array(
    [
        0.0447,
        0.1413,
        0.2492,
        0.3715,
        0.5129,
        0.6797,
        0.8844,
        1.1481,
        1.5195,
        2.1551,
        -0.0447,
        -0.1413,
        -0.2492,
        -0.3715,
        -0.5129,
        -0.6797,
        -0.8844,
        -1.1481,
        -1.5195,
        -2.1551,
    ],
    dtype=np.float64,
)  # shape (20,) — kept for backward import only


# ---------------------------------------------------------------------------
# LSP statistics: only the UMi LOS row is used (Sec 3.2.2 of predecessor
# report). NLOS entry is removed; LOS parameters are applied universally.
# ---------------------------------------------------------------------------
_LSP_PARAMS: dict = {
    ("umi", True): {
        "ds_mu": -7.19,
        "ds_sigma": 0.40,  # TR 38.901 Table 7.5-6 UMi LOS DS
        "asa_mu": 1.81,
        "asa_sigma": 0.20,  # UMi LOS ASA
        "asd_mu": 1.20,
        "asd_sigma": 0.41,  # UMi LOS ASD
        "k_mu_db": 9.0,
        "k_sigma_db": 5.0,  # UMi LOS K-factor (dB)
        "sf_sigma_db": 4.0,  # UMi LOS shadow fading std
        "r_tau": 3.0,  # delay-scaling ratio
        "xi_db": 3.0,  # per-cluster shadow std (unused in geo model)
        "n_clusters": 12,
        "n_rays": 20,
        "cluster_asa_deg": 17.0,  # intra-cluster ASA, Table 3.2
        "cluster_asd_deg": 3.0,  # intra-cluster ASD, Table 3.2
    },
}


@dataclass(frozen=True)
class ChannelParams:
    """Configuration for a channel Monte Carlo trial.

    Defaults correspond to the UMi LOS table (Table 3.1 of the predecessor
    report, equivalently TR 38.901 Table 7.5-6 UMi LOS row).
    """

    fc_hz: float = 28e9
    bandwidth_hz: float = 100e6
    h_bs: float = 10.0
    h_ut: float = 1.5
    n_clusters: int = 12
    n_rays_per_cluster: int = 20
    cluster_asa_deg: float = 17.0  # Table 3.2 UMi LOS
    cluster_asd_deg: float = 3.0
    k_factor_mean_db: float = 9.0
    k_factor_std_db: float = 5.0
    cluster_shadow_std_db: float = 3.0
    los_probability: float = 1.0  # set <1.0 for stochastic LOS state
    blockage_rate_per_sec: float = 0.0  # unused; kept for API compat
    scenario: str = "umi"
    ue_speed_mps: float = 0.0
    # Scatterer placement radius around BS (m). Tune to match delay spread.
    scatterer_radius_m: float = 200.0
    # Self-blocker width: 120 deg (portrait) or 160 deg (landscape).
    self_blocker_width_deg: float = 120.0
    # When True, skip the NLOS cluster sum (LOS component only).
    # Used for the "without reflectors" Fig 6.9 variant.
    disable_clusters: bool = False


# ---------------------------------------------------------------------------
# Path loss (unchanged — matches predecessor report Sec 3.2.1)
# ---------------------------------------------------------------------------


def umi_path_loss_db(d_2d_m: float, fc_hz: float, h_bs: float, h_ut: float, los: bool) -> float:
    """3GPP TR 38.901 §7.4.1, UMi-Street-Canyon path loss in dB.

    Two-slope LOS model with breakpoint distance, NLOS lower-bounded by LOS.
    """
    fc_ghz = fc_hz / 1e9
    h_e = 1.0
    d_bp = 4 * (h_bs - h_e) * (h_ut - h_e) * fc_hz / SPEED_OF_LIGHT
    d_3d = np.sqrt(d_2d_m**2 + (h_bs - h_ut) ** 2)
    pl_los_close = 32.4 + 21 * np.log10(d_3d) + 20 * np.log10(fc_ghz)
    if d_2d_m <= d_bp:
        pl_los = pl_los_close
    else:
        pl_los = (
            32.4
            + 40 * np.log10(d_3d)
            + 20 * np.log10(fc_ghz)
            - 9.5 * np.log10(d_bp**2 + (h_bs - h_ut) ** 2)
        )
    if los:
        return pl_los
    pl_nlos = 35.3 * np.log10(d_3d) + 22.4 + 21.3 * np.log10(fc_ghz) - 0.3 * (h_ut - 1.5)
    return max(pl_los, pl_nlos)


def uma_path_loss_db(d_2d_m: float, fc_hz: float, h_bs: float, h_ut: float, los: bool) -> float:
    """3GPP TR 38.901 §7.4.1, UMa path loss in dB (convenience wrapper)."""
    fc_ghz = fc_hz / 1e9
    h_e = 1.0
    d_bp = 4 * (h_bs - h_e) * (h_ut - h_e) * fc_hz / SPEED_OF_LIGHT
    d_3d = np.sqrt(d_2d_m**2 + (h_bs - h_ut) ** 2)
    if d_2d_m <= d_bp:
        pl_los = 28.0 + 22 * np.log10(d_3d) + 20 * np.log10(fc_ghz)
    else:
        pl_los = (
            28.0
            + 40 * np.log10(d_3d)
            + 20 * np.log10(fc_ghz)
            - 9.0 * np.log10(d_bp**2 + (h_bs - h_ut) ** 2)
        )
    if los:
        return pl_los
    pl_nlos = 13.54 + 39.08 * np.log10(d_3d) + 20 * np.log10(fc_ghz) - 0.6 * (h_ut - 1.5)
    return max(pl_los, pl_nlos)


# ---------------------------------------------------------------------------
# LOS probability (unchanged — matches predecessor report Sec 3.2.1)
# ---------------------------------------------------------------------------


def umi_los_probability(d_2d_m: float) -> float:
    """TR 38.901 §7.4.2 UMi-Street-Canyon LOS probability.

    P_LOS = min(18/d_2D, 1) * (1 - exp(-d_2D/36)) + exp(-d_2D/36)
    """
    d = max(d_2d_m, 1.0)
    return min(18.0 / d, 1.0) * (1.0 - np.exp(-d / 36.0)) + np.exp(-d / 36.0)


# ---------------------------------------------------------------------------
# LSP draws (always from LOS table per predecessor Sec 3.2.2)
# ---------------------------------------------------------------------------

_NO_K_FACTOR = object()  # sentinel: suppress K-factor draw (NLOS state)


def _draw_lsps(
    rng: np.random.Generator,
    lsp: dict,
    k_mu_db_override: float | None = None,
    k_sigma_db_override: float | None = None,
) -> dict:
    """Draw independent large-scale parameters for one UE drop.

    Returns dict with keys: ds_s, asa_deg, asd_deg, k_db, k_lin, sf_db.
    Always uses the LOS statistics table regardless of is_los state.

    Pass k_mu_db_override=_NO_K_FACTOR to suppress K-factor (NLOS case).
    """
    ds_s = 10 ** rng.normal(lsp["ds_mu"], lsp["ds_sigma"])
    asa_deg = 10 ** rng.normal(lsp["asa_mu"], lsp["asa_sigma"])
    asd_deg = 10 ** rng.normal(lsp["asd_mu"], lsp["asd_sigma"])
    sf_db = rng.normal(0.0, lsp["sf_sigma_db"])

    if k_mu_db_override is _NO_K_FACTOR:
        k_db = -np.inf
        k_lin = 0.0
    else:
        k_mu = k_mu_db_override if k_mu_db_override is not None else lsp["k_mu_db"]
        k_sigma = k_sigma_db_override if k_sigma_db_override is not None else lsp["k_sigma_db"]
        if k_mu is not None:
            k_db = float(
                np.clip(rng.normal(k_mu, k_sigma if k_sigma is not None else 0.0), -3.0, 20.0)
            )
            k_lin = float(10 ** (k_db / 10.0))
        else:
            k_db = -np.inf
            k_lin = 0.0
    return {
        "ds_s": ds_s,
        "asa_deg": asa_deg,
        "asd_deg": asd_deg,
        "k_db": k_db,
        "k_lin": k_lin,
        "sf_db": sf_db,
    }


# ---------------------------------------------------------------------------
# Cluster delays (unchanged procedure)
# ---------------------------------------------------------------------------


def _draw_cluster_delays(
    rng: np.random.Generator, n: int, r_tau: float, ds_s: float
) -> NDArray[np.float64]:
    """TR 38.901 §7.5 Step 5: draw and sort cluster delays.

    tau'_n ~ -r_tau * DS * ln(X_n),  X_n ~ Uniform(0,1).
    Normalised to tau'_1 = 0.
    """
    x = rng.uniform(0.0, 1.0, size=n)
    tau_raw = -r_tau * ds_s * np.log(x)
    return np.sort(tau_raw - tau_raw.min())


# ---------------------------------------------------------------------------
# Geometric cluster power (predecessor Sec 3.2.2)
# ---------------------------------------------------------------------------


def _geometric_cluster_powers(
    bs_xy: NDArray[np.float64],
    ue_xy: NDArray[np.float64],
    scatterer_xy: NDArray[np.float64],
    fc_hz: float,
    h_bs: float,
    h_ut: float,
    k_lin: float,
    is_los: bool,
) -> NDArray[np.float64]:
    """Compute cluster powers from geometric extra path-length (Sec 3.2.2).

    For each scatterer S:
      extra = d(BS,S) + d(S,UE) - d(BS,UE)
      cluster_pl_db = PL(d_BS_UE + extra) - PL(d_BS_UE)   [additional loss]
      cluster_amp   = 10**(-cluster_pl_db / 20)

    Amplitude is relative to the direct path.  Powers are normalised to sum=1.
    """
    d_bs_ue = float(np.linalg.norm(ue_xy - bs_xy))
    d_bs_s = np.linalg.norm(scatterer_xy - bs_xy, axis=1)
    d_s_ue = np.linalg.norm(scatterer_xy - ue_xy, axis=1)
    extra = d_bs_s + d_s_ue - d_bs_ue

    pl_ref = umi_path_loss_db(max(d_bs_ue, 1.0), fc_hz, h_bs, h_ut, los=False)
    pl_sc = np.array(
        [umi_path_loss_db(max(d_bs_ue + ex, 1.0), fc_hz, h_bs, h_ut, los=False) for ex in extra]
    )
    delta_pl_db = pl_sc - pl_ref  # additional loss per cluster (>= 0)

    amp = 10 ** (-delta_pl_db / 20.0)

    if is_los and k_lin > 0:
        amp = amp / (1.0 + k_lin)

    total = amp.sum()
    if total <= 0:
        return np.ones(len(amp)) / len(amp)
    return amp / total


# ---------------------------------------------------------------------------
# Laplacian sub-ray angle offsets (predecessor Eq 3.14, Sec 3.2.3)
# ---------------------------------------------------------------------------


def _laplacian_subray_offsets(
    rng: np.random.Generator,
    n_clusters: int,
    n_rays: int,
    spread_rad: float,
) -> NDArray[np.float64]:
    """Draw sub-ray AoA/AoD offsets from Laplacian with scale = spread/sqrt(2).

    Returns shape (n_clusters, n_rays).

    Per Eq 3.14: the first sub-ray of each cluster keeps offset=0 (large-scale
    component at cluster mean); remaining n_rays-1 rays are Laplacian draws
    centred at 0, distributed around the cluster mean.
    Laplacian scale: S_alpha = spread / sqrt(2)  (matches the parameterisation
    in Eq 3.13: p(alpha) = 1/(2*S) * exp(-|alpha-mu|/S)).
    """
    scale = spread_rad / math.sqrt(2.0)
    offsets = np.zeros((n_clusters, n_rays))
    if n_rays > 1:
        offsets[:, 1:] = rng.laplace(loc=0.0, scale=scale, size=(n_clusters, n_rays - 1))
    return offsets


# ---------------------------------------------------------------------------
# Blockage Model A (predecessor Sec 3.2.4, Eqs 3.15–3.18)
# ---------------------------------------------------------------------------


def _ked_attenuation_db(
    phi_rel: NDArray[np.float64], phi_k: float, x_k: float
) -> NDArray[np.float64]:
    """KED-based attenuation (dB) for a single non-self blocker (Eq 3.16–3.17).

    phi_rel: array of relative AoA angles (rad) in UE body frame.
    phi_k:   blocker centre angle (rad).
    x_k:     blocker width (rad).

    Returns attenuation array in dB (>= 0).
    """
    half = x_k / 2.0
    delta = phi_rel - phi_k
    # Wrap delta to [-pi, pi]
    delta = (delta + np.pi) % (2 * np.pi) - np.pi

    outside = np.abs(delta) > x_k
    result = np.zeros_like(phi_rel)

    # Determine sign pattern per angular region (Eq 3.16 sign table)
    sign_plus = np.where(delta > half, -1.0, np.where(delta >= -half, 1.0, 1.0))
    sign_minus = np.where(delta > half, 1.0, np.where(delta >= -half, 1.0, -1.0))

    # Eq 3.17: beta_k(phi) = (pi/2) * sqrt((pi/lambda_eff) - 1) / cos(delta - half)
    # The TR 38.901 formula uses lambda=0.4 m (≈750 MHz) as a shape parameter.
    # We interpret it as a dimensionless shape parameter beta_scale = pi/lambda_eff.
    # From the report Eq 3.17: beta = (pi/2)*sqrt((pi/lambda_eff - 1) / |cos(delta - half)|)
    # We use lambda_eff = 0.4 (shape, not wavelength) as per TR 38.901 §7.6.4.1.
    lambda_eff = 0.4
    beta_scale = np.pi / lambda_eff - 1.0  # = pi/0.4 - 1 ≈ 6.85

    with np.errstate(divide="ignore", invalid="ignore"):
        cos_val_plus = np.cos(np.abs(delta) - half)
        cos_val_minus = np.cos(np.abs(delta) + half)
        # Avoid division by zero at edges
        cos_val_plus = np.where(np.abs(cos_val_plus) < 1e-9, 1e-9, cos_val_plus)
        cos_val_minus = np.where(np.abs(cos_val_minus) < 1e-9, 1e-9, cos_val_minus)

        beta_plus = (np.pi / 2.0) * np.sqrt(np.maximum(beta_scale / np.abs(cos_val_plus), 0.0))
        beta_minus = (np.pi / 2.0) * np.sqrt(np.maximum(beta_scale / np.abs(cos_val_minus), 0.0))

    t_plus = np.arctan(sign_plus * beta_plus) / np.pi
    t_minus = np.arctan(sign_minus * beta_minus) / np.pi

    # Eq 3.16: L_k = 20*log10(1 - t_plus - t_minus) * 22   [clipped at 0]
    inner = 1.0 - t_plus - t_minus
    inner = np.maximum(inner, 1e-10)
    atten = np.abs(20.0 * np.log10(inner) * 22.0)
    result = np.where(outside, 0.0, atten)
    return result


@dataclass
class BlockageState:
    """Persistent blockage state per UE trial (non-self blockers drift over time).

    Initialised once per trial; updated per channel_matrix() call via
    update(ue_xy, time_s).  Implements Eq 3.18 autocorrelation model.
    """

    n_non_self: int
    phi_k: NDArray[np.float64]  # non-self blocker centre angles (rad), body frame
    x_k: NDArray[np.float64]  # non-self blocker widths (rad)
    self_width_rad: float  # self-blocker width (rad)
    self_centre_rad: float = math.pi  # centre at back of UE (180 deg)
    _prev_ue_xy: NDArray[np.float64] | None = field(default=None)
    _dcorr_m: float = 10.0  # correlation distance (m)
    _rng: np.random.Generator | None = field(default=None)  # trial RNG for Eq 3.18

    def update_blocker_angles(self, ue_xy: NDArray[np.float64]) -> None:
        """Drift non-self blocker angles per Eq 3.18 based on UE displacement."""
        if self._prev_ue_xy is None:
            self._prev_ue_xy = ue_xy.copy()
            return
        dx = float(np.linalg.norm(ue_xy - self._prev_ue_xy))
        self._prev_ue_xy = ue_xy.copy()
        if dx < 1e-9:
            return
        # Autocorrelation: R(dx) = exp(-dx/dcorr). Drift sigma per unit distance.
        sigma = math.sqrt(2.0 * dx / self._dcorr_m)
        # Use trial RNG so drift is stochastic across calls (not position-seeded).
        rng = self._rng if self._rng is not None else np.random.default_rng()
        self.phi_k += rng.normal(0.0, sigma, size=self.n_non_self)
        self.phi_k = (self.phi_k + np.pi) % (2 * np.pi) - np.pi

    def attenuation_db(self, aoa_body_frame: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return total blockage attenuation (dB) for each AoA in body frame.

        aoa_body_frame: array of angles (rad). Returns array of same shape.
        """
        total = np.zeros_like(aoa_body_frame)

        # Self-blocker (Eq 3.15): flat 30 dB within ±width/2 of back
        half_sb = self.self_width_rad / 2.0
        delta_sb = (aoa_body_frame - self.self_centre_rad + np.pi) % (2 * np.pi) - np.pi
        total = np.where(np.abs(delta_sb) <= half_sb, total + 30.0, total)

        # Non-self blockers (Eqs 3.16–3.17)
        for k in range(self.n_non_self):
            total += _ked_attenuation_db(aoa_body_frame, self.phi_k[k], self.x_k[k])

        return total


def _init_blockage_state(rng: np.random.Generator, self_width_deg: float) -> BlockageState:
    """Initialise blockage state: 1 self-blocker + 4 non-self blockers."""
    n = 4
    phi_k = rng.uniform(-np.pi, np.pi, size=n)
    x_k = np.deg2rad(rng.uniform(5.0, 15.0, size=n))
    return BlockageState(
        n_non_self=n,
        phi_k=phi_k,
        x_k=x_k,
        self_width_rad=math.radians(self_width_deg),
        _rng=rng,
    )


# ---------------------------------------------------------------------------
# Main channel realisation
# ---------------------------------------------------------------------------


@dataclass
class ChannelRealisation:
    """One Monte Carlo realisation: scatterer geometry fixed at trial start.

    Implements the predecessor MSc report Sec 3.2 channel model:
      - LOS params always used (Table 3.1); blockage handles LOS/NLOS split.
      - Laplacian sub-ray offsets (Eq 3.14).
      - Geometric cluster powers (Sec 3.2.2).
      - No random initial phases.
      - Blockage Model A (Eqs 3.15–3.18), evaluated per UE pose.
    """

    params: ChannelParams
    bs_xy: NDArray[np.float64]
    bs_yaw: float
    n_bs_elements: int
    n_ue_elements: int
    rng: np.random.Generator
    is_los: bool = True
    scatterer_xy: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 2)))
    cluster_powers: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    cluster_delays_s: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    sub_ray_aoa_offsets: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    sub_ray_aod_offsets: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    k_lin: float = 1.0
    los_blocked: bool = False
    lsp: dict = field(default_factory=dict)
    _blockage: BlockageState | None = field(default=None)

    def __post_init__(self) -> None:
        rng = self.rng
        p = self.params

        # ------------------------------------------------------------------
        # Step 1 – LOS / NLOS determination (stochastic if los_probability<1)
        # ------------------------------------------------------------------
        if p.los_probability < 1.0:
            self.is_los = rng.random() < p.los_probability

        # ------------------------------------------------------------------
        # Steps 3–4 – Large-scale parameters: ALWAYS from UMi LOS table
        # (predecessor Sec 3.2.2: "generate the channel using the LOS
        # parameters and later apply the blockage model")
        # ------------------------------------------------------------------
        lsp_table = _LSP_PARAMS[("umi", True)]
        drawn = _draw_lsps(
            rng,
            lsp_table,
            k_mu_db_override=(p.k_factor_mean_db if self.is_los else _NO_K_FACTOR),  # type: ignore[arg-type]  # _NO_K_FACTOR is an object() sentinel for NLOS suppression
            k_sigma_db_override=(p.k_factor_std_db if self.is_los else None),
        )
        self.lsp = drawn
        self.k_lin = drawn["k_lin"]

        n = lsp_table["n_clusters"]
        nr = lsp_table["n_rays"]
        cluster_asa_rad = math.radians(lsp_table["cluster_asa_deg"])
        cluster_asd_rad = math.radians(lsp_table["cluster_asd_deg"])

        # ------------------------------------------------------------------
        # Step 5 – Cluster delays
        # ------------------------------------------------------------------
        taus = _draw_cluster_delays(rng, n, lsp_table["r_tau"], drawn["ds_s"])
        self.cluster_delays_s = taus

        # ------------------------------------------------------------------
        # Step 7 – Scatterer positions: uniform disc around BS
        # ------------------------------------------------------------------
        radius = p.scatterer_radius_m
        r_vals = radius * np.sqrt(rng.random(n))
        theta_vals = 2 * np.pi * rng.random(n)
        self.scatterer_xy = self.bs_xy + np.column_stack(
            [r_vals * np.cos(theta_vals), r_vals * np.sin(theta_vals)]
        )

        # ------------------------------------------------------------------
        # Cluster powers: geometric (Sec 3.2.2)
        # Deferred until channel_matrix() where ue_xy is known; pre-compute
        # a placeholder using BS as UE position; updated on first call.
        # ------------------------------------------------------------------
        self.cluster_powers = np.ones(n) / n  # placeholder, updated per call

        # ------------------------------------------------------------------
        # Step 8 – Sub-ray offsets: Laplacian (Eq 3.14, Sec 3.2.3)
        # First sub-ray keeps cluster mean (offset=0); L-1 others are draws.
        # ------------------------------------------------------------------
        self.sub_ray_aoa_offsets = _laplacian_subray_offsets(rng, n, nr, cluster_asa_rad)
        self.sub_ray_aod_offsets = _laplacian_subray_offsets(rng, n, nr, cluster_asd_rad)

        # ------------------------------------------------------------------
        # Blockage state (Model A, Sec 3.2.4)
        # ------------------------------------------------------------------
        self._blockage = _init_blockage_state(rng, p.self_blocker_width_deg)

    def _apply_blockage(
        self, ue_xy: NDArray[np.float64], ue_yaw: float, cluster_aoa_world: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return per-cluster blockage attenuation (dB) for current UE pose.

        cluster_aoa_world: world-frame AoA for each cluster (rad), shape (n,).
        Also evaluates LOS blockage (returned as element [-1] of the n+1 array).
        Returns shape (n+1,) where index -1 is LOS attenuation.
        """
        blk = self._blockage
        if blk is None:
            n = len(cluster_aoa_world)
            return np.zeros(n + 1)

        blk.update_blocker_angles(ue_xy)

        # Convert world-frame AoA to body frame (subtract UE heading)
        aoa_body = _wrap_pi(cluster_aoa_world - ue_yaw)

        # LOS direction in body frame
        los_aoa_world = float(np.arctan2(self.bs_xy[1] - ue_xy[1], self.bs_xy[0] - ue_xy[0]))
        los_body = float(_wrap_pi(los_aoa_world - ue_yaw))

        all_aoa = np.append(aoa_body, los_body)
        return blk.attenuation_db(all_aoa)

    def channel_matrix(
        self, ue_xy: NDArray[np.float64], ue_yaw: float, time_s: float = 0.0
    ) -> NDArray[np.complex128]:
        """(n_ue_elements, n_bs_elements) downlink channel matrix at UE pose.

        Convention: y = w^H H f x, H in C^{N_UE x N_BS}.
        Blockage Model A (Sec 3.2.4) is evaluated per pose.
        Cluster powers are re-computed geometrically per pose (Sec 3.2.2).
        """
        p = self.params
        ue_xy = np.asarray(ue_xy, dtype=np.float64)

        d_2d = float(np.linalg.norm(ue_xy - self.bs_xy))
        pl_db = umi_path_loss_db(d_2d, p.fc_hz, p.h_bs, p.h_ut, los=self.is_los)
        pl_lin = 10 ** (-pl_db / 20.0)

        lam = SPEED_OF_LIGHT / p.fc_hz

        # Recompute geometric cluster powers for this UE position
        cluster_powers = _geometric_cluster_powers(
            self.bs_xy,
            ue_xy,
            self.scatterer_xy,
            p.fc_hz,
            p.h_bs,
            p.h_ut,
            self.k_lin,
            self.is_los,
        )
        self.cluster_powers = cluster_powers

        h = np.zeros((self.n_ue_elements, self.n_bs_elements), dtype=np.complex128)

        # Scatterer world-frame angles
        sc_x = self.scatterer_xy[:, 0]
        sc_y = self.scatterer_xy[:, 1]
        aoa_world_c = np.arctan2(sc_y - ue_xy[1], sc_x - ue_xy[0])
        aod_world_c = np.arctan2(sc_y - self.bs_xy[1], sc_x - self.bs_xy[0])

        # Blockage attenuation (n_clusters + 1); last element is LOS
        blk_all = self._apply_blockage(ue_xy, ue_yaw, aoa_world_c)
        blk_cluster_amp = 10 ** (-blk_all[:-1] / 20.0)
        blk_los_amp = float(10 ** (-blk_all[-1] / 20.0))

        # ------------------------------------------------------------------
        # LOS direct path
        # ------------------------------------------------------------------
        if self.is_los:
            aoa_world_los = float(np.arctan2(self.bs_xy[1] - ue_xy[1], self.bs_xy[0] - ue_xy[0]))
            aod_world_los = float(np.arctan2(ue_xy[1] - self.bs_xy[1], ue_xy[0] - self.bs_xy[0]))
            aoa_rel_los = float(_wrap_pi(aoa_world_los - ue_yaw))
            aod_rel_los = float(_wrap_pi(aod_world_los - self.bs_yaw))
            a_ue = steering_vector(self.n_ue_elements, aoa_rel_los)
            a_bs = steering_vector(self.n_bs_elements, aod_rel_los)
            los_amp = pl_lin * np.sqrt(self.k_lin / (1.0 + self.k_lin)) * blk_los_amp
            los_doppler = _los_doppler_phase(ue_xy, self.bs_xy, p.ue_speed_mps, ue_yaw, lam, time_s)
            h += los_amp * los_doppler * np.outer(a_ue, a_bs.conj())

        # ------------------------------------------------------------------
        # NLOS cluster contributions (vectorised)
        # ------------------------------------------------------------------
        nlos_total_amp = pl_lin / np.sqrt(1.0 + self.k_lin) if self.is_los else pl_lin

        n_cl = len(cluster_powers)
        nr = p.n_rays_per_cluster

        # Per-cluster amplitude including blockage
        cluster_amp = nlos_total_amp * np.sqrt(cluster_powers) * blk_cluster_amp  # (n_cl,)

        # Sub-ray relative angles (n_cl, nr)
        aoa_rel: NDArray[np.float64] = _wrap_pi(
            aoa_world_c[:, None] + self.sub_ray_aoa_offsets - ue_yaw
        )  # type: ignore[assignment]  # _wrap_pi returns Union; array at runtime
        aod_rel: NDArray[np.float64] = _wrap_pi(
            aod_world_c[:, None] + self.sub_ray_aod_offsets - self.bs_yaw
        )  # type: ignore[assignment]  # same

        # Steering vectors over all sub-rays
        n_ue_idx = np.arange(self.n_ue_elements)
        n_bs_idx = np.arange(self.n_bs_elements)
        sin_aoa = np.sin(aoa_rel).reshape(-1)
        sin_aod = np.sin(aod_rel).reshape(-1)
        phase_ue = -2.0 * np.pi * 0.5 * np.outer(sin_aoa, n_ue_idx)
        phase_bs = -2.0 * np.pi * 0.5 * np.outer(sin_aod, n_bs_idx)
        a_ue_all = np.exp(1j * phase_ue) / np.sqrt(self.n_ue_elements)
        a_bs_all = np.exp(1j * phase_bs) / np.sqrt(self.n_bs_elements)

        # Doppler phase per sub-ray
        if p.ue_speed_mps != 0.0 and time_s != 0.0:
            ray_angle_world = aoa_world_c[:, None] + self.sub_ray_aoa_offsets
            v_x = p.ue_speed_mps * np.cos(ue_yaw)
            v_y = p.ue_speed_mps * np.sin(ue_yaw)
            f_d = (v_x * np.cos(ray_angle_world) + v_y * np.sin(ray_angle_world)) / lam
            doppler_phase = np.exp(1j * 2.0 * np.pi * f_d * time_s).reshape(-1)
        else:
            doppler_phase = np.ones(n_cl * nr, dtype=np.complex128)

        # Ray amplitude per sub-ray (n_cl*nr,): no random initial phase (Sec 3.2)
        ray_amp = (cluster_amp[:, None] * doppler_phase.reshape(n_cl, nr) / np.sqrt(nr)).reshape(-1)

        if not p.disable_clusters:
            h += np.einsum("ri,rj,r->ij", a_ue_all, a_bs_all.conj(), ray_amp)
        return h

    def los_aoa_world(self, ue_xy: NDArray[np.float64]) -> float:
        return float(np.arctan2(self.bs_xy[1] - ue_xy[1], self.bs_xy[0] - ue_xy[0]))

    def los_aod_world(self, ue_xy: NDArray[np.float64]) -> float:
        return float(np.arctan2(ue_xy[1] - self.bs_xy[1], ue_xy[0] - self.bs_xy[0]))


# ---------------------------------------------------------------------------
# Doppler helpers
# ---------------------------------------------------------------------------


def _los_doppler_phase(
    ue_xy: NDArray[np.float64],
    bs_xy: NDArray[np.float64],
    speed_mps: float,
    ue_yaw: float,
    lam: float,
    time_s: float,
) -> complex:
    """Doppler phase for LOS ray; v aligned with ue_yaw heading."""
    if speed_mps == 0.0 or time_s == 0.0:
        return 1.0 + 0j
    d = bs_xy - ue_xy
    norm = np.linalg.norm(d)
    if norm < 1e-9:
        return 1.0 + 0j
    k_hat = d / norm
    v_vec = speed_mps * np.array([np.cos(ue_yaw), np.sin(ue_yaw)])
    f_d = float(np.dot(v_vec, k_hat)) / lam
    return complex(np.exp(1j * 2 * np.pi * f_d * time_s))


def _ray_doppler_phase(
    ray_angle_world: float, speed_mps: float, ue_yaw: float, lam: float, time_s: float
) -> complex:
    """Doppler phase for a sub-ray arriving from world angle ray_angle_world."""
    if speed_mps == 0.0 or time_s == 0.0:
        return 1.0 + 0j
    k_hat = np.array([np.cos(ray_angle_world), np.sin(ray_angle_world)])
    v_vec = speed_mps * np.array([np.cos(ue_yaw), np.sin(ue_yaw)])
    f_d = float(np.dot(v_vec, k_hat)) / lam
    return complex(np.exp(1j * 2 * np.pi * f_d * time_s))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _wrap_pi(a: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Free-space single-LOS channel (unchanged API)
# ---------------------------------------------------------------------------


@dataclass
class FreeSpaceLosChannel:
    """Single-LOS-component channel for the noiseless rotation experiment."""

    bs_xy: NDArray[np.float64]
    bs_yaw: float
    n_bs_elements: int
    n_ue_elements: int

    def channel_matrix(
        self, ue_xy: NDArray[np.float64], ue_yaw: float, time_s: float = 0.0
    ) -> NDArray[np.complex128]:
        aoa_world = np.arctan2(self.bs_xy[1] - ue_xy[1], self.bs_xy[0] - ue_xy[0])
        aod_world = np.arctan2(ue_xy[1] - self.bs_xy[1], ue_xy[0] - self.bs_xy[0])
        a_ue = steering_vector(self.n_ue_elements, float(_wrap_pi(aoa_world - ue_yaw)))
        a_bs = steering_vector(self.n_bs_elements, float(_wrap_pi(aod_world - self.bs_yaw)))
        return np.outer(a_ue, a_bs.conj())


@dataclass
class PlanarFreeSpaceLosChannel:
    """Single-LOS-component channel with Uniform Planar Array (UPA) endpoints.

    Companion to :class:`FreeSpaceLosChannel` for use with
    :class:`beamsim.codebook.PlanarCodebook`. Both UE and BS arrays are UPAs
    in the xy-plane at half-wavelength spacing; the LOS arrival/departure
    angles are computed from BS / UE positions and the elements are flattened
    in row-major order (``i * n_y + j``) so the channel matrix has the
    expected ``(n_ue_elements, n_bs_elements)`` shape that
    :class:`beamsim.bplm.BPLMState` works with.

    Elevation is treated as zero throughout (LOS in xy-plane), matching
    MATLAB ``placodebook.m`` and the predecessor's azimuth-only scope.
    """

    bs_xy: NDArray[np.float64]
    bs_yaw: float
    n_bs_x: int
    n_bs_y: int
    n_ue_x: int
    n_ue_y: int

    @property
    def n_bs_elements(self) -> int:
        return self.n_bs_x * self.n_bs_y

    @property
    def n_ue_elements(self) -> int:
        return self.n_ue_x * self.n_ue_y

    def channel_matrix(
        self, ue_xy: NDArray[np.float64], ue_yaw: float, time_s: float = 0.0
    ) -> NDArray[np.complex128]:
        from beamsim.codebook import planar_steering_vector

        aoa_world = np.arctan2(self.bs_xy[1] - ue_xy[1], self.bs_xy[0] - ue_xy[0])
        aod_world = np.arctan2(ue_xy[1] - self.bs_xy[1], ue_xy[0] - self.bs_xy[0])
        a_ue = planar_steering_vector(self.n_ue_x, self.n_ue_y, float(_wrap_pi(aoa_world - ue_yaw)))
        a_bs = planar_steering_vector(
            self.n_bs_x, self.n_bs_y, float(_wrap_pi(aod_world - self.bs_yaw))
        )
        return np.outer(a_ue, a_bs.conj())
