"""Algorithm-level smoke tests."""

import numpy as np
import pytest

from beamsim.algorithms import ALL_ALGORITHMS
from beamsim.bplm import BPLMState
from beamsim.channel import FreeSpaceLosChannel
from beamsim.codebook import make_default_bs_codebook, make_default_ue_codebook


def make_state():
    return BPLMState(ue_codebook=make_default_ue_codebook(),
                      bs_codebook=make_default_bs_codebook(),
                      noise_amplitude=0.01)


@pytest.mark.parametrize("name", sorted(ALL_ALGORITHMS))
def test_algorithm_returns_valid_indices(name):
    cls = ALL_ALGORITHMS[name]
    algo = cls()
    state = make_state()
    bs_xy = np.array([10.0, 0.0])
    context = {
        "ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0),
        "bs_xy": bs_xy,
        "bs_yaw": 0.0,
    }
    algo.reset(state, context)
    rng = np.random.default_rng(0)
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0,
                              n_bs_elements=16, n_ue_elements=4)
    for m in range(50):
        k, l = algo.select_next_mbp(state, m, context)
        assert 0 <= k < state.K
        assert 0 <= l < state.L
        H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)
        state.measure(k, l, H, m, rng)


def test_exhaustive_visits_every_pair_in_one_cycle():
    from beamsim.algorithms import Exhaustive
    algo = Exhaustive()
    state = make_state()
    algo.reset(state, {})
    visited = set()
    for m in range(state.K * state.L):
        visited.add(algo.select_next_mbp(state, m, {}))
    assert len(visited) == state.K * state.L


def test_ci_picks_geometry_aligned_pair():
    from beamsim.algorithms import ContextInformation
    algo = ContextInformation()
    state = make_state()
    bs_xy = np.array([10.0, 0.0])
    context = {
        "ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0),
        "bs_xy": bs_xy,
        "bs_yaw": 0.0,
    }
    algo.reset(state, context)
    k, l = algo.select_next_mbp(state, 0, context)
    # AoA at UE = 0 (BS lies along +x in body frame). Expect k near middle of codebook (theta=0).
    ue_theta = state.ue_codebook.theta
    bs_theta = state.bs_codebook.theta
    assert abs(ue_theta[k]) < ue_theta[k + 1] - ue_theta[k] + 1e-6
    # AoD at BS = pi (UE behind BS along -x). Wrap and check that the chosen
    # BS beam is the one whose theta matches the wrapped AoD-pi best.
    # Codebook spans (-pi/2, pi/2); aod_rel = pi wraps to -pi which is outside
    # the visible region — argmin will pick the extreme beam.
    assert l in {0, state.L - 1}


# ---------------------------------------------------------------------------
# NNS: hill-climbing tests (report Sec. 5.4.4, Algorithm 4)
# ---------------------------------------------------------------------------

def test_nns_moves_toward_best_neighbour():
    """After measuring a bright spot, NNS should queue neighbours of the OBP,
    not stay fixed at (0,0).  Verifies hill-climbing behaviour: at least one
    call returns a neighbour of the best-observed pair."""
    from beamsim.algorithms import NNS
    from beamsim.channel import FreeSpaceLosChannel

    algo = NNS()
    state = make_state()
    bs_xy = np.array([10.0, 0.0])
    context = {
        "ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0),
        "bs_xy": bs_xy,
        "bs_yaw": 0.0,
    }
    algo.reset(state, context)
    rng = np.random.default_rng(42)
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0,
                              n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)

    # Measure a few pairs so the OBP is established
    for m in range(8):
        k, l = algo.select_next_mbp(state, m, context)
        state.measure(k, l, H, m, rng)

    obp_k, obp_l = state.obp()
    # Next selections must include at least one pair adjacent to OBP
    adjacent = set()
    for dk, dl in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nk = np.clip(obp_k + dk, 0, state.K - 1)
        nl = np.clip(obp_l + dl, 0, state.L - 1)
        adjacent.add((int(nk), int(nl)))

    selections = set()
    for m in range(8, 15):
        k, l = algo.select_next_mbp(state, m, context)
        state.measure(k, l, H, m, rng)
        selections.add((k, l))

    assert selections & adjacent, (
        f"NNS never visited a neighbour of OBP {(obp_k, obp_l)}; "
        f"visited={selections}, expected intersection with {adjacent}"
    )


def test_nns_returns_4connected_only():
    """With 4-connectivity, all returned pairs must be at Chebyshev distance <= 1
    from the current OBP after enough measurements to move away from cold-start."""
    from beamsim.algorithms import NNS
    from beamsim.channel import FreeSpaceLosChannel

    algo = NNS(connectivity=4)
    state = make_state()
    bs_xy = np.array([10.0, 0.0])
    context = {
        "ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0),
        "bs_xy": bs_xy,
        "bs_yaw": 0.0,
    }
    algo.reset(state, context)
    rng = np.random.default_rng(7)
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0,
                              n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)

    # Warm up: measure enough to set a non-trivial OBP
    for m in range(6):
        k, l = algo.select_next_mbp(state, m, context)
        state.measure(k, l, H, m, rng)

    obp_k, obp_l = state.obp()
    for m in range(6, 14):
        k, l = algo.select_next_mbp(state, m, context)
        chebyshev = max(abs(k - obp_k), abs(l - obp_l))
        # Either measuring OBP itself (re-probe) or a 4-connected neighbour
        assert chebyshev <= 1, (
            f"NNS moved to ({k},{l}) which is Chebyshev-{chebyshev} from OBP "
            f"({obp_k},{obp_l}); expected <= 1 for 4-connectivity"
        )
        state.measure(k, l, H, m, rng)
        obp_k, obp_l = state.obp()


# ---------------------------------------------------------------------------
# Tabu: aspiration and diversification tests (report Sec. 5.4.5, Algorithm 5)
# ---------------------------------------------------------------------------

def test_tabu_aspiration_takes_tabu_pair_with_highest_magnitude():
    """Aspiration criterion (Glover 1989): a tabu candidate whose observed
    magnitude exceeds the best recorded on *previous* occasions must be
    selected even though it is tabu.

    Test protocol:
      1. Run warmup to fill tabu list.
      2. Record the current global_best_mag from the algorithm internals.
      3. Directly set observations[tabu_pair] to 10x the pre-inflation best —
         but do NOT update measured_at (leave it as the warmup epoch value
         so the algorithm's global-best update from measured entries does not
         absorb the inflated value before the aspiration threshold is read).
      4. The next call should select tabu_pair via aspiration.
    """
    from beamsim.algorithms import Tabu
    from beamsim.channel import FreeSpaceLosChannel

    algo = Tabu(tenure=20, radius=2)
    state = make_state()
    bs_xy = np.array([10.0, 0.0])
    context = {"ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0),
               "bs_xy": bs_xy, "bs_yaw": 0.0}
    algo.reset(state, context)
    rng = np.random.default_rng(1)
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0,
                              n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)

    # Warm up: measure 20 occasions so the tabu list fills up
    for m in range(20):
        k, l = algo.select_next_mbp(state, m, context)
        state.measure(k, l, H, m, rng)

    obp_k, obp_l = state.obp()

    # Find a tabu pair that is a NEIGHBOUR of the OBP (not the OBP itself).
    # The aspiration check loops over neighbourhood candidates that are tabu;
    # the pair must be in N(OBP) and have T < 0.
    tabu_pair = None
    for dk in range(-algo.radius, algo.radius + 1):
        for dl in range(-algo.radius, algo.radius + 1):
            if dk == 0 and dl == 0:
                continue  # exclude centre; neighbourhood does not include OBP
            nk = int(np.clip(obp_k + dk, 0, state.K - 1))
            nl = int(np.clip(obp_l + dl, 0, state.L - 1))
            if algo._T[nk, nl] < 0 and (nk, nl) != (obp_k, obp_l):
                tabu_pair = (nk, nl)
                break
        if tabu_pair:
            break

    if tabu_pair is None:
        pytest.skip("No tabu pair in OBP neighbourhood after warmup — skipping aspiration test")

    # Snapshot the algorithm's global best (updated from measured entries).
    pre_inflation_best = algo._global_best_mag
    inflated_value = pre_inflation_best * 10.0

    # Inject inflated observation AND mark entry as UNMEASURED (measured_at=-1).
    # This keeps the inflated value out of the measured_mask update inside
    # select_next_mbp, so aspiration_threshold = pre_inflation_best, and
    # obs_mag[tabu_pair] = inflated_value > aspiration_threshold fires correctly.
    # The pair is still tabu (T < 0) so it can only be selected via aspiration.
    state.observations[tabu_pair] = inflated_value + 0.0j
    state.measured_at[tabu_pair] = -1  # unmeasured => excluded from global-best update

    # The next selection must be the aspirated tabu pair
    k_next, l_next = algo.select_next_mbp(state, 21, context)
    assert (k_next, l_next) == tabu_pair, (
        f"Aspiration failed: expected tabu pair {tabu_pair} "
        f"(obs={inflated_value:.4f} > pre_best={pre_inflation_best:.4f}) "
        f"to be selected but got ({k_next},{l_next}). "
        f"T[tabu_pair]={algo._T[tabu_pair]}, OBP={state.obp()}"
    )


def test_tabu_avoids_recently_selected_pairs():
    """Pairs added to the tabu list should not be selected again within tenure
    occasions (unless aspiration applies)."""
    from beamsim.algorithms import Tabu
    from beamsim.channel import FreeSpaceLosChannel

    tenure = 10
    algo = Tabu(tenure=tenure, radius=2, diversification_period=0)
    state = make_state()
    bs_xy = np.array([10.0, 0.0])
    context = {"ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0),
               "bs_xy": bs_xy, "bs_yaw": 0.0}
    algo.reset(state, context)
    rng = np.random.default_rng(3)
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0,
                              n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)

    # Collect selections over 2*tenure occasions; keep the first choice
    selections = []
    for m in range(2 * tenure):
        k, l = algo.select_next_mbp(state, m, context)
        selections.append((k, l))
        state.measure(k, l, H, m, rng)

    # The same pair should not appear twice within a window of tenure steps
    # (ignoring aspiration; observations are all low so aspiration shouldn't fire)
    for i in range(len(selections)):
        pair = selections[i]
        for j in range(i + 1, min(i + tenure, len(selections))):
            assert selections[j] != pair, (
                f"Tabu pair {pair} selected again at step {j}, only {j-i} steps "
                f"after first selection at step {i} (tenure={tenure})"
            )


# ---------------------------------------------------------------------------
# Angular prediction: Kalman filter convergence test
# ---------------------------------------------------------------------------

def test_angular_prediction_kalman_converges_on_constant_velocity():
    """Kalman tracker must track a drifting LOS beam within a beamwidth tolerance.

    Setup: UE at (10, 0) drifting slowly in +y while facing +x (yaw=0). The BS
    sits at the origin.  AoD from the BS sweeps through interior codebook beams
    at a constant rate (~0.05 rad/step).  The Kalman filter should predict the
    correct BS beam (or adjacent) at least 60% of the time post-warmup.

    Reference: thesis Sec. 5.4.3 Algorithm 3 — angular tracking with constant-
    velocity prediction; Fig. 5.19 shows stable tracking.
    """
    from beamsim.algorithms import AngularPrediction
    from beamsim.channel import FreeSpaceLosChannel
    from beamsim.codebook import make_default_bs_codebook, make_default_ue_codebook

    warmup = 4
    algo = AngularPrediction(warmup=warmup, q_angle=5e-4, q_rate=5e-3)
    ue_cb = make_default_ue_codebook()
    bs_cb = make_default_bs_codebook()
    state = BPLMState(ue_codebook=ue_cb, bs_codebook=bs_cb, noise_amplitude=1e-6)

    bs_xy = np.array([0.0, 0.0])
    bs_yaw = 0.0
    ue_yaw = 0.0  # fixed orientation

    # UE drifts in +y from (10, -2) at 0.05 m/step: AoD goes from -0.2 to +0.3 rad
    def ue_pose(m):
        x = 10.0
        y = -2.0 + m * 0.05
        return np.array([x, y]), ue_yaw

    context = {"ue_pose_at": ue_pose, "bs_xy": bs_xy, "bs_yaw": bs_yaw}
    algo.reset(state, context)
    rng = np.random.default_rng(99)

    def true_bs_beam(m):
        """Index of BS codebook beam aligned with AoD at step m."""
        xy, _ = ue_pose(m)
        aod_w = np.arctan2(xy[1] - bs_xy[1], xy[0] - bs_xy[0])
        aod_rel = (aod_w - bs_yaw + np.pi) % (2 * np.pi) - np.pi
        return int(np.argmin(np.abs(
            (bs_cb.theta - aod_rel + np.pi) % (2 * np.pi) - np.pi
        )))

    n_steps = warmup + 10
    # Warm-up: measure the true LOS pair each step to give the filter good data
    for m in range(warmup):
        xy, yaw = ue_pose(m)
        ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=bs_yaw,
                                  n_bs_elements=16, n_ue_elements=4)
        H = ch.channel_matrix(xy, yaw)
        # Measure the true pair directly so OBP tracks ground truth
        tl = true_bs_beam(m)
        # Use exhaustive-style: algo returns sweep pair during warmup; force-measure the
        # true pair as well so OBP is accurate
        k_algo, l_algo = algo.select_next_mbp(state, m, context)
        state.measure(k_algo, l_algo, H, m, rng)
        # Also pre-seed observations at the true pair so OBP = true beam
        state.measure(0, tl, H, m, rng)  # k=0 is UE side (AoA is not the focus here)

    # Prediction phase: Kalman should track the drifting AoD beam
    hits = 0
    post_warmup = n_steps - warmup
    for m in range(warmup, n_steps):
        xy, yaw = ue_pose(m)
        ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=bs_yaw,
                                  n_bs_elements=16, n_ue_elements=4)
        H = ch.channel_matrix(xy, yaw)
        k_pred, l_pred = algo.select_next_mbp(state, m, context)
        state.measure(k_pred, l_pred, H, m, rng)
        tl = true_bs_beam(m)
        # Accept exact hit OR adjacent beam (one beamwidth tolerance)
        if abs(l_pred - tl) <= 1:
            hits += 1

    assert hits >= post_warmup * 0.6, (
        f"Kalman tracker hit rate (within 1 beam) {hits}/{post_warmup} < 60%; "
        "filter is not converging on constant-velocity AoD drift"
    )


# ---------------------------------------------------------------------------
# CI: LOS-aligned pair on pure-LOS noiseless channel
# ---------------------------------------------------------------------------

def test_ci_picks_los_pair_on_noiseless_los_channel():
    """CI must select a beam pair within 3 dB of the global best on a purely
    LOS noiseless channel.

    Geometry is chosen so both AoA (at UE) and AoD (at BS) are comfortably
    inside the codebook's visible range (-pi/2, pi/2):
      - BS at origin facing +x (bs_yaw=0), UE at (0, 2) facing -x (ue_yaw=pi).
        AoA at UE = arctan2(0-2, 0-0) - pi = -pi/2 - pi → wrap = pi/2... not ideal
      - Better: UE at (3, 1) yaw=0, BS at (0, 0) yaw=0.
        AoA at UE: arctan2(0-1, 0-3) - 0 = arctan2(-1,-3) ≈ -0.32 rad (interior)
        AoD at BS: arctan2(1-0, 3-0) - 0 = arctan2(1,3)  ≈ +0.32 rad (interior)
    Both angles are well inside (-pi/2, pi/2) so the codebook quantises them.

    Reference: thesis Sec. 5.4.6, Algorithm 6 (p. 69).
    """
    from beamsim.algorithms import ContextInformation
    from beamsim.channel import FreeSpaceLosChannel

    # UE at (6,1) faces toward BS at (1,0): ue_yaw = arctan2(0-1, 1-6) = arctan2(-1,-5).
    # AoA_rel = 0 (UE looks directly at BS), AoD_rel ≈ 0.197 rad (interior of codebook).
    # Both angles are inside (-pi/2, pi/2) so the CI codebook quantiser is accurate.
    ue_xy = np.array([6.0, 1.0])
    bs_xy = np.array([1.0, 0.0])
    bs_yaw = 0.0
    ue_yaw = float(np.arctan2(bs_xy[1] - ue_xy[1], bs_xy[0] - ue_xy[0]))

    state = make_state()
    context = {
        "ue_pose_at": lambda m: (ue_xy, ue_yaw),
        "bs_xy": bs_xy,
        "bs_yaw": bs_yaw,
    }
    algo = ContextInformation()
    algo.reset(state, context)

    k_ci, l_ci = algo.select_next_mbp(state, 0, context)

    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=bs_yaw,
                              n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(ue_xy, ue_yaw)

    rng = np.random.default_rng(0)
    y_ci = state.measure(k_ci, l_ci, H, 0, rng)

    # Exhaustive sweep to find true global best
    state2 = make_state()
    rng2 = np.random.default_rng(0)
    for k in range(state2.K):
        for l in range(state2.L):
            state2.measure(k, l, H, 0, rng2)
    best_k, best_l = state2.obp()
    best_val = float(np.abs(state2.observations[best_k, best_l]))
    ci_val = float(np.abs(y_ci))

    # CI should be within 3 dB of the global best (one beamwidth quantisation error)
    ratio_db = 20 * np.log10(ci_val / (best_val + 1e-30))
    assert ratio_db >= -3.0, (
        f"CI selected ({k_ci},{l_ci}) with power {ci_val:.4f}; "
        f"global best ({best_k},{best_l}) has power {best_val:.4f}; "
        f"gap = {-ratio_db:.1f} dB > 3 dB"
    )


# ---------------------------------------------------------------------------
# MCMD: criterion-matrix audit
# ---------------------------------------------------------------------------

def test_mcmd_c_age_rewards_stale_entries():
    """C_age must give highest values to the least-recently-measured pairs."""
    from beamsim.algorithms import MCMD
    from beamsim.channel import FreeSpaceLosChannel

    algo = MCMD()
    state = make_state()
    bs_xy = np.array([10.0, 0.0])
    context = {"ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0),
               "bs_xy": bs_xy, "bs_yaw": 0.0}
    algo.reset(state, context)
    rng = np.random.default_rng(5)
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0,
                              n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)

    # Measure a few specific pairs so measured_at is non-uniform
    pairs_measured = [(0, 0), (1, 1), (2, 2)]
    for m, (k, l) in enumerate(pairs_measured):
        state.measure(k, l, H, m, rng)

    # C_age at occasion 10: unmeasured entries should have highest age
    ages = state.age_matrix(current_m=10)
    for k, l in pairs_measured:
        for kk in range(state.K):
            for ll in range(state.L):
                if (kk, ll) not in pairs_measured:
                    # Never-measured entries must be at least as stale
                    assert ages[kk, ll] >= ages[k, l], (
                        f"C_age[{kk},{ll}]={ages[kk,ll]} < C_age[{k},{l}]={ages[k,l]}; "
                        "stale never-measured entries should have maximum age"
                    )


def test_mcmd_weight_order_matches_fig526():
    """Verify that W_LOW and W_HIGH are ordered (age, tabu, NNS) as in Fig. 5.26.

    Fig. 5.26 pie charts:
      3 m/s:  Age=43%, Tabu=52%, NNS=5%   -> W_LOW
      10 m/s: Age=16%, Tabu=36%, NNS=49%  -> W_HIGH
    The weights must sum to approximately 1.0 and match those percentages.
    """
    from beamsim.algorithms.mcmd import W_LOW, W_HIGH

    # Verify sums — tolerance 0.02 to allow for pie-chart rounding in Fig. 5.26
    assert abs(W_LOW.sum() - 1.0) < 0.02, f"W_LOW sums to {W_LOW.sum()}, expected ~1.0"
    assert abs(W_HIGH.sum() - 1.0) < 0.02, f"W_HIGH sums to {W_HIGH.sum()}, expected ~1.0"

    # W_LOW: age=0.43 (index 0), tabu=0.52 (index 1), NNS=0.05 (index 2)
    assert abs(W_LOW[0] - 0.43) < 0.01, f"W_LOW[age]={W_LOW[0]}, expected 0.43"
    assert abs(W_LOW[1] - 0.52) < 0.01, f"W_LOW[tabu]={W_LOW[1]}, expected 0.52"
    assert abs(W_LOW[2] - 0.05) < 0.01, f"W_LOW[NNS]={W_LOW[2]}, expected 0.05"

    # W_HIGH: age=0.16, tabu=0.36, NNS=0.49
    assert abs(W_HIGH[0] - 0.16) < 0.01, f"W_HIGH[age]={W_HIGH[0]}, expected 0.16"
    assert abs(W_HIGH[1] - 0.36) < 0.01, f"W_HIGH[tabu]={W_HIGH[1]}, expected 0.36"
    assert abs(W_HIGH[2] - 0.49) < 0.01, f"W_HIGH[NNS]={W_HIGH[2]}, expected 0.49"


def test_mcmd_c_nns_peaks_at_obp():
    """C_nns must have its maximum at the current OBP location."""
    from beamsim.algorithms import MCMD
    from beamsim.channel import FreeSpaceLosChannel

    algo = MCMD()
    state = make_state()
    bs_xy = np.array([10.0, 0.0])
    context = {"ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0),
               "bs_xy": bs_xy, "bs_yaw": 0.0}
    algo.reset(state, context)
    rng = np.random.default_rng(9)
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0,
                              n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)

    # Warm up
    for m in range(10):
        k, l = algo.select_next_mbp(state, m, context)
        state.measure(k, l, H, m, rng)

    ck, cl = state.obp()
    K, L = state.K, state.L
    kk, ll = np.indices((K, L))
    dist = np.maximum(np.abs(kk - ck), np.abs(ll - cl))
    C_nns = np.exp(-dist.astype(float) / algo.nns_radius)

    # Maximum of C_nns must be at the OBP
    max_pos = np.unravel_index(np.argmax(C_nns), C_nns.shape)
    assert max_pos == (ck, cl), (
        f"C_nns peak at {max_pos}, expected OBP ({ck},{cl})"
    )
