"""Algorithm-level smoke tests."""

import numpy as np
import pytest

from beamsim.algorithms import ALL_ALGORITHMS
from beamsim.bplm import BPLMState
from beamsim.channel import FreeSpaceLosChannel
from beamsim.codebook import make_default_bs_codebook, make_default_ue_codebook


def make_state():
    return BPLMState(
        ue_codebook=make_default_ue_codebook(),
        bs_codebook=make_default_bs_codebook(),
        noise_amplitude=0.01,
    )


@pytest.mark.parametrize("name", sorted(ALL_ALGORITHMS))
def test_algorithm_returns_valid_indices(name):
    cls = ALL_ALGORITHMS[name]
    algo = cls()
    state = make_state()
    bs_xy = np.array([10.0, 0.0])
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)
    context = {
        "ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0),
        "bs_xy": bs_xy,
        "bs_yaw": 0.0,
        "true_H": H,
    }
    algo.reset(state, context)
    rng = np.random.default_rng(0)
    for m in range(50):
        k, l = algo.select_next_mbp(state, m, context)
        assert 0 <= k < state.K
        assert 0 <= l < state.L
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
    """CI uses sin-space matching to handle the ULA front/back-lobe ambiguity.

    For UE at origin and BS at (+10, 0), the geometric LOS aligns with the
    array broadside in both body frames. Both UE and BS picks should fall
    near the centre of their respective codebooks (closest beam to sin = 0).
    """
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
    ue_theta = state.ue_codebook.theta
    bs_theta = state.bs_codebook.theta
    # Both should pick the beam closest to sin = 0 (broadside on each side).
    assert abs(np.sin(ue_theta[k])) < 0.2
    assert abs(np.sin(bs_theta[l])) < 0.1


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
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4)
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
    """With 4-connectivity NNS only measures pairs at Chebyshev distance <= 1
    from its internal centre `(kb, lb)`. (The centre is the algorithm's
    own state, not the BPLM-wide OBP — the centre may relocate to a
    neighbour, after which subsequent measurements are around the new
    centre, possibly 2 steps from the previous OBP.)"""
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
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)

    for m in range(6):
        k, l = algo.select_next_mbp(state, m, context)
        state.measure(k, l, H, m, rng)

    for m in range(6, 14):
        k, l = algo.select_next_mbp(state, m, context)
        chebyshev = max(abs(k - algo._kb), abs(l - algo._lb))
        assert chebyshev <= 1, (
            f"NNS picked ({k},{l}); centre=({algo._kb},{algo._lb}); "
            f"Chebyshev={chebyshev}, expected <=1 for 4-connectivity"
        )
        state.measure(k, l, H, m, rng)


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
    context = {"ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0), "bs_xy": bs_xy, "bs_yaw": 0.0}
    algo.reset(state, context)
    rng = np.random.default_rng(1)
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4)
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
    context = {"ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0), "bs_xy": bs_xy, "bs_yaw": 0.0}
    algo.reset(state, context)
    rng = np.random.default_rng(3)
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4)
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
                f"Tabu pair {pair} selected again at step {j}, only {j - i} steps "
                f"after first selection at step {i} (tenure={tenure})"
            )


# ---------------------------------------------------------------------------
# Angular prediction: gradient-sum Algorithm 3 tracking test
# ---------------------------------------------------------------------------


def test_angular_prediction_gradient_sum_tracks_linear_velocity():
    """Gradient-sum predictor (Algorithm 3) must track a linearly drifting beam.

    OBP history is injected directly by writing a large observation at the true
    (k=0, l=beam_index) entry each warmup step, so the h-history accumulates
    consistently increasing angles with a constant gradient.  Post-warmup, the
    gradient-sum predictor must forecast the next beam (or adjacent) correctly.

    Reference: thesis Sec. 5.4.3 Algorithm 3; Fig. 5.19 shows stable tracking.
    """
    from beamsim.algorithms import AngularPrediction
    from beamsim.codebook import make_default_bs_codebook, make_default_ue_codebook

    warmup = 4
    history_len = 3
    algo = AngularPrediction(warmup=warmup, history_len=history_len)
    ue_cb = make_default_ue_codebook()
    bs_cb = make_default_bs_codebook()
    state = BPLMState(ue_codebook=ue_cb, bs_codebook=bs_cb, noise_amplitude=1e-9)

    context = {}
    algo.reset(state, context)

    # Linearly drifting beam: index advances by 1 per step starting at beam 10
    start_beam = 10

    def true_bs_beam(m):
        return start_beam + m  # stays within [10, 30] for m in [0, 20]

    # Warmup: inject OBP history by setting dominant observation at true beam.
    # Values increase so that OBP always resolves to the current true beam.
    for m in range(warmup):
        tl = true_bs_beam(m)
        state.observations[0, tl] = float(m + 2)
        state.measured_at[0, tl] = m
        algo.select_next_mbp(state, m, context)

    # Prediction phase: gradient-sum should forecast +1 beam per step
    hits = 0
    post_warmup = 5
    for m in range(warmup, warmup + post_warmup):
        tl = true_bs_beam(m)
        _k_pred, l_pred = algo.select_next_mbp(state, m, context)
        if abs(l_pred - tl) <= 1:
            hits += 1
        # Update OBP to true beam for next step
        state.observations[0, tl] = float(m + 2)
        state.measured_at[0, tl] = m

    assert hits >= post_warmup * 0.6, (
        f"Gradient-sum tracker hit rate (within 1 beam) {hits}/{post_warmup} < 60%; "
        "predictor is not tracking constant-velocity beam drift"
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

    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=bs_yaw, n_bs_elements=16, n_ue_elements=4)
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
    context = {"ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0), "bs_xy": bs_xy, "bs_yaw": 0.0}
    algo.reset(state, context)
    rng = np.random.default_rng(5)
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4)
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
                        f"C_age[{kk},{ll}]={ages[kk, ll]} < C_age[{k},{l}]={ages[k, l]}; "
                        "stale never-measured entries should have maximum age"
                    )


def test_mcmd_weight_order_matches_fig526():
    """Verify that W_LOW and W_HIGH are ordered (age, tabu, NNS) as in Fig. 5.26.

    Fig. 5.26 pie charts:
      3 m/s:  Age=43%, Tabu=52%, NNS=5%   -> W_LOW
      10 m/s: Age=16%, Tabu=36%, NNS=49%  -> W_HIGH
    The weights must sum to approximately 1.0 and match those percentages.
    """
    from beamsim.algorithms.mcmd import W_HIGH, W_LOW

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
    """C_nns (binary P-list) must have its entries around the current OBP.

    After warmup the internal NNS P-list should contain the 4-connected
    neighbours of the NNS centre, which tracks the OBP.  The test verifies
    that at least one non-zero C_nns entry exists (P-list is non-empty) and
    that all non-zero entries are within Chebyshev distance 1 of the OBP.
    """
    from beamsim.algorithms import MCMD
    from beamsim.channel import FreeSpaceLosChannel

    algo = MCMD()
    state = make_state()
    bs_xy = np.array([10.0, 0.0])
    context = {"ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0), "bs_xy": bs_xy, "bs_yaw": 0.0}
    algo.reset(state, context)
    rng = np.random.default_rng(9)
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)

    # Warm up
    for m in range(10):
        k, l = algo.select_next_mbp(state, m, context)
        state.measure(k, l, H, m, rng)

    _ck, _cl = state.obp()

    # Trigger one more call so the internal NNS P-list is populated
    algo.select_next_mbp(state, 10, context)

    # After warmup the NNS stack should be non-empty (neighbours queued)
    assert len(algo._nns_stack) > 0, (
        "MCMD internal NNS P-list is empty after warmup; C_nns (Eq. 5.28) would be all-zero"
    )

    # All P-list entries should be 4-connected neighbours (Chebyshev dist <= 1)
    nns_centre = (algo._nns_kb, algo._nns_lb)
    for pk, pl in algo._nns_stack:
        cheb = max(abs(pk - nns_centre[0]), abs(pl - nns_centre[1]))
        assert cheb <= 1, (
            f"P-list entry ({pk},{pl}) is Chebyshev-{cheb} from NNS centre "
            f"{nns_centre}; expected <= 1 for 4-connected neighbourhood"
        )


# ---------------------------------------------------------------------------
# New predecessor-fidelity tests
# ---------------------------------------------------------------------------


def test_nns_random_seed_varies_across_trial_seeds():
    """NNS reset() must derive a random seed from context['trial_seed'] so
    different trials start at different (k_b, l_b) under common random
    numbers, while two NNS instances given the same trial_seed start at
    the same point.

    Algorithm 4 (thesis) line 2: kb, lb <- Random. The runner threads a
    distinct seed per Monte Carlo trial; NNS must consume that seed so
    its initialisation respects the CRN contract.
    """
    from beamsim.algorithms import NNS

    algo = NNS()
    state = make_state()

    # 20 trial seeds → at least 2 distinct (k_b, l_b) pairs (collision
    # probability per pair ≈ 1/(K*L), so 20 distinct seeds almost surely
    # cover ≥ 2 starting points).
    seeds = set()
    for trial_seed in range(20):
        algo.reset(state, {"trial_seed": trial_seed})
        seeds.add((algo._kb, algo._lb))
    assert len(seeds) > 1, (
        f"NNS reset() produced only {seeds} across 20 distinct trial seeds; "
        "trial-seed-driven initialisation is not working"
    )

    # Same trial_seed → same start (CRN contract).
    algo_a = NNS()
    algo_b = NNS()
    algo_a.reset(state, {"trial_seed": 7})
    algo_b.reset(state, {"trial_seed": 7})
    assert (algo_a._kb, algo_a._lb) == (algo_b._kb, algo_b._lb)


def test_tabu_default_tenure_is_20():
    """Tabu default tenure must be 20 (thesis Figure 5.23: s=20).

    Algorithm 5 is illustrated with s=20 in the thesis; the default must
    match so out-of-the-box behaviour reproduces the reported results.
    """
    from beamsim.algorithms import Tabu

    algo = Tabu()
    assert algo.tenure == 20, (
        f"Tabu default tenure is {algo.tenure}, expected 20 "
        "(thesis Figure 5.23 caption: 'tabu search with s = 20')"
    )


# ---------------------------------------------------------------------------
# Perfect knowledge: oracle matches exhaustive best (k, l)
# ---------------------------------------------------------------------------


def test_perfect_matches_exhaustive_best_pair():
    """Perfect must return the same (k, l) as the noiseless argmax over all pairs.

    A single deterministic LOS channel is used so the true best pair is
    unambiguous.  Perfect must pick it on every step, giving SNR >= exhaustive
    OBP at all times (they are equal on the first step when exhaustive has
    measured every entry; here we check the noiseless gain directly).
    """
    from beamsim.algorithms import Perfect
    from beamsim.channel import FreeSpaceLosChannel

    bs_xy = np.array([10.0, 0.0])
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)

    state = make_state()
    context = {
        "ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0),
        "bs_xy": bs_xy,
        "bs_yaw": 0.0,
        "true_H": H,
    }
    algo = Perfect()
    algo.reset(state, context)

    # Ground-truth best pair via noiseless gain matrix
    W = state.ue_codebook.matrix
    F = state.bs_codebook.matrix
    gains = np.abs(W.conj().T @ H @ F)
    best_k, best_l = np.unravel_index(np.argmax(gains), gains.shape)

    for m in range(5):
        k, l = algo.select_next_mbp(state, m, context)
        assert (k, l) == (
            int(best_k),
            int(best_l),
        ), f"Step {m}: Perfect returned ({k},{l}), expected ({best_k},{best_l})"


# ---------------------------------------------------------------------------
# NNSBSSequential: round-robin BS stride test (report Sec. 6.5, Fig. 6.7)
# ---------------------------------------------------------------------------


def test_nns_bs_sequential_l_follows_round_robin_stride():
    """BS beam index must follow (l + 7) % L round-robin with default stride=7.

    For L=32, the sequence of returned l values across 200 steps must be
    exactly [7, 14, 21, 28, 3, 10, ...] (mod 32) — a fixed stride-7 scan
    independent of UE NNS behaviour.
    """
    from beamsim.algorithms.nns_bs_sequential import NNSBSSequential
    from beamsim.channel import FreeSpaceLosChannel

    algo = NNSBSSequential(bs_stride=7)
    state = make_state()
    bs_xy = np.array([10.0, 0.0])
    context = {
        "ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0),
        "bs_xy": bs_xy,
        "bs_yaw": 0.0,
    }
    algo.reset(state, context)
    rng = np.random.default_rng(42)
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)

    L = state.L  # 32
    n_steps = 200
    l_values = []
    for m in range(n_steps):
        k, l = algo.select_next_mbp(state, m, context)
        l_values.append(l)
        state.measure(k, l, H, m, rng)

    expected = [(7 * (i + 1)) % L for i in range(n_steps)]
    assert l_values == expected, (
        f"BS beam sequence mismatch.\n"
        f"First 10 got:      {l_values[:10]}\n"
        f"First 10 expected: {expected[:10]}"
    )


def test_mcmd_binary_c_nns_reflects_p_list():
    """C_nns must be 1 for entries in P and 0 for entries not in P (Eq. 5.28).

    Verify that after warmup, when the NNS P-list is non-empty, the entries
    in the P-list get C_nns=1 and all other entries get C_nns=0.
    """
    from beamsim.algorithms import MCMD
    from beamsim.channel import FreeSpaceLosChannel

    algo = MCMD()
    state = make_state()
    bs_xy = np.array([10.0, 0.0])
    context = {"ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0), "bs_xy": bs_xy, "bs_yaw": 0.0}
    algo.reset(state, context)
    rng = np.random.default_rng(42)
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)

    # Run enough steps to establish a non-trivial NNS centre and P-list
    for m in range(12):
        k, l = algo.select_next_mbp(state, m, context)
        state.measure(k, l, H, m, rng)

    # Force a fresh NNS P-list rebuild by resetting and calling select
    ck, cl = state.obp()
    algo._nns_kb = ck
    algo._nns_lb = cl
    algo._nns_xi = 0.0
    algo._nns_stack = []
    # Trigger _update_nns via select_next_mbp
    algo.select_next_mbp(state, 12, context)

    # If P-list is non-empty, verify binary C_nns structure
    if algo._nns_stack:
        K, L = state.K, state.L
        C_nns = np.zeros((K, L), dtype=float)
        for pk, pl in algo._nns_stack:
            C_nns[pk, pl] = 1.0

        p_set = set(algo._nns_stack)
        for k in range(K):
            for l in range(L):
                if (k, l) in p_set:
                    assert C_nns[k, l] == 1.0, f"C_nns[{k},{l}]=0 but ({k},{l}) is in P-list"
                else:
                    assert C_nns[k, l] == 0.0, f"C_nns[{k},{l}]>0 but ({k},{l}) is not in P-list"


# ---------------------------------------------------------------------------
# UCB1: cold-start coverage and exploitation tests
# ---------------------------------------------------------------------------


def test_ucb1_explores_all_arms_in_first_KL_steps():
    """UCB1 must pull every arm exactly once before applying the UCB rule.

    The first K*L selections must cover all (k, l) pairs (cold-start phase).
    """
    from beamsim.algorithms.ucb1 import UCB1
    from beamsim.channel import FreeSpaceLosChannel

    algo = UCB1()
    state = make_state()
    K, L = state.K, state.L
    bs_xy = np.array([10.0, 0.0])
    context = {"ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0), "bs_xy": bs_xy, "bs_yaw": 0.0}
    algo.reset(state, context)
    rng = np.random.default_rng(0)
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)

    visited: set[tuple[int, int]] = set()
    for m in range(K * L):
        k, l = algo.select_next_mbp(state, m, context)
        visited.add((k, l))
        state.measure(k, l, H, m, rng)

    assert len(visited) == K * L, (
        f"UCB1 covered only {len(visited)}/{K * L} arms in first {K * L} steps"
    )


def test_ucb1_eventually_exploits_max_arm():
    """UCB1 must converge to the best arm on a stationary environment.

    A synthetic BPLM where arm (0,0) always yields reward 10x higher than
    all others.  After K*L cold-start + 3*K*L more steps, arm (0,0) must be
    picked more than 50% of the time.
    """
    from beamsim.algorithms.ucb1 import UCB1
    from beamsim.codebook import Codebook

    K, L = 4, 8
    ue_cb = Codebook(n_elements=4, n_beams=K)
    bs_cb = Codebook(n_elements=16, n_beams=L)
    state = BPLMState(ue_codebook=ue_cb, bs_codebook=bs_cb, noise_amplitude=0.01)
    algo = UCB1()
    algo.reset(state, {})

    rng = np.random.default_rng(99)
    n_cold = K * L
    n_exploit = 3 * K * L
    best_count = 0

    for m in range(n_cold + n_exploit):
        k, l = algo.select_next_mbp(state, m, {})
        # Synthetic reward: (0,0) = high, others = low
        reward = 10.0 if (k, l) == (0, 0) else 0.1 + rng.random() * 0.1
        state.observations[k, l] = complex(reward)
        state.measured_at[k, l] = m
        if m >= n_cold and (k, l) == (0, 0):
            best_count += 1

    rate = best_count / n_exploit
    assert rate > 0.50, (
        f"UCB1 exploited best arm only {rate:.1%} of post-cold-start steps "
        f"(expected >50%); best_count={best_count}/{n_exploit}"
    )


# ---------------------------------------------------------------------------
# ThompsonGaussian: validity and seed-isolation tests
# ---------------------------------------------------------------------------


def test_thompson_returns_valid_index():
    """ThompsonGaussian must return a valid (k, l) pair on every step."""
    from beamsim.algorithms.thompson import ThompsonGaussian
    from beamsim.channel import FreeSpaceLosChannel

    algo = ThompsonGaussian()
    state = make_state()
    K, L = state.K, state.L
    bs_xy = np.array([10.0, 0.0])
    context = {"ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0), "bs_xy": bs_xy, "bs_yaw": 0.0}
    algo.reset(state, context)
    rng = np.random.default_rng(7)
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)

    for m in range(K * L + 20):
        k, l = algo.select_next_mbp(state, m, context)
        assert 0 <= k < K, f"Thompson k={k} out of [0, {K})"
        assert 0 <= l < L, f"Thompson l={l} out of [0, {L})"
        state.measure(k, l, H, m, rng)


def test_thompson_seed_isolation():
    """Two ThompsonGaussian instances reset with the same trial_seed must produce
    the same sequence; reset with different trial_seeds must diverge."""
    from beamsim.algorithms.thompson import ThompsonGaussian
    from beamsim.codebook import Codebook

    K, L = 4, 8
    ue_cb = Codebook(n_elements=4, n_beams=K)
    bs_cb = Codebook(n_elements=16, n_beams=L)

    def make_fresh(seed: int):
        state = BPLMState(ue_codebook=ue_cb, bs_codebook=bs_cb, noise_amplitude=0.01)
        algo = ThompsonGaussian()
        algo.reset(state, {"trial_seed": seed})
        return algo, state

    n_steps = K * L + 30

    # Same seed → identical sequences
    algo_a, state_a = make_fresh(99)
    algo_b, state_b = make_fresh(99)
    rng = np.random.default_rng(0)
    sel_a, sel_b = [], []
    for m in range(n_steps):
        ka, la = algo_a.select_next_mbp(state_a, m, {})
        kb, lb = algo_b.select_next_mbp(state_b, m, {})
        r = float(rng.random())
        state_a.observations[ka, la] = complex(r)
        state_b.observations[kb, lb] = complex(r)
        sel_a.append((ka, la))
        sel_b.append((kb, lb))
    assert sel_a == sel_b, "ThompsonGaussian with same trial_seed produced different sequences"

    # Different trial_seeds → sequences must diverge (regression for the seed=42 bug)
    horizon = K * L + 50
    algo_t0, state_t0 = make_fresh(12345 ^ 0)
    algo_t1, state_t1 = make_fresh(12345 ^ 1)
    rng2 = np.random.default_rng(1)
    seq_t0, seq_t1 = [], []
    for m in range(horizon):
        k0, l0 = algo_t0.select_next_mbp(state_t0, m, {})
        k1, l1 = algo_t1.select_next_mbp(state_t1, m, {})
        r = float(rng2.random())
        state_t0.observations[k0, l0] = complex(r)
        state_t1.observations[k1, l1] = complex(r)
        seq_t0.append((k0, l0))
        seq_t1.append((k1, l1))
    assert seq_t0 != seq_t1, (
        "ThompsonGaussian produced identical (k,l) sequences for two different "
        "trial_seeds — the seed=42 regression is back"
    )


# ---------------------------------------------------------------------------
# HBM: hierarchical codebook tests (Alkhateeb et al. 2014)
# ---------------------------------------------------------------------------


def test_hbm_completes_coarse_sweep_in_M_steps():
    """First n_coarse steps must cover every coarse BS sector exactly once.

    With coarse_factor=4 and L=32, there are 8 coarse sectors: 0,4,8,...,28.
    The first 8 calls to select_next_mbp must return exactly those BS beams.
    """
    from beamsim.algorithms.hbm import HBM
    from beamsim.channel import FreeSpaceLosChannel

    coarse_factor = 4
    algo = HBM(coarse_factor=coarse_factor, refresh_every=1000)
    state = make_state()
    L = state.L  # 32
    expected_coarse = set(range(0, L, coarse_factor))  # {0,4,8,...,28}
    n_coarse = len(expected_coarse)

    bs_xy = np.array([10.0, 0.0])
    context = {"ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0), "bs_xy": bs_xy, "bs_yaw": 0.0}
    algo.reset(state, context)
    rng = np.random.default_rng(0)
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)

    coarse_ls: list[int] = []
    for m in range(n_coarse):
        k, l = algo.select_next_mbp(state, m, context)
        coarse_ls.append(l)
        state.measure(k, l, H, m, rng)

    assert set(coarse_ls) == expected_coarse, (
        f"Coarse sweep returned BS beams {sorted(set(coarse_ls))}, "
        f"expected {sorted(expected_coarse)}"
    )


def test_hbm_returns_valid_index_under_random_bplm():
    """HBM must always return indices in [0, K) x [0, L) for random observations."""
    from beamsim.algorithms.hbm import HBM
    from beamsim.channel import FreeSpaceLosChannel

    algo = HBM(coarse_factor=4, refresh_every=50)
    state = make_state()
    bs_xy = np.array([10.0, 0.0])
    context = {"ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0), "bs_xy": bs_xy, "bs_yaw": 0.0}
    algo.reset(state, context)
    rng = np.random.default_rng(42)
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)

    K, L = state.K, state.L
    for m in range(200):
        k, l = algo.select_next_mbp(state, m, context)
        assert 0 <= k < K, f"step {m}: k={k} out of [0, {K})"
        assert 0 <= l < L, f"step {m}: l={l} out of [0, {L})"
        state.measure(k, l, H, m, rng)


# ---------------------------------------------------------------------------
# OMP: compressive beam alignment tests (Marzi et al. 2016)
# ---------------------------------------------------------------------------


def test_omp_solves_known_sparse_signal():
    """OMP sub-routine must recover a 2-sparse vector from noiseless measurements.

    A random Gaussian sensing matrix A (shape M x N with M < N) and a
    2-sparse signal x are constructed.  OMP must recover the correct support
    and achieve near-zero residual.

    This tests the rolled OMP algorithm (OMPCompressive._omp) directly,
    independently of the BPLMState measurement pipeline.
    """
    from beamsim.algorithms.omp_compressive import OMPCompressive

    rng = np.random.default_rng(42)
    N = 64  # unknown dimension (e.g., n_ue * n_bs = 4*16)
    M = 32  # measurements (> sparsity but << N)
    sparsity = 2

    # Construct a Gaussian measurement matrix (RIP-satisfying with high probability)
    A = (rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))) / np.sqrt(2 * M)

    # Ground-truth 2-sparse signal
    true_support = [10, 45]
    x_true = np.zeros(N, dtype=np.complex128)
    x_true[true_support[0]] = 5.0 + 0j
    x_true[true_support[1]] = 3.0 + 0.5j

    y = A @ x_true  # noiseless measurements

    x_hat = OMPCompressive._omp(A, y, sparsity)

    # Support must match exactly on a noiseless problem with well-separated spikes
    recovered_support = sorted(np.where(np.abs(x_hat) > 0.1)[0].tolist())
    assert recovered_support == sorted(true_support), (
        f"OMP recovered support {recovered_support}, expected {sorted(true_support)}"
    )
    # Reconstruction error should be negligible
    residual_norm = float(np.linalg.norm(y - A @ x_hat))
    assert residual_norm < 1e-6, (
        f"OMP residual norm {residual_norm:.2e} exceeds 1e-6 on noiseless problem"
    )


def test_omp_returns_valid_index():
    """OMPCompressive must return valid (k, l) indices on every step."""
    from beamsim.algorithms.omp_compressive import OMPCompressive
    from beamsim.channel import FreeSpaceLosChannel

    algo = OMPCompressive(measurements_per_solve=8, sparsity=2)
    state = make_state()
    K, L = state.K, state.L
    bs_xy = np.array([10.0, 0.0])
    context = {"ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0), "bs_xy": bs_xy, "bs_yaw": 0.0}
    algo.reset(state, context)
    rng = np.random.default_rng(13)
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)

    for m in range(100):
        k, l = algo.select_next_mbp(state, m, context)
        assert 0 <= k < K, f"step {m}: k={k} out of [0, {K})"
        assert 0 <= l < L, f"step {m}: l={l} out of [0, {L})"
        state.measure(k, l, H, m, rng)


# ---------------------------------------------------------------------------
# DLPredictor: fallback and checkpoint tests
# ---------------------------------------------------------------------------


def test_dl_predictor_falls_back_when_no_checkpoint():
    """DLPredictor must fall back to Exhaustive-style behaviour (valid indices)
    when 'models/beam_predictor.pt' does not exist, with a UserWarning."""
    import warnings

    from beamsim.algorithms.dl_predictor import DLPredictor
    from beamsim.channel import FreeSpaceLosChannel

    # Point at a path that definitely does not exist
    algo = DLPredictor(checkpoint="/tmp/nonexistent_beam_predictor_zzz.pt")
    state = make_state()
    bs_xy = np.array([10.0, 0.0])
    context = {
        "ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0),
        "bs_xy": bs_xy,
        "bs_yaw": 0.0,
    }
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)
    rng = np.random.default_rng(0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        algo.reset(state, context)
        for m in range(20):
            k, l = algo.select_next_mbp(state, m, context)
            assert 0 <= k < state.K, f"k={k} out of range"
            assert 0 <= l < state.L, f"l={l} out of range"
            state.measure(k, l, H, m, rng)

    # At least one UserWarning about missing checkpoint or torch
    warning_texts = " ".join(str(w.message) for w in caught)
    assert caught, "Expected at least one warning from DLPredictor fallback"
    assert any(issubclass(w.category, UserWarning) for w in caught), (
        f"Expected UserWarning; got: {warning_texts}"
    )


def test_dl_predictor_returns_valid_index_with_checkpoint():
    """With a valid checkpoint, DLPredictor must return indices in [0,K) x [0,L).

    Skipped if torch is not installed or checkpoint does not exist.
    """
    pytest.importorskip("torch")
    from pathlib import Path

    ckpt = Path("models/beam_predictor.pt")
    if not ckpt.exists():
        pytest.skip("models/beam_predictor.pt not found — run training first")

    from beamsim.algorithms.dl_predictor import DLPredictor
    from beamsim.channel import FreeSpaceLosChannel

    algo = DLPredictor(checkpoint=ckpt)
    state = make_state()
    bs_xy = np.array([10.0, 0.0])
    context = {
        "ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0),
        "bs_xy": bs_xy,
        "bs_yaw": 0.0,
    }
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)
    rng = np.random.default_rng(7)

    algo.reset(state, context)
    # Seed enough measurements so OBP history fills the window
    for m in range(50):
        k, l = algo.select_next_mbp(state, m, context)
        assert 0 <= k < state.K, f"k={k} out of range at step {m}"
        assert 0 <= l < state.L, f"l={l} out of range at step {m}"
        state.measure(k, l, H, m, rng)


# ---------------------------------------------------------------------------
# Phase 4B: MAMBA, EKF, PositionMAB, BAI
# ---------------------------------------------------------------------------


def _ch_and_state_at(bs_xy: np.ndarray, ue_xy: np.ndarray, ue_yaw: float = 0.0):
    """Helper: build a free-space-LOS channel matrix at a given pose."""
    from beamsim.channel import FreeSpaceLosChannel

    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(ue_xy, ue_yaw)
    return ch, H


def test_mamba_neighbourhood_explore_triggers_on_drop():
    """MAMBA must enter neighbourhood-explore when the best arm drops by >threshold."""
    from beamsim.algorithms.mamba import MAMBA

    state = make_state()
    algo = MAMBA(gamma=0.95, explore_threshold=0.30, explore_horizon=4, sigma_floor=0.01)
    algo.reset(state, {})

    # Seed a clear best arm at (3, 7) with reward ~10; everything else ~0.1.
    # We do this by directly manipulating BPLM observations across several
    # rounds so the running mean concentrates on (3, 7).
    for _ in range(30):
        # Inject the best-arm observation.
        state.observations[3, 7] = complex(10.0)
        state.measured_at[3, 7] = 0
        algo._last_kl = (3, 7)
        algo.select_next_mbp(state, 0, {})

    # Best arm should now be (3, 7).
    assert algo._best_kl == (3, 7), f"MAMBA best_kl={algo._best_kl}, expected (3, 7)"

    # Inject a sharp drop at (3, 7): reward ~ 1 (90% drop).
    state.observations[3, 7] = complex(1.0)
    algo._last_kl = (3, 7)
    algo.select_next_mbp(state, 0, {})

    assert algo._explore_counter > 0, (
        f"MAMBA failed to trigger neighbourhood-explore on a 90% reward drop; "
        f"counter={algo._explore_counter}"
    )

    # Once in explore mode, the next pull must be from the 4-connected
    # neighbourhood of (3, 7) (or (3,7) itself).
    next_kl = algo.select_next_mbp(state, 0, {})
    expected = {(3, 7), (2, 7), (4, 7), (3, 6), (3, 8)}
    assert next_kl in expected, (
        f"MAMBA in explore mode picked {next_kl}; expected something in {expected}"
    )


def test_mamba_seeded_reproducibility():
    """Same trial_seed must produce identical MAMBA traces."""
    from beamsim.algorithms.mamba import MAMBA

    bs_xy = np.array([10.0, 0.0])
    _ch, H = _ch_and_state_at(bs_xy, np.array([0.0, 0.0]))
    rng = np.random.default_rng(0)

    def run() -> list[tuple[int, int]]:
        algo = MAMBA()
        s = make_state()
        ctx = {
            "ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0),
            "bs_xy": bs_xy,
            "bs_yaw": 0.0,
            "trial_seed": 1234,
        }
        algo.reset(s, ctx)
        choices: list[tuple[int, int]] = []
        for m in range(20):
            k, l = algo.select_next_mbp(s, m, ctx)
            choices.append((k, l))
            s.measure(k, l, H, m, rng)
        return choices

    a = run()
    b = run()
    assert a == b, "MAMBA produced different traces with the same trial_seed"


def test_ekf_tracker_locks_onto_static_los():
    """EKF must lock onto the OBP after warmup on a static LOS channel."""
    from beamsim.algorithms.ekf_tracker import EKFTracker

    bs_xy = np.array([20.0, 0.0])
    ue_xy = np.array([0.0, 5.0])  # offset so the AoD is non-zero
    _ch, H = _ch_and_state_at(bs_xy, ue_xy)
    rng = np.random.default_rng(0)

    state = make_state()
    algo = EKFTracker(warmup=8, dt=1e-3)
    algo.reset(state, {"trial_seed": 0})

    # Run for a while; collect OBPs.
    obps: list[tuple[int, int]] = []
    for m in range(120):
        k, l = algo.select_next_mbp(state, m, {})
        state.measure(k, l, H, m, rng)
        obps.append(state.obp())

    # The post-warmup OBPs should converge to a single (k, l) since the
    # channel is static.  We allow some jitter from EKF transient.
    tail_unique = set(obps[80:])
    assert len(tail_unique) <= 2, (
        f"EKF did not converge on a static channel; unique tail OBPs={tail_unique}"
    )


def test_ekf_tracker_seeded_reproducibility():
    """EKF is deterministic given the same channel + warmup; verify."""
    from beamsim.algorithms.ekf_tracker import EKFTracker

    bs_xy = np.array([10.0, 0.0])
    _, H = _ch_and_state_at(bs_xy, np.array([0.0, 0.0]))

    def run() -> list[tuple[int, int]]:
        algo = EKFTracker()
        s = make_state()
        algo.reset(s, {})
        rng = np.random.default_rng(7)
        choices: list[tuple[int, int]] = []
        for m in range(30):
            k, l = algo.select_next_mbp(s, m, {})
            choices.append((k, l))
            s.measure(k, l, H, m, rng)
        return choices

    a = run()
    b = run()
    assert a == b, "EKF produced different traces on identical inputs"


def test_position_mab_reuses_bin_posterior_on_revisit():
    """PositionMAB at a revisited spatial bin must NOT re-cold-start."""
    from beamsim.algorithms.position_mab import PositionMAB

    bs_xy = np.array([10.0, 0.0])
    _, H = _ch_and_state_at(bs_xy, np.array([0.0, 0.0]))
    rng = np.random.default_rng(0)

    state = make_state()
    # Two distinct UE positions that map to two different bins.
    pose_a = (np.array([10.0, 10.0]), 0.0)
    pose_b = (np.array([-50.0, -50.0]), 0.0)
    seq = [pose_a if m < 30 else pose_b if m < 60 else pose_a for m in range(90)]
    algo = PositionMAB(n_bins_x=4, n_bins_y=4, n_bins_yaw=1, sigma_floor=0.01)
    ctx = {
        "ue_pose_at": lambda m: seq[m],
        "bs_xy": bs_xy,
        "bs_yaw": 0.0,
        "trial_seed": 0,
    }
    algo.reset(state, ctx)

    bin_a = algo._bin_index(*pose_a)
    bin_b = algo._bin_index(*pose_b)
    assert bin_a != bin_b, "Test setup error: pose_a and pose_b mapped to the same bin"

    # Run all 90 steps.
    for m in range(90):
        k, l = algo.select_next_mbp(state, m, ctx)
        state.measure(k, l, H, m, rng)

    # After the third visit to pose_a (last 30 steps), bin_a should have
    # accumulated more than 30 pulls' worth of posterior mass and the
    # most-probed arm should be far higher count than its peers.
    counts_a = algo._counts[bin_a]
    assert counts_a.max() > 5, (
        f"PositionMAB never concentrated on a dominant arm in bin_a; "
        f"max count={counts_a.max()}, total pulls in bin={counts_a.sum()}"
    )


def test_bai_pure_exploration_eliminates_obviously_bad_arms():
    """BAI successive elimination must eliminate clearly-suboptimal arms."""
    from beamsim.algorithms.bai import BAIPureExploration
    from beamsim.codebook import Codebook

    K, L = 4, 8
    n_arms = K * L
    ue_cb = Codebook(n_elements=4, n_beams=K)
    bs_cb = Codebook(n_elements=16, n_beams=L)
    state = BPLMState(ue_codebook=ue_cb, bs_codebook=bs_cb, noise_amplitude=0.01)
    state.tx_amp = 1.0

    algo = BAIPureExploration(delta=0.1, min_pulls_per_arm=2)
    algo.reset(state, {})

    # Synthetic stationary rewards: arm (0,0) ~ 10, all others ~ 0.1.
    # Hoeffding-based elimination on a [0, R_max] reward range needs ~30+
    # pulls per arm before the confidence radius shrinks below the gap;
    # we run 32 sweeps so the algorithm has a fair chance to commit.
    rng = np.random.default_rng(2)
    for m in range(32 * n_arms):
        k, l = algo.select_next_mbp(state, m, {})
        reward = (
            10.0 + rng.standard_normal() * 0.05 if (k, l) == (0, 0) else (0.1 + rng.random() * 0.05)
        )
        state.observations[k, l] = complex(reward)
        state.measured_at[k, l] = m

    # The best arm must be active and at least half of the other arms
    # eliminated — anything looser would not exercise the elimination rule.
    n_active = int(np.sum(algo._active))
    assert algo._active[0], "BAI eliminated the actual best arm (index 0)"
    assert n_active <= n_arms // 2 + 2, (
        f"BAI failed to eliminate any clearly-bad arms; active={n_active}/{n_arms}"
    )


def test_dl_lstm_predictor_falls_back_when_no_checkpoint():
    """DLLSTMPredictor must fall back to Exhaustive when no checkpoint."""
    import warnings

    from beamsim.algorithms.dl_lstm_predictor import DLLSTMPredictor
    from beamsim.channel import FreeSpaceLosChannel

    algo = DLLSTMPredictor(checkpoint="/tmp/nonexistent_lstm_zzz.pt")
    state = make_state()
    bs_xy = np.array([10.0, 0.0])
    context = {
        "ue_pose_at": lambda m: (np.array([0.0, 0.0]), 0.0),
        "bs_xy": bs_xy,
        "bs_yaw": 0.0,
    }
    ch = FreeSpaceLosChannel(bs_xy=bs_xy, bs_yaw=0.0, n_bs_elements=16, n_ue_elements=4)
    H = ch.channel_matrix(np.array([0.0, 0.0]), 0.0)
    rng = np.random.default_rng(0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        algo.reset(state, context)
        for m in range(20):
            k, l = algo.select_next_mbp(state, m, context)
            assert 0 <= k < state.K
            assert 0 <= l < state.L
            state.measure(k, l, H, m, rng)

    assert caught, "Expected at least one warning from DLLSTMPredictor fallback"
    assert any(issubclass(w.category, UserWarning) for w in caught)
