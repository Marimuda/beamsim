"""Monte Carlo orchestrator with CRN pairing and process-based parallelism.

Design notes
------------
CRN pairing
    Each trial is seeded from ``experiment.seed ^ trial_index``.  Within a
    trial a *root* RNG is split into three independent streams:

    1. ``channel_rng``  — used by ``channel_factory`` (same for every algo).
    2. ``track_rng``    — used by ``track_factory`` (same for every algo).
    3. Per-algo noise RNGs via ``root_rng.spawn(n_algorithms)`` so that every
       algorithm receives an *identical* noise sequence across the (k, l)
       measurements it triggers.  Because algorithms make different numbers of
       measurements on different pairs the noise consumed diverges, but CRN
       guarantees that a fixed (k, l) always receives the same noise sample
       within a trial, which is what matters for fair comparison.

Parallelism
    ``concurrent.futures.ProcessPoolExecutor`` is used.  Each worker receives
    a *serialisable* description of a single trial and returns a ``TrialResult``.
    Factories are callables that must be picklable (plain functions or
    ``functools.partial`` objects work; lambdas do NOT survive pickling across
    processes — use ``functools.partial`` or module-level functions in
    experiment scripts).

Multi-BS handover
    When ``len(experiment.bs_positions) > 1`` each BS gets its own
    ``ChannelRealisation`` (or custom channel object) from
    ``channel_factory(rng, bs_index)``.  The runner tracks the best-BS index
    per occasion by comparing the OBP-SNR across all BS BPLMs.  Algorithms
    also receive ``context["bs_list"]`` so they may implement handover logic.

Storage
    ``save_experiment`` writes a ``npz`` archive; loading with
    ``np.load(path, allow_pickle=True)`` recovers all arrays and object arrays
    for metadata.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from beamsim.algorithms import ALL_ALGORITHMS
from beamsim.bplm import BPLMState
from beamsim.codebook import make_default_bs_codebook, make_default_ue_codebook
from beamsim.geometry import Track

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """Outputs of a single Monte Carlo trial for all algorithms."""

    snr_db: dict[str, NDArray[np.float64]]  # algo -> (n_steps,)
    obp_history: dict[str, NDArray[np.int_]]  # algo -> (n_steps, 2) [k, l]
    selected_bs: dict[str, NDArray[np.int_]] | None  # algo -> (n_steps,); None if single-BS
    seed: int
    snr_db_best: dict[str, NDArray[np.float64]] | None = (
        None  # algo -> (n_steps,); None if single-BS
    )
    snr_oracle: NDArray[np.float64] | None = None  # (n_steps,) true oracle; None if single-BS


@dataclass
class Experiment:
    """Fully-specified Monte Carlo experiment."""

    name: str
    n_steps: int
    dt: float
    n_trials: int
    algorithms: list[str]  # keys in ALL_ALGORITHMS
    bs_positions: list[tuple[float, float]]  # 1 BS or N BS for handover
    bs_yaws: list[float]
    track_factory: Callable[[np.random.Generator], Track]  # per-trial
    channel_factory: Callable[[np.random.Generator, int], object]  # (rng, bs_index) -> channel
    noise_amplitude: float
    tx_amp: float
    seed: int = 12345
    # Codebook factories (optional; defaults to UE=4×8, BS=16×32)
    ue_codebook_factory: Callable[[], object] = field(
        default_factory=lambda: make_default_ue_codebook
    )
    bs_codebook_factory: Callable[[], object] = field(
        default_factory=lambda: make_default_bs_codebook
    )


# ---------------------------------------------------------------------------
# Trial worker (must be a top-level function to be picklable)
# ---------------------------------------------------------------------------


def _run_trial(
    trial_index: int,
    experiment: Experiment,
) -> TrialResult:
    """Execute one Monte Carlo trial for all algorithms with CRN pairing."""

    trial_seed = int(experiment.seed) ^ int(trial_index)
    root_rng = np.random.default_rng(trial_seed)

    # Spawn three independent child streams: track, channels, and then one
    # per algo for noise.  Spawning before any draws keeps the streams
    # independent of algorithm count.
    track_rng, channel_rng, *algo_noise_rngs = root_rng.spawn(2 + len(experiment.algorithms))

    # --- Build shared channel(s) and track -----------------------------------
    n_bs = len(experiment.bs_positions)
    channels = [experiment.channel_factory(channel_rng, b) for b in range(n_bs)]
    track: Track = experiment.track_factory(track_rng)

    # --- Prepare per-algorithm state -----------------------------------------
    ue_cb = experiment.ue_codebook_factory()
    bs_cb = experiment.bs_codebook_factory()
    multi_bs = n_bs > 1

    algo_instances = {}
    bplm_per_algo_per_bs: dict[str, list[BPLMState]] = {}

    for algo_name in experiment.algorithms:
        cls = ALL_ALGORITHMS[algo_name]
        algo_instances[algo_name] = cls()
        # One BPLM per BS per algorithm
        bplm_per_algo_per_bs[algo_name] = [
            _make_bplm(ue_cb, bs_cb, experiment.noise_amplitude, experiment.tx_amp)
            for _ in range(n_bs)
        ]

    # Build context shared across all algorithms (read-only after reset)
    # For multi-BS, use the first BS as the "primary" context entry.
    context = {
        "ue_pose_at": lambda m: (track.positions[m], float(track.orientations[m])),
        "bs_xy": np.array(experiment.bs_positions[0]),
        "bs_yaw": experiment.bs_yaws[0],
        "trial_seed": trial_seed,
    }
    if multi_bs:
        context["bs_list"] = [
            {"bs_xy": np.array(experiment.bs_positions[b]), "bs_yaw": experiment.bs_yaws[b]}
            for b in range(n_bs)
        ]

    # Reset each algorithm
    for algo_name, algo in algo_instances.items():
        primary_bplm = bplm_per_algo_per_bs[algo_name][0]
        algo.reset(primary_bplm, context)

    # --- Output arrays -------------------------------------------------------
    snr_db = {a: np.zeros(experiment.n_steps) for a in experiment.algorithms}
    obp_history = {
        a: np.zeros((experiment.n_steps, 2), dtype=np.int_) for a in experiment.algorithms
    }
    selected_bs_out: dict[str, NDArray[np.int_]] | None = (
        {a: np.zeros(experiment.n_steps, dtype=np.int_) for a in experiment.algorithms}
        if multi_bs
        else None
    )
    # Best-BS noiseless SNR per step (only for multi-BS experiments)
    snr_db_best_out: dict[str, NDArray[np.float64]] | None = (
        {a: np.zeros(experiment.n_steps) for a in experiment.algorithms} if multi_bs else None
    )
    # Codebook oracle: max over all (BS, k, l) of noiseless post-beamforming
    # SNR, i.e. the strongest SNR achievable on the *simulated finite UE×BS
    # codebook* given the same channel realisation. Not Shannon capacity.
    snr_oracle_out: NDArray[np.float64] | None = np.zeros(experiment.n_steps) if multi_bs else None

    # Pre-build full codebook matrices for vectorised oracle computation.
    # W: (K, n_ue_elements), F: (L, n_bs_elements)
    if multi_bs:
        _W = ue_cb.matrix.T.conj()  # type: ignore[attr-defined]  # Codebook factory returns object; .matrix exists at runtime
        _F = bs_cb.matrix.T  # type: ignore[attr-defined]  # same
        _sigma_sq = experiment.noise_amplitude**2
        _tx_amp_sq = experiment.tx_amp**2

    # --- Main loop -----------------------------------------------------------
    for m in range(experiment.n_steps):
        ue_xy = track.positions[m]
        ue_yaw = float(track.orientations[m])
        time_s = m * experiment.dt

        # Compute channel matrix for each BS (shared across algos — CRN).
        H_per_bs = [
            channels[b].channel_matrix(ue_xy, ue_yaw, time_s)  # type: ignore[attr-defined]  # channel factory returns object; .channel_matrix exists at runtime
            for b in range(n_bs)
        ]

        context["true_H"] = H_per_bs[0]

        # Oracle: max noiseless SNR over all (BS, k, l) — algo-independent.
        if multi_bs and snr_oracle_out is not None:
            # H_stack: (n_bs, Nue, Nbs)
            H_stack = np.stack(H_per_bs, axis=0)
            # gains[b, k, l] = |w_k^H @ H_b @ f_l|^2
            # _W[k,i] = w_k.conj()[i], _F[l,j] = f_l[j]
            # einsum: sum_{i,j} _W[k,i] * H[b,i,j] * _F[l,j] = w_k^H H_b f_l
            gains = np.abs(np.einsum("ki,bij,lj->bkl", _W, H_stack, _F)) ** 2
            oracle_lin = float(gains.max()) * _tx_amp_sq / _sigma_sq
            snr_oracle_out[m] = 10.0 * np.log10(max(oracle_lin, 1e-10))

        for a_idx, algo_name in enumerate(experiment.algorithms):
            noise_rng = algo_noise_rngs[a_idx]
            algo = algo_instances[algo_name]
            bplms = bplm_per_algo_per_bs[algo_name]

            # Select MBP using the primary BPLM (BS 0)
            k, l = algo.select_next_mbp(bplms[0], m, context)

            # Measure into primary BPLM
            bplms[0].measure(k, l, H_per_bs[0], m, noise_rng)

            # For multi-BS: measure the same (k, l) pair into the other BPLMs
            # using the *same* noise stream so CRN holds across BS comparisons.
            for b in range(1, n_bs):
                bplms[b].measure(k, l, H_per_bs[b], m, noise_rng)

            # OBP from stale matrix (argmax of |Y_tilde|)
            k_obp, l_obp = bplms[0].obp()
            obp_history[algo_name][m] = [k_obp, l_obp]

            # Compute output SNR for each BS using the OBP codewords
            bs_snr_lin = _obp_snr_linear(
                k_obp, l_obp, H_per_bs, ue_cb, bs_cb, experiment.tx_amp, experiment.noise_amplitude
            )

            if multi_bs:
                best_b = int(np.argmax(bs_snr_lin))
                selected_bs_out[algo_name][m] = best_b  # type: ignore[index]
                snr_lin = bs_snr_lin[best_b]
                # Record oracle best-BS SNR (max over all BSs)
                snr_db_best_out[algo_name][m] = 10.0 * np.log10(  # type: ignore[index]
                    max(float(bs_snr_lin.max()), 1e-10)
                )
            else:
                snr_lin = bs_snr_lin[0]

            snr_db[algo_name][m] = 10.0 * np.log10(max(snr_lin, 1e-10))

    return TrialResult(
        snr_db=snr_db,
        obp_history=obp_history,
        selected_bs=selected_bs_out,
        seed=trial_seed,
        snr_db_best=snr_db_best_out,
        snr_oracle=snr_oracle_out,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bplm(ue_cb: object, bs_cb: object, noise_amplitude: float, tx_amp: float) -> BPLMState:
    state = BPLMState(
        ue_codebook=ue_cb,  # type: ignore[arg-type]  # codebook factory returns object; Codebook at runtime
        bs_codebook=bs_cb,  # type: ignore[arg-type]  # same
        noise_amplitude=noise_amplitude,
    )
    state.tx_amp = tx_amp
    return state


def _obp_snr_linear(
    k_obp: int,
    l_obp: int,
    H_per_bs: list[NDArray[np.complex128]],
    ue_cb: object,
    bs_cb: object,
    tx_amp: float,
    noise_amplitude: float,
) -> NDArray[np.float64]:
    """Noiseless OBP gain: |w_k^H H f_l|^2 * tx_amp^2 / sigma_n^2."""
    w = ue_cb.codeword(k_obp)  # type: ignore[attr-defined]  # Codebook factory returns object; .codeword exists at runtime
    f = bs_cb.codeword(l_obp)  # type: ignore[attr-defined]  # same
    sigma_sq = noise_amplitude**2
    snr_lin = np.empty(len(H_per_bs))
    for b, H in enumerate(H_per_bs):
        gain_sq = abs(w.conj() @ H @ f) ** 2
        snr_lin[b] = gain_sq * tx_amp**2 / sigma_sq
    return snr_lin


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------


def run_experiment(
    experiment: Experiment,
    *,
    n_workers: int | None = None,
    progress: bool = True,
) -> dict:
    """Run a full Monte Carlo experiment with process-based parallelism.

    Returns a result dict with keys:

    ``snr_db``
        Per-algorithm SNR matrix of shape ``(n_trials, n_steps)``.
    ``obp_history``
        Per-algorithm array of shape ``(n_trials, n_steps, 2)``.
    ``selected_bs``
        Per-algorithm array of shape ``(n_trials, n_steps)`` or ``None``.
    ``seeds``
        1-D array of per-trial seeds.
    ``algorithms``, ``n_trials``, ``n_steps``, ``noise_amplitude``, ``name``
        Metadata fields mirroring the experiment.
    """
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    algos = experiment.algorithms
    n_trials = experiment.n_trials

    # Pre-allocate output arrays
    snr_db_out: dict[str, NDArray[np.float64]] = {
        a: np.zeros((n_trials, experiment.n_steps)) for a in algos
    }
    obp_out: dict[str, NDArray[np.int_]] = {
        a: np.zeros((n_trials, experiment.n_steps, 2), dtype=np.int_) for a in algos
    }
    multi_bs = len(experiment.bs_positions) > 1
    sel_bs_out: dict[str, NDArray[np.int_]] | None = (
        {a: np.zeros((n_trials, experiment.n_steps), dtype=np.int_) for a in algos}
        if multi_bs
        else None
    )
    snr_db_best_agg: dict[str, NDArray[np.float64]] | None = (
        {a: np.zeros((n_trials, experiment.n_steps)) for a in algos} if multi_bs else None
    )
    snr_oracle_agg: NDArray[np.float64] | None = (
        np.zeros((n_trials, experiment.n_steps)) if multi_bs else None
    )
    seeds = np.zeros(n_trials, dtype=np.int64)

    # Progress bar (optional)
    try:
        if progress:
            from tqdm import tqdm

            pbar = tqdm(total=n_trials, desc=experiment.name, unit="trial")
        else:
            pbar = None
    except ImportError:
        pbar = None

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_trial, t, experiment): t for t in range(n_trials)}
        for fut in as_completed(futures):
            t = futures[fut]
            result: TrialResult = fut.result()
            seeds[t] = result.seed
            for a in algos:
                snr_db_out[a][t] = result.snr_db[a]
                obp_out[a][t] = result.obp_history[a]
                if sel_bs_out is not None and result.selected_bs is not None:
                    sel_bs_out[a][t] = result.selected_bs[a]
                if snr_db_best_agg is not None and result.snr_db_best is not None:
                    snr_db_best_agg[a][t] = result.snr_db_best[a]
            if snr_oracle_agg is not None and result.snr_oracle is not None:
                snr_oracle_agg[t] = result.snr_oracle
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    return {
        "name": experiment.name,
        "algorithms": algos,
        "n_trials": n_trials,
        "n_steps": experiment.n_steps,
        "noise_amplitude": experiment.noise_amplitude,
        "snr_db": snr_db_out,
        "obp_history": obp_out,
        "selected_bs": sel_bs_out,
        "snr_db_best": snr_db_best_agg,
        "snr_oracle": snr_oracle_agg,
        "seeds": seeds,
    }


def save_experiment(result: dict, path: str | Path) -> None:
    """Compress and save experiment results to an ``npz`` archive.

    All per-algorithm arrays are stored with keys ``snr_db/<algo>``,
    ``obp_history/<algo>``, etc.  Scalar metadata is stored as 0-d arrays.
    Load with ``np.load(path, allow_pickle=True)``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    arrays["seeds"] = result["seeds"]
    arrays["n_trials"] = np.array(result["n_trials"])
    arrays["n_steps"] = np.array(result["n_steps"])
    arrays["noise_amplitude"] = np.array(result["noise_amplitude"])
    arrays["algorithms"] = np.array(result["algorithms"])
    arrays["name"] = np.array(result["name"])

    for algo in result["algorithms"]:
        arrays[f"snr_db/{algo}"] = result["snr_db"][algo]
        arrays[f"obp_history/{algo}"] = result["obp_history"][algo]

    if result.get("selected_bs") is not None:
        for algo in result["algorithms"]:
            arrays[f"selected_bs/{algo}"] = result["selected_bs"][algo]

    if result.get("snr_db_best") is not None:
        for algo in result["algorithms"]:
            arrays[f"snr_db_best/{algo}"] = result["snr_db_best"][algo]

    if result.get("snr_oracle") is not None:
        arrays["snr_oracle"] = result["snr_oracle"]

    np.savez_compressed(str(path), **arrays)  # type: ignore[arg-type]  # mypy overload signature doesn't handle **kwargs well
