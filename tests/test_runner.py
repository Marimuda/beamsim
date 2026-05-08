"""Smoke tests for beamsim.runner.

The factories used here are module-level functions (not lambdas) so they
survive pickle-based serialisation into subprocess workers.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from beamsim.channel import FreeSpaceLosChannel
from beamsim.geometry import rotation_track, Track
from beamsim.runner import Experiment, TrialResult, _run_trial, run_experiment, save_experiment


# ---------------------------------------------------------------------------
# Picklable factories (module-level so ProcessPoolExecutor can serialise them)
# ---------------------------------------------------------------------------

_BS_XY = np.array([10.0, 0.0])
_BS_YAW = 0.0
_N_BS_ELEMENTS = 16
_N_UE_ELEMENTS = 4


def _track_factory(rng: np.random.Generator) -> Track:
    """Stationary rotation track for smoke tests."""
    return rotation_track(
        position_xy=(0.0, 0.0),
        rpm=6.0,
        n_steps=200,
        dt=0.01,
        initial_orientation=float(rng.uniform(0, 2 * np.pi)),
    )


def _channel_factory(rng: np.random.Generator, bs_index: int) -> FreeSpaceLosChannel:
    """Free-space LOS channel — deterministic given BS position."""
    return FreeSpaceLosChannel(
        bs_xy=_BS_XY,
        bs_yaw=_BS_YAW,
        n_bs_elements=_N_BS_ELEMENTS,
        n_ue_elements=_N_UE_ELEMENTS,
    )


def _make_experiment(seed: int = 12345, n_trials: int = 3) -> Experiment:
    return Experiment(
        name="smoke",
        n_steps=200,
        dt=0.01,
        n_trials=n_trials,
        algorithms=["exhaustive", "ci"],
        bs_positions=[(10.0, 0.0)],
        bs_yaws=[0.0],
        track_factory=_track_factory,
        channel_factory=_channel_factory,
        noise_amplitude=1e-3,
        tx_amp=1.0,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrialResult:
    """Unit-level: test _run_trial directly (in-process)."""

    def test_shapes(self):
        exp = _make_experiment()
        result = _run_trial(0, exp)
        assert isinstance(result, TrialResult)
        for algo in exp.algorithms:
            assert result.snr_db[algo].shape == (exp.n_steps,), (
                f"snr_db[{algo}] shape mismatch"
            )
            assert result.obp_history[algo].shape == (exp.n_steps, 2), (
                f"obp_history[{algo}] shape mismatch"
            )

    def test_all_finite(self):
        exp = _make_experiment()
        result = _run_trial(0, exp)
        for algo in exp.algorithms:
            assert np.all(np.isfinite(result.snr_db[algo])), (
                f"Non-finite SNR values for algo={algo}"
            )

    def test_single_bs_no_selected_bs(self):
        exp = _make_experiment()
        result = _run_trial(0, exp)
        assert result.selected_bs is None

    def test_distinct_seeds_distinct_traces(self):
        """Different trial indices must produce different SNR traces."""
        exp = _make_experiment()
        r0 = _run_trial(0, exp)
        r1 = _run_trial(1, exp)
        # At least one algorithm's trace must differ
        any_differ = any(
            not np.allclose(r0.snr_db[a], r1.snr_db[a])
            for a in exp.algorithms
        )
        assert any_differ, "Distinct trial seeds produced identical SNR traces"

    def test_obp_indices_in_range(self):
        from beamsim.codebook import make_default_ue_codebook, make_default_bs_codebook
        exp = _make_experiment()
        result = _run_trial(0, exp)
        ue_cb = make_default_ue_codebook()
        bs_cb = make_default_bs_codebook()
        for algo in exp.algorithms:
            hist = result.obp_history[algo]
            assert np.all(hist[:, 0] >= 0) and np.all(hist[:, 0] < ue_cb.n_beams)
            assert np.all(hist[:, 1] >= 0) and np.all(hist[:, 1] < bs_cb.n_beams)


class TestRunExperiment:
    """Integration-level: run_experiment with multiple trials."""

    def test_output_shapes(self):
        exp = _make_experiment(n_trials=3)
        result = run_experiment(exp, n_workers=1, progress=False)
        for algo in exp.algorithms:
            assert result["snr_db"][algo].shape == (3, 200)
            assert result["obp_history"][algo].shape == (3, 200, 2)

    def test_all_values_finite(self):
        exp = _make_experiment(n_trials=3)
        result = run_experiment(exp, n_workers=1, progress=False)
        for algo in exp.algorithms:
            assert np.all(np.isfinite(result["snr_db"][algo]))

    def test_metadata_fields(self):
        exp = _make_experiment(n_trials=3)
        result = run_experiment(exp, n_workers=1, progress=False)
        assert result["n_trials"] == 3
        assert result["n_steps"] == 200
        assert set(result["algorithms"]) == {"exhaustive", "ci"}
        assert len(result["seeds"]) == 3

    def test_distinct_seeds_in_result(self):
        """All per-trial seeds should be distinct."""
        exp = _make_experiment(n_trials=3)
        result = run_experiment(exp, n_workers=1, progress=False)
        assert len(set(result["seeds"].tolist())) == 3

    def test_different_experiment_seeds_give_different_traces(self):
        exp_a = _make_experiment(seed=1111, n_trials=2)
        exp_b = _make_experiment(seed=2222, n_trials=2)
        ra = run_experiment(exp_a, n_workers=1, progress=False)
        rb = run_experiment(exp_b, n_workers=1, progress=False)
        algo = "exhaustive"
        assert not np.allclose(ra["snr_db"][algo], rb["snr_db"][algo]), (
            "Different seeds produced identical SNR matrices"
        )


class TestSaveExperiment:
    """Verify save/load round-trip."""

    def test_npz_roundtrip(self, tmp_path):
        exp = _make_experiment(n_trials=2)
        result = run_experiment(exp, n_workers=1, progress=False)
        out = tmp_path / "smoke.npz"
        save_experiment(result, out)
        loaded = np.load(str(out), allow_pickle=True)
        for algo in exp.algorithms:
            np.testing.assert_array_equal(
                loaded[f"snr_db/{algo}"], result["snr_db"][algo]
            )
        assert loaded["n_trials"] == 2
        assert loaded["n_steps"] == 200

    def test_file_created(self, tmp_path):
        exp = _make_experiment(n_trials=1)
        result = run_experiment(exp, n_workers=1, progress=False)
        out = tmp_path / "sub" / "exp.npz"
        save_experiment(result, out)
        assert out.exists()
