"""Tests for beamsim.plotting — journal-quality figure generation."""

import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless; must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pytest

from beamsim.plotting import (
    ALGORITHM_LABELS,
    ALGORITHM_PALETTE,
    SN_DOUBLE_COLUMN_INCHES,
    SN_SINGLE_COLUMN_INCHES,
    fig_double_column,
    fig_single_column,
    plot_curves_with_ci,
    save_figure,
    set_publication_style,
    plot_rotational,
    plot_alpha_sweep,
    plot_snr_sweep,
    plot_handover,
)


# ---------------------------------------------------------------------------
# Palette completeness
# ---------------------------------------------------------------------------

EXPECTED_ALGOS = {"exhaustive", "nns", "tabu", "angular_prediction", "ci", "mcmd"}


def test_palette_covers_all_algorithms():
    assert set(ALGORITHM_PALETTE.keys()) == EXPECTED_ALGOS


def test_labels_covers_all_algorithms():
    assert set(ALGORITHM_LABELS.keys()) == EXPECTED_ALGOS


def test_palette_colours_are_valid_hex():
    import re
    hex_re = re.compile(r"^#[0-9a-fA-F]{6}$")
    for algo, colour in ALGORITHM_PALETTE.items():
        assert hex_re.match(colour), f"Invalid hex colour for {algo!r}: {colour!r}"


# ---------------------------------------------------------------------------
# Figure-size sanity
# ---------------------------------------------------------------------------

def test_fig_single_column_size():
    fig = fig_single_column()
    w, h = fig.get_size_inches()
    assert abs(w - SN_SINGLE_COLUMN_INCHES[0]) < 0.01
    assert abs(h - SN_SINGLE_COLUMN_INCHES[1]) < 0.01
    plt.close(fig)


def test_fig_double_column_size():
    fig = fig_double_column()
    w, h = fig.get_size_inches()
    assert abs(w - SN_DOUBLE_COLUMN_INCHES[0]) < 0.01
    assert abs(h - SN_DOUBLE_COLUMN_INCHES[1]) < 0.01
    plt.close(fig)


# ---------------------------------------------------------------------------
# set_publication_style round-trip
# ---------------------------------------------------------------------------

def test_set_publication_style_applies_rcparams():
    set_publication_style()
    assert plt.rcParams["font.family"] == ["serif"]
    assert plt.rcParams["mathtext.fontset"] == "cm"
    assert plt.rcParams["font.size"] == 9
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["legend.frameon"] is False


# ---------------------------------------------------------------------------
# plot_curves_with_ci — happy path (multiple trials)
# ---------------------------------------------------------------------------

def test_plot_curves_with_ci_returns_axes():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 50)
    traces = {"exhaustive": rng.random((10, 50)), "nns": rng.random((10, 50))}
    fig, ax_in = plt.subplots()
    ax_out = plot_curves_with_ci(x, traces, ax_in, xlabel="x", ylabel="y", rng=rng)
    assert ax_out is ax_in
    plt.close(fig)


def test_plot_curves_with_ci_produces_lines_and_ribbons():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 20)
    traces = {"mcmd": rng.random((5, 20)), "tabu": rng.random((5, 20))}
    fig, ax = plt.subplots()
    plot_curves_with_ci(x, traces, ax, xlabel="X", ylabel="Y", rng=rng)
    # 2 mean lines
    assert len(ax.lines) == 2
    # 2 fill_between collections (ribbons)
    assert len(ax.collections) == 2
    plt.close(fig)


def test_plot_curves_with_ci_legend_present():
    rng = np.random.default_rng(2)
    x = np.arange(10, dtype=float)
    traces = {"ci": rng.random((3, 10))}
    fig, ax = plt.subplots()
    plot_curves_with_ci(x, traces, ax, xlabel="X", ylabel="Y", rng=rng)
    assert ax.get_legend() is not None
    plt.close(fig)


# ---------------------------------------------------------------------------
# Degenerate case: single trial — no CI ribbon should be drawn
# ---------------------------------------------------------------------------

def test_single_trial_no_ribbon():
    rng = np.random.default_rng(3)
    x = np.linspace(-5, 5, 30)
    traces = {"exhaustive": rng.random((1, 30))}
    fig, ax = plt.subplots()
    plot_curves_with_ci(x, traces, ax, xlabel="X", ylabel="Y", rng=rng)
    # One mean line
    assert len(ax.lines) == 1
    # No fill_between collections
    assert len(ax.collections) == 0
    plt.close(fig)


# ---------------------------------------------------------------------------
# save_figure round-trip
# ---------------------------------------------------------------------------

def test_save_figure_writes_nonempty_pdf():
    set_publication_style()
    fig, ax = plt.subplots(figsize=SN_SINGLE_COLUMN_INCHES)
    ax.plot([0, 1], [0, 1])
    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / "test_output.pdf"
        save_figure(fig, dest)
        assert dest.exists(), "PDF was not created"
        assert os.path.getsize(dest) > 0, "PDF is empty"
    plt.close(fig)


def test_save_figure_creates_parent_directories():
    set_publication_style()
    fig, ax = plt.subplots(figsize=SN_SINGLE_COLUMN_INCHES)
    ax.plot([0, 1], [1, 0])
    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / "nested" / "dir" / "fig.pdf"
        save_figure(fig, dest)
        assert dest.exists()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Specialised wrapper smoke tests (NPZ fixtures generated in-memory)
# ---------------------------------------------------------------------------

def _write_npz(tmpdir: Path, algos: list[str], n_x: int, n_trials: int) -> Path:
    rng = np.random.default_rng(99)
    npz_path = tmpdir / "data.npz"
    payload = {"x": np.linspace(0, 1, n_x)}
    for algo in algos:
        payload[algo] = rng.random((n_trials, n_x))
    np.savez(npz_path, **payload)
    return npz_path


@pytest.fixture()
def tmp(tmp_path):
    return tmp_path


def test_plot_rotational_produces_pdf(tmp):
    npz = _write_npz(tmp, ["exhaustive", "mcmd"], n_x=20, n_trials=5)
    out = tmp / "rotational.pdf"
    result = plot_rotational(npz, output_path=out)
    assert result == out
    assert out.exists() and os.path.getsize(out) > 0


def test_plot_alpha_sweep_produces_pdf(tmp):
    npz = _write_npz(tmp, ["nns", "tabu"], n_x=15, n_trials=4)
    out = tmp / "alpha.pdf"
    result = plot_alpha_sweep(npz, gamma_th_db=-5.0, output_path=out)
    assert result == out
    assert out.exists() and os.path.getsize(out) > 0


def test_plot_snr_sweep_produces_pdf(tmp):
    npz = _write_npz(tmp, ["ci", "angular_prediction"], n_x=61, n_trials=3)
    out = tmp / "snr.pdf"
    result = plot_snr_sweep(npz, gamma_th_db=0.0, output_path=out)
    assert result == out
    assert out.exists() and os.path.getsize(out) > 0


def test_plot_handover_produces_pdf(tmp):
    npz = _write_npz(tmp, ["exhaustive", "mcmd"], n_x=25, n_trials=6)
    out = tmp / "handover.pdf"
    result = plot_handover(npz, output_path=out)
    assert result == out
    assert out.exists() and os.path.getsize(out) > 0


def test_plot_rotational_with_snr_db_prefix(tmp):
    """Runner may store traces as ``snr_db_<algo>``; wrapper must handle it."""
    rng = np.random.default_rng(7)
    npz_path = tmp / "data_prefixed.npz"
    np.savez(
        npz_path,
        x=np.logspace(0, 3, 20),
        snr_db_exhaustive=rng.random((5, 20)),
        snr_db_mcmd=rng.random((5, 20)),
    )
    out = tmp / "rotational_prefixed.pdf"
    result = plot_rotational(npz_path, output_path=out)
    assert out.exists() and os.path.getsize(out) > 0


def test_plot_curves_with_ci_xscale_log(tmp):
    """Log x-scale should not raise for positive x values."""
    rng = np.random.default_rng(5)
    x = np.logspace(0, 2, 30)
    traces = {"exhaustive": rng.random((4, 30))}
    fig, ax = plt.subplots()
    plot_curves_with_ci(x, traces, ax, xlabel="rpm", ylabel="dB",
                        xscale="log", rng=rng)
    assert ax.get_xscale() == "log"
    plt.close(fig)
