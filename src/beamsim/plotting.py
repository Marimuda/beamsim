"""Journal-quality figure generation for beamsim experiments.

Produces Springer Nature mathphys-num style figures with bootstrap confidence-
interval ribbons.  Pure matplotlib + numpy; no seaborn or pandas dependency.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger("beamsim.plotting")

# ---------------------------------------------------------------------------
# Colour palette and label mapping
# ---------------------------------------------------------------------------

ALGORITHM_PALETTE: dict[str, str] = {
    "exhaustive":         "#1f77b4",
    "nns":                "#2ca02c",
    "tabu":               "#9467bd",
    "angular_prediction": "#ff7f0e",
    "ci":                 "#8c564b",
    "mcmd":               "#d62728",
}

ALGORITHM_LABELS: dict[str, str] = {
    "exhaustive":         "Exhaustive",
    "nns":                "NNS",
    "tabu":               "Tabu",
    "angular_prediction": "Angular prediction",
    "ci":                 "Context information",
    "mcmd":               "MCMD",
}

# ---------------------------------------------------------------------------
# Figure dimensions (Springer Nature column widths)
# ---------------------------------------------------------------------------

SN_SINGLE_COLUMN_INCHES: tuple[float, float] = (3.5, 2.6)   # ~89 mm wide, 4:3
SN_DOUBLE_COLUMN_INCHES: tuple[float, float] = (7.16, 3.0)  # ~183 mm


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

def set_publication_style() -> None:
    """Apply rcParams for Springer Nature mathphys-num journal style."""
    plt.rcParams.update({
        # Font
        "mathtext.fontset":       "cm",
        "font.family":            "serif",
        "font.size":              9,
        "axes.labelsize":         9,
        "legend.fontsize":        8,
        "xtick.labelsize":        8,
        "ytick.labelsize":        8,
        # Lines / axes
        "axes.linewidth":         0.6,
        "lines.linewidth":        1.0,
        "legend.frameon":         False,
        # Resolution and save settings
        "figure.dpi":             300,
        "savefig.dpi":            300,
        "savefig.bbox":           "tight",
        "savefig.transparent":    False,
        # PDF text embedding (TrueType; reviewers can copy text)
        "pdf.fonttype":           42,
    })


# ---------------------------------------------------------------------------
# Figure-size helpers
# ---------------------------------------------------------------------------

def fig_single_column() -> matplotlib.figure.Figure:
    """Return a new Figure sized for a single SN column (~89 mm)."""
    set_publication_style()
    return plt.figure(figsize=SN_SINGLE_COLUMN_INCHES)


def fig_double_column() -> matplotlib.figure.Figure:
    """Return a new Figure sized for a full SN double column (~183 mm)."""
    set_publication_style()
    return plt.figure(figsize=SN_DOUBLE_COLUMN_INCHES)


# ---------------------------------------------------------------------------
# Bootstrap CI helper
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    samples: np.ndarray,
    n_boot: int,
    ci_alpha: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) percentile bands over axis-0 via parametric bootstrap.

    ``samples`` has shape (n_trials, n_x).  When n_trials == 1 the ribbon
    degenerates to the single-trial mean (lo == hi == mean).
    """
    n_trials, n_x = samples.shape
    if n_trials == 1:
        mean = samples[0]
        return mean.copy(), mean.copy()

    boot_means = np.empty((n_boot, n_x), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n_trials, size=n_trials)
        boot_means[b] = samples[idx].mean(axis=0)

    lo = np.percentile(boot_means, 100 * (ci_alpha / 2), axis=0)
    hi = np.percentile(boot_means, 100 * (1 - ci_alpha / 2), axis=0)
    return lo, hi


# ---------------------------------------------------------------------------
# Core plotting primitive
# ---------------------------------------------------------------------------

def plot_curves_with_ci(
    x: np.ndarray,
    traces_per_algo: dict[str, np.ndarray],
    ax: plt.Axes | None = None,
    *,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    ci_alpha: float = 0.05,
    n_boot: int = 1000,
    fill_alpha: float = 0.25,
    legend_loc: str = "best",
    rng: np.random.Generator | None = None,
) -> plt.Axes:
    """Plot mean line + bootstrap CI ribbon per algorithm.

    Parameters
    ----------
    x:
        1-D array of x-axis values, length n_x.
    traces_per_algo:
        Mapping from algorithm key to a (n_trials, n_x) float array of
        per-trial measurements.
    ax:
        Target Axes; created if *None*.
    xlabel / ylabel / title:
        Axis labels.
    xscale / yscale:
        Axis scale strings accepted by ``Axes.set_xscale``.
    ci_alpha:
        Coverage level; ribbon spans (alpha/2, 1-alpha/2) bootstrap
        percentiles.  Set to 0 to suppress ribbons.
    n_boot:
        Number of bootstrap resamples.
    fill_alpha:
        Transparency of the CI ribbon patch.
    legend_loc:
        Passed to ``Axes.legend``.
    rng:
        Seeded RNG for reproducibility.  Created with default seed if *None*.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if ax is None:
        fig_single_column()
        ax = plt.gca()

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, pad=4)

    for algo_key, traces in traces_per_algo.items():
        traces = np.asarray(traces, dtype=np.float64)
        if traces.ndim == 1:
            traces = traces[np.newaxis, :]   # (1, n_x)
        n_trials, n_x = traces.shape

        color = ALGORITHM_PALETTE.get(algo_key, None)
        label = ALGORITHM_LABELS.get(algo_key, algo_key)

        mean = traces.mean(axis=0)
        ax.plot(x, mean, color=color, label=label)

        if ci_alpha > 0 and n_trials > 1:
            lo, hi = _bootstrap_ci(traces, n_boot, ci_alpha, rng)
            ax.fill_between(x, lo, hi, color=color, alpha=fill_alpha, linewidth=0)
            half_width = float(np.mean((hi - lo) / 2))
        else:
            half_width = 0.0

        logger.info(
            "%s: n_trials=%d, mean=%.4g, CI half-width=%.4g",
            algo_key, n_trials, float(mean.mean()), half_width,
        )

    ax.legend(loc=legend_loc)
    return ax


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_figure(fig: matplotlib.figure.Figure, path: str | Path) -> None:
    """Save *fig* as a PDF with tight layout; log file size and page dims."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, format="pdf", bbox_inches="tight", transparent=False)
    size_kb = os.path.getsize(dest) / 1024
    w_in, h_in = fig.get_size_inches()
    logger.info(
        "Saved %s  (%.1f KB, %.2f x %.2f in)",
        dest, size_kb, w_in, h_in,
    )


# ---------------------------------------------------------------------------
# Specialised wrappers — one per experiment
# ---------------------------------------------------------------------------

def plot_rotational(
    npz_path: str | Path,
    *,
    gamma_th_db: float | None = None,
    output_path: str | Path,
) -> Path:
    """Rotational-velocity sweep: mean received power (dB) vs rpm (log x).

    Expected NPZ keys: ``x`` (rpm array), ``<algo>`` or ``snr_db_<algo>``
    traces shaped (n_trials, n_x).
    """
    set_publication_style()
    data = np.load(npz_path)
    x = data["x"]

    traces = _extract_traces(data)
    fig = fig_single_column()
    ax = fig.add_subplot(111)

    plot_curves_with_ci(
        x, traces, ax,
        xlabel="Rotational velocity (rpm)",
        ylabel="Mean received power (dB)",
        xscale="log",
        yscale="linear",
    )

    if gamma_th_db is not None:
        ax.axhline(gamma_th_db, color="k", linewidth=0.6, linestyle="--",
                   label=rf"$\Gamma_{{th}}={gamma_th_db}\,\mathrm{{dB}}$")
        ax.legend()

    dest = Path(output_path)
    save_figure(fig, dest)
    plt.close(fig)
    return dest


def plot_alpha_sweep(
    npz_path: str | Path,
    *,
    gamma_th_db: float,
    output_path: str | Path,
) -> Path:
    """Alpha-factor sweep: coverage rate vs alpha (log x, 0.5–8)."""
    set_publication_style()
    data = np.load(npz_path)
    x = data["x"]

    traces = _extract_traces(data)
    fig = fig_single_column()
    ax = fig.add_subplot(111)

    plot_curves_with_ci(
        x, traces, ax,
        xlabel=r"Measurement-rate factor $\alpha$",
        ylabel="Coverage rate",
        xscale="log",
        yscale="linear",
    )

    ax.axhline(gamma_th_db, color="k", linewidth=0.6, linestyle="--",
               label=rf"$\Gamma_{{th}}={gamma_th_db}\,\mathrm{{dB}}$")
    ax.legend()

    dest = Path(output_path)
    save_figure(fig, dest)
    plt.close(fig)
    return dest


def plot_snr_sweep(
    npz_path: str | Path,
    *,
    gamma_th_db: float,
    output_path: str | Path,
) -> Path:
    """SNR sweep: coverage rate vs input SNR (dB), 61-step linear range."""
    set_publication_style()
    data = np.load(npz_path)
    x = data["x"]

    traces = _extract_traces(data)
    fig = fig_single_column()
    ax = fig.add_subplot(111)

    plot_curves_with_ci(
        x, traces, ax,
        xlabel="Input SNR (dB)",
        ylabel="Coverage rate",
        xscale="linear",
        yscale="linear",
    )

    ax.axhline(gamma_th_db, color="k", linewidth=0.6, linestyle="--",
               label=rf"$\Gamma_{{th}}={gamma_th_db}\,\mathrm{{dB}}$")
    ax.legend()

    dest = Path(output_path)
    save_figure(fig, dest)
    plt.close(fig)
    return dest


def plot_handover(
    npz_path: str | Path,
    *,
    output_path: str | Path,
) -> Path:
    """Handover figure: L_BS (dB) vs occasion / distance."""
    set_publication_style()
    data = np.load(npz_path)
    x = data["x"]

    traces = _extract_traces(data)
    fig = fig_single_column()
    ax = fig.add_subplot(111)

    plot_curves_with_ci(
        x, traces, ax,
        xlabel="Occasion / distance",
        ylabel=r"$L_{\mathrm{BS}}$ (dB)",
        xscale="linear",
        yscale="linear",
    )

    dest = Path(output_path)
    save_figure(fig, dest)
    plt.close(fig)
    return dest


# ---------------------------------------------------------------------------
# Private utility
# ---------------------------------------------------------------------------

def _extract_traces(data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    """Parse an NPZ file produced by the runner into algo->traces mapping.

    Supports two key conventions:
    - ``snr_db_<algo>``  (runner-tagged form)
    - ``<algo>``         (bare algo name)
    """
    result: dict[str, np.ndarray] = {}
    for key in data.files:
        if key == "x":
            continue
        stripped = key.removeprefix("snr_db_")
        if stripped in ALGORITHM_PALETTE:
            result[stripped] = data[key]
        elif key in ALGORITHM_PALETTE:
            result[key] = data[key]
    return result
