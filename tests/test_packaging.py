"""Smoke tests for the installed package surface.

Locks the public re-exports so accidental removals raise immediately, and
asserts the ``py.typed`` marker is shipped.
"""

from __future__ import annotations

import importlib

import beamsim


def test_version_is_a_str():
    assert isinstance(beamsim.__version__, str)
    assert beamsim.__version__.count(".") >= 2  # major.minor.patch


def test_public_api_is_complete():
    expected = {
        "BPLMState",
        "ChannelParams",
        "ChannelRealisation",
        "Codebook",
        "Experiment",
        "FreeSpaceLosChannel",
        "Track",
        "TrialResult",
        "__version__",
        "run_experiment",
        "save_experiment",
    }
    assert set(beamsim.__all__) == expected
    for name in expected:
        assert hasattr(beamsim, name), f"public symbol missing at runtime: {name}"


def test_algorithms_subpackage_exports_are_complete():
    algos = importlib.import_module("beamsim.algorithms")
    assert "ALL_ALGORITHMS" in algos.__all__
    # Every key in ALL_ALGORITHMS must resolve to a class.
    for key, klass in algos.ALL_ALGORITHMS.items():
        assert isinstance(klass, type), f"{key!r} -> {klass!r} is not a class"


def test_py_typed_marker_is_present():
    """py.typed must ship with the installed package so downstream type-checkers
    treat beamsim as typed."""
    pkg_dir = importlib.util.find_spec("beamsim").submodule_search_locations  # type: ignore[union-attr]
    assert pkg_dir is not None
    marker_files = [str(p) for p in pkg_dir]
    found = False
    for d in marker_files:
        from pathlib import Path

        if (Path(d) / "py.typed").is_file():
            found = True
            break
    assert found, f"py.typed not shipped under any of: {marker_files}"
