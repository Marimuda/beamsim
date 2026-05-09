"""Smoke tests for the runnable examples.

The minimal example is executed as a subprocess (the way users invoke it),
because ``ProcessPoolExecutor`` requires factory callables to live in a
real importable module, which an in-test ``importlib.util.spec_from_file_location``
load does not provide.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"


def test_minimal_example_runs():
    """The minimal example finishes cleanly and reports finite per-algo SNR."""
    script = EXAMPLES / "minimal_example.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"minimal_example.py exited {proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    pattern = re.compile(r"^(exhaustive|nns|mcmd)\s+mean SNR\s+=\s+([+\-\d.]+)\s+dB", re.MULTILINE)
    combined_output = proc.stdout + proc.stderr  # logging defaults to stderr
    matches = pattern.findall(combined_output)
    assert {algo for algo, _ in matches} == {"exhaustive", "nns", "mcmd"}, (
        f"missing per-algo lines in:\n{combined_output}"
    )
    for _, snr in matches:
        assert float(snr) > -200.0  # basic sanity bound; floor is just "finite-ish"


def test_minimal_example_uses_only_public_api():
    """Guard against the example reaching into beamsim internals."""
    src = (EXAMPLES / "minimal_example.py").read_text()
    forbidden = ["beamsim._", "from beamsim.algorithms.base import", "BPLMState._"]
    for token in forbidden:
        assert token not in src, f"example imports private symbol: {token!r}"
