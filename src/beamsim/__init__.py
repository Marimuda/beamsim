"""beamsim — geometry-based mmWave beam-alignment simulator.

Public entry points:

- :mod:`beamsim.algorithms` — beam-management policies (one class per file)
- :class:`beamsim.runner.Experiment` / :func:`beamsim.runner.run_experiment` —
  Monte Carlo orchestration with common-random-number pairing across algorithms
- :class:`beamsim.channel.ChannelRealisation` — simplified TR 38.901 cluster
  generator at 28 GHz
- :class:`beamsim.codebook.Codebook` — cosine-spaced linear-phase ULA codebook
- :class:`beamsim.bplm.BPLMState` — beam-pair-level measurement bookkeeping

The CLI ``beamsim-run`` (declared in ``[project.scripts]``) is a thin Hydra
adapter over :func:`beamsim.run.run_from_config`.
"""

from beamsim.bplm import BPLMState
from beamsim.channel import ChannelParams, ChannelRealisation, FreeSpaceLosChannel
from beamsim.codebook import Codebook
from beamsim.geometry import Track
from beamsim.runner import Experiment, TrialResult, run_experiment, save_experiment

__version__ = "0.2.1"

__all__ = [
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
]
