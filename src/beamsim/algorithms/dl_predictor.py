"""DL beam-prediction algorithm wrapper.

Wraps BeamPredictorMLP as an Algorithm instance.  On ``reset``, lazily loads
a pre-trained checkpoint from ``models/beam_predictor.pt`` (relative to the
working directory, or an explicit path).  If the checkpoint is missing or
torch is unavailable, falls back to Exhaustive with a loud warning.

Reference (paraphrased):
    Klautau et al. (2018), Heng et al. (2021): DL beam-selection baseline
    uses a small MLP trained on offline exhaustive-sweep trajectories.
"""

from __future__ import annotations

import logging
import warnings
from collections import deque
from pathlib import Path

import numpy as np

from beamsim.algorithms.base import Algorithm
from beamsim.algorithms.exhaustive import Exhaustive
from beamsim.bplm import BPLMState

logger = logging.getLogger("beamsim.dl_predictor")

_DEFAULT_CHECKPOINT = Path("models/beam_predictor.pt")
_WINDOW = 4  # must match training


class DLPredictor(Algorithm):
    """MLP beam-prediction baseline (Klautau 2018 / Heng 2021 style).

    Parameters
    ----------
    checkpoint:
        Path to the ``torch.save`` checkpoint produced by
        ``beamsim.algorithms._dl.train``.  Defaults to
        ``models/beam_predictor.pt`` (relative to cwd).
    """

    name = "dl_predictor"

    def __init__(self, checkpoint: str | Path = _DEFAULT_CHECKPOINT) -> None:
        self._ckpt_path = Path(checkpoint)
        self._model: object | None = None
        self._L: int = 32
        self._window: int = _WINDOW
        self._fallback: Exhaustive | None = None
        self._use_fallback: bool = False

    def reset(self, state: BPLMState, context: dict) -> None:
        self._obp_history: deque[tuple[int, int]] = deque(maxlen=self._window)
        self._fallback = Exhaustive()
        self._fallback.reset(state, context)
        self._use_fallback = False
        self._L = state.L

        if self._model is None:
            self._try_load_model()

    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        # Update OBP history from current BPLM state
        if np.any(state.measured_at >= 0):
            self._obp_history.append(state.obp())

        if self._use_fallback or self._model is None:
            assert self._fallback is not None
            return self._fallback.select_next_mbp(state, m, context)

        if len(self._obp_history) < self._window:
            # Not enough history yet — use fallback
            assert self._fallback is not None
            return self._fallback.select_next_mbp(state, m, context)

        return self._predict(state)

    def _predict(self, state: BPLMState) -> tuple[int, int]:
        """Run one forward pass and convert argmax to (k, l)."""
        try:
            from collections.abc import Callable

            import torch

            x = np.array(list(self._obp_history), dtype=np.float32).ravel()
            model: Callable = self._model  # type: ignore[assignment]
            with torch.no_grad():
                logits = model(torch.tensor(x).unsqueeze(0))
                flat_idx = int(logits.argmax(dim=1).item())
            k = flat_idx // self._L
            l = flat_idx % self._L
            k = int(np.clip(k, 0, state.K - 1))
            l = int(np.clip(l, 0, state.L - 1))
            return k, l
        except Exception as exc:
            logger.warning("DLPredictor forward pass failed (%s); using fallback.", exc)
            self._use_fallback = True
            assert self._fallback is not None
            return self._fallback.select_next_mbp(state, 0, {})

    def _try_load_model(self) -> None:
        """Attempt to load checkpoint; set _use_fallback on failure."""
        try:
            import torch

            from beamsim.algorithms._dl.mlp_predictor import BeamPredictorMLP
        except ImportError:
            warnings.warn(
                "DLPredictor: torch not installed — falling back to Exhaustive. "
                "Install with: pip install -e .[dl]",
                stacklevel=3,
            )
            self._use_fallback = True
            return

        if not self._ckpt_path.exists():
            warnings.warn(
                f"DLPredictor: checkpoint '{self._ckpt_path}' not found — "
                "falling back to Exhaustive. "
                "Train with: python -m beamsim.algorithms._dl.train",
                stacklevel=3,
            )
            self._use_fallback = True
            return

        ckpt = torch.load(self._ckpt_path, map_location="cpu", weights_only=True)
        model = BeamPredictorMLP(
            input_dim=ckpt.get("input_dim", 8),
            output_dim=ckpt.get("output_dim", 256),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        self._model = model
        self._L = ckpt.get("L", 32)
        self._window = ckpt.get("window", _WINDOW)
        logger.info("DLPredictor: loaded checkpoint from %s", self._ckpt_path)
