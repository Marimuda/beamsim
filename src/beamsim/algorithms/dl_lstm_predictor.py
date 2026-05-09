"""LSTM-based DL beam-prediction algorithm wrapper.

Loads a pre-trained LSTM (``BeamPredictorLSTM``) from
``models/beam_predictor_lstm.pt`` by default and consumes a window of
recent OBP pairs as a sequence input.  Falls back to ``Exhaustive`` (with
a loud warning) when the checkpoint is missing or torch is unavailable
unless ``require_checkpoint=True``.

Reference:
    Kim, Y., Yoon, S., Lee, J., Cho, J. (2023). "Machine Learning Based
    Time Domain Millimeter-Wave Beam Prediction for 5G-Advanced and
    Beyond: Design, Analysis, and Over-The-Air Experiments," IEEE
    J. Sel. Areas Commun. 41(6), DOI: 10.1109/JSAC.2023.3275613.

This wrapper is the modern (sequence-based) counterpart to the legacy
``DLPredictor`` (MLP) and shares the same training script.

Train with::

    python -m beamsim.algorithms._dl.train --model lstm \\
        --output models/beam_predictor_lstm.pt
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

logger = logging.getLogger("beamsim.dl_lstm_predictor")

_DEFAULT_CHECKPOINT = Path("models/beam_predictor_lstm.pt")
_WINDOW = 4


class DLLSTMPredictor(Algorithm):
    """LSTM beam-prediction baseline (Kim et al. JSAC 2023 style)."""

    name = "dl_lstm_predictor"

    def __init__(
        self,
        checkpoint: str | Path = _DEFAULT_CHECKPOINT,
        require_checkpoint: bool = False,
    ) -> None:
        self._ckpt_path = Path(checkpoint)
        self._require_checkpoint = require_checkpoint
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
        if np.any(state.measured_at >= 0):
            self._obp_history.append(state.obp())

        if self._use_fallback or self._model is None:
            assert self._fallback is not None
            return self._fallback.select_next_mbp(state, m, context)

        if len(self._obp_history) < self._window:
            assert self._fallback is not None
            return self._fallback.select_next_mbp(state, m, context)

        return self._predict(state)

    def _predict(self, state: BPLMState) -> tuple[int, int]:
        try:
            from collections.abc import Callable

            import torch

            # Sequence input: (1, window, 2).
            seq = np.array(list(self._obp_history), dtype=np.float32).reshape(1, self._window, 2)
            model: Callable = self._model  # type: ignore[assignment]
            with torch.no_grad():
                logits = model(torch.tensor(seq))
                flat_idx = int(logits.argmax(dim=1).item())
            k = flat_idx // self._L
            l = flat_idx % self._L
            k = int(np.clip(k, 0, state.K - 1))
            l = int(np.clip(l, 0, state.L - 1))
            return k, l
        except Exception as exc:
            logger.warning("DLLSTMPredictor forward pass failed (%s); using fallback.", exc)
            self._use_fallback = True
            assert self._fallback is not None
            return self._fallback.select_next_mbp(state, 0, {})

    def _try_load_model(self) -> None:
        try:
            import torch

            from beamsim.algorithms._dl.lstm_predictor import BeamPredictorLSTM
        except ImportError:
            warnings.warn(
                "DLLSTMPredictor: torch not installed — falling back to Exhaustive. "
                "Install with: pip install -e .[dl]",
                stacklevel=3,
            )
            self._use_fallback = True
            return

        if not self._ckpt_path.exists():
            if self._require_checkpoint:
                raise FileNotFoundError(
                    f"DLLSTMPredictor: checkpoint '{self._ckpt_path}' not found and "
                    "require_checkpoint=True. Train with: "
                    "python -m beamsim.algorithms._dl.train --model lstm"
                )
            warnings.warn(
                f"DLLSTMPredictor: checkpoint '{self._ckpt_path}' not found — "
                "falling back to Exhaustive. "
                "Train with: python -m beamsim.algorithms._dl.train --model lstm",
                stacklevel=3,
            )
            self._use_fallback = True
            return

        ckpt = torch.load(self._ckpt_path, map_location="cpu", weights_only=True)
        model = BeamPredictorLSTM(
            input_dim=ckpt.get("input_dim", 2),
            hidden_dim=ckpt.get("hidden_dim", 64),
            output_dim=ckpt.get("output_dim", 256),
            n_layers=ckpt.get("n_layers", 1),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        self._model = model
        self._L = ckpt.get("L", 32)
        self._window = ckpt.get("window", _WINDOW)
        logger.info("DLLSTMPredictor: loaded checkpoint from %s", self._ckpt_path)
