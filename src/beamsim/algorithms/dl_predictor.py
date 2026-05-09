"""DL beam-prediction algorithm wrapper.

Wraps BeamPredictorMLP as an Algorithm instance.  On ``reset``, lazily loads
a pre-trained checkpoint from ``models/beam_predictor.pt`` (relative to the
working directory, or an explicit path).  If ``require_checkpoint=True`` and
the checkpoint is missing, ``reset`` raises ``FileNotFoundError`` so a
benchmark cannot silently fall through to Exhaustive.  When called from
unit tests we keep the fallback-with-warning path so the test suite can
exercise the wrapper without a trained model.

Reference (modern equivalent):
    Kim et al. (2023). "Machine Learning Based Time Domain Millimeter-Wave
    Beam Prediction for 5G-Advanced and Beyond: Design, Analysis, and
    Over-The-Air Experiments," IEEE J. Sel. Areas Commun. 41(6),
    DOI: 10.1109/JSAC.2023.3275613.  Kim et al. use an LSTM on a window
    of past RSRP measurements to predict the next-best beam pair, with
    >50% beam-management power saving in OTA tests.

We ship a *simpler* MLP variant (3 hidden layers) on a 4-step OBP-index
window — closer to early sequence-based prediction baselines.  Phase 4C
will replace this MLP with an LSTM trained on the same trajectories so
the comparison is on equal footing with Kim et al.

Klautau et al. 2018 ("5G MIMO Data for Machine Learning") was previously
cited here but is **not** the right reference: that paper uses ray-tracing
channel matrices and environment metadata as DL inputs, not a window of
past OBP indices, and applies a different output head (top-K beam ranking).

Known limitation: train/inference distribution mismatch.  Training labels
are produced by running ``Exhaustive`` (which sweeps every (k,l) per step),
so the OBP at training time is the argmax of a fully-populated BPLM.  At
inference time only the most recent (k,l) is updated each step, so
``state.obp()`` is the argmax of a stale, partially-swept BPLM.  We document
the gap rather than papering over it; a fully online predictor (or a
training-time policy that probes one beam per step) would close it.
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
            if self._require_checkpoint:
                raise FileNotFoundError(
                    f"DLPredictor: checkpoint '{self._ckpt_path}' not found and "
                    "require_checkpoint=True. Train with: "
                    "python -m beamsim.algorithms._dl.train"
                )
            warnings.warn(
                f"DLPredictor: checkpoint '{self._ckpt_path}' not found — "
                "falling back to Exhaustive (set require_checkpoint=True to fail loudly). "
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
