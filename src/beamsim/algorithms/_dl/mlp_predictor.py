"""Beam-predictor MLP for offline-trained DL baseline.

Implements a 3-hidden-layer MLP that maps a window of recent beam-pair
observations to a logit distribution over the K*L = 256 flat beam indices.

Reference (paraphrased):
  Klautau et al. (2018) "Deep Learning for Beam Selection at mmWave" and
  Heng et al. (2021) survey of ML beam-alignment baselines.  Input: last
  ``window`` OBP pairs (k, l) flattened; output: classification over all
  K*L indices via softmax at test time (argmax during inference).

Architecture choices:
  - Input  : 2*window = 8 floats (4 OBPs of 2 indices each).
  - Hidden : 3 layers, 128 units, ReLU, dropout 0.1.
  - Output : K*L = 8*32 = 256 logits.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True

    class BeamPredictorMLP(nn.Module):
        """3-layer MLP beam predictor.

        Parameters
        ----------
        input_dim:
            2 * window size (default 8 for window=4).
        output_dim:
            K * L (default 256 for UE=8 beams, BS=32 beams).
        hidden:
            Units per hidden layer.
        dropout:
            Dropout probability applied after each hidden ReLU.
        """

        def __init__(
            self,
            input_dim: int = 8,
            output_dim: int = 256,
            hidden: int = 128,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return logits of shape (batch, output_dim)."""
            return self.net(x)

except ImportError:
    TORCH_AVAILABLE = False

    class BeamPredictorMLP:  # type: ignore[no-redef]
        """Stub — torch not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "torch is required for BeamPredictorMLP. Install with: pip install -e .[dl]"
            )
