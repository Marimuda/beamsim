"""LSTM beam-sequence predictor (DL beam-prediction baseline).

Reference:
    Kim, Y., Yoon, S., Lee, J., Cho, J. (2023). "Machine Learning Based
    Time Domain Millimeter-Wave Beam Prediction for 5G-Advanced and
    Beyond: Design, Analysis, and Over-The-Air Experiments," IEEE
    J. Sel. Areas Commun. 41(6), DOI: 10.1109/JSAC.2023.3275613.

Input window of past OBP indices -> next-step beam pair, classified
over K * L flat indices.  This is the modern "TBP" (time-domain beam
prediction) variant of supervised DL beam selection — Kim et al. 2023
report >50% beam-management power saving on a 3GPP NR-compliant
testbed using an LSTM on RSRP-history features.

We use a smaller architecture than Kim 2023 (single-layer LSTM,
hidden=64) because our window is short (4 OBPs) and our codebook is
small (K=8, L=32, 256 flat classes).  The same training pipeline
(``beamsim.algorithms._dl.train --model lstm``) drives both this
predictor and the legacy MLP variant.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True

    class BeamPredictorLSTM(nn.Module):
        """Single-layer LSTM beam predictor.

        Parameters
        ----------
        input_dim:
            Per-time-step input dimension.  We use ``2`` (the (k, l)
            integer pair as a 2-vector); for one-hot or angle-embedded
            inputs the dimension would change.
        hidden_dim:
            LSTM hidden state size.
        output_dim:
            K * L flat-index logit vector.
        n_layers:
            Number of stacked LSTM layers.
        """

        def __init__(
            self,
            input_dim: int = 2,
            hidden_dim: int = 64,
            output_dim: int = 256,
            n_layers: int = 1,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden_dim, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run LSTM over a window and map the final hidden state to logits.

            Parameters
            ----------
            x:
                ``(batch, window, input_dim)`` float tensor.

            Returns
            -------
            ``(batch, output_dim)`` logits.
            """
            out, _ = self.lstm(x)
            # Use the last time step's hidden representation.
            last = out[:, -1, :]
            return self.head(last)

except ImportError:
    TORCH_AVAILABLE = False

    class BeamPredictorLSTM:  # type: ignore[no-redef]
        """Stub — torch not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "torch is required for BeamPredictorLSTM. "
                "Install with: pip install -e .[dl]"
            )
