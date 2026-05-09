"""Offline trainer for BeamPredictorMLP.

Generates training data from Case A UMi 10 m/s straight-line trajectories,
using Exhaustive as the oracle teacher.  Trains a sliding-window MLP
classifier for 50 epochs and saves the checkpoint.

Usage
-----
    python -m beamsim.algorithms._dl.train [--epochs 50] [--output models/beam_predictor.pt]

Reference (paraphrased):
    Klautau et al. (2018), Heng et al. (2021): offline beam-prediction
    baselines use recent OBP history as input features for a classification
    network trained over exhaustive-sweep labels.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("beamsim.dl.train")

# --- constants ---------------------------------------------------------------
WINDOW = 4  # number of past OBPs used as input
K_UE = 8  # UE codebook beams
L_BS = 32  # BS codebook beams
OUTPUT_DIM = K_UE * L_BS  # 256 classes
INPUT_DIM = 2 * WINDOW  # 8 floats
N_TRAJECTORIES = 50
N_STEPS = 1000
SPEED_MPS = 10.0
DT = 1e-3


def _build_features_labels(
    obp_history: np.ndarray,
    window: int = WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window features and labels from one OBP trajectory.

    Parameters
    ----------
    obp_history:
        Array of shape (T, 2) with (k, l) OBP at each step.
    window:
        History window size.

    Returns
    -------
    X : float32, (T-window, 2*window)
    y : int64,   (T-window,)   flat index = k*L_BS + l
    """
    T = len(obp_history)
    n = T - window
    X = np.zeros((n, 2 * window), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        X[i] = obp_history[i : i + window].ravel().astype(np.float32)
        k_next, l_next = obp_history[i + window]
        y[i] = int(k_next) * L_BS + int(l_next)
    return X, y


def _collect_data(
    n_trajectories: int, n_steps: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Run Exhaustive on Case A UMi 10 m/s and collect OBP trajectories."""
    from beamsim.algorithms.exhaustive import Exhaustive
    from beamsim.bplm import BPLMState
    from beamsim.channel import ChannelParams, ChannelRealisation
    from beamsim.codebook import make_default_bs_codebook, make_default_ue_codebook
    from beamsim.geometry import straight_line_track

    ue_cb = make_default_ue_codebook()
    bs_cb = make_default_bs_codebook()
    # Case A: BS at origin, UE path at y=150 m (predecessor Section 5.2.2)
    bs_xy = np.array([0.0, 0.0])

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    rng_master = np.random.default_rng(seed)

    for traj in range(n_trajectories):
        rng = rng_master.spawn(1)[0]
        track_rng, ch_rng, noise_rng = rng.spawn(3)

        start_x = float(track_rng.uniform(-100.0, 100.0))
        track = straight_line_track(
            start_xy=(start_x, 150.0),
            heading=0.0,
            speed_mps=SPEED_MPS,
            n_steps=n_steps,
            dt=DT,
        )

        params = ChannelParams(ue_speed_mps=SPEED_MPS)
        ch = ChannelRealisation(
            params=params,
            bs_xy=bs_xy,
            bs_yaw=0.0,
            n_bs_elements=16,
            n_ue_elements=4,
            rng=ch_rng,
        )

        state = BPLMState(ue_codebook=ue_cb, bs_codebook=bs_cb, noise_amplitude=1e-3)
        state.tx_amp = 1.0
        algo = Exhaustive()
        algo.reset(state, {})

        obp_hist = np.zeros((n_steps, 2), dtype=np.int32)
        for m in range(n_steps):
            ue_xy = track.positions[m]
            ue_yaw = float(track.orientations[m])
            H = ch.channel_matrix(ue_xy, ue_yaw, m * DT)
            k, l = algo.select_next_mbp(state, m, {})
            state.measure(k, l, H, m, noise_rng)
            k_obp, l_obp = state.obp()
            obp_hist[m] = [k_obp, l_obp]

        X, y = _build_features_labels(obp_hist, window=WINDOW)
        all_X.append(X)
        all_y.append(y)

        if (traj + 1) % 10 == 0:
            logger.info("Collected trajectory %d/%d", traj + 1, n_trajectories)

    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


def train(
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    output_path: Path = Path("models/beam_predictor.pt"),
    seed: int = 42,
) -> float:
    """Train BeamPredictorMLP and save checkpoint.  Returns test accuracy."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.error("torch not installed — run: pip install -e .[dl]")
        sys.exit(1)

    try:
        from tqdm import tqdm

        USE_TQDM = True
    except ImportError:
        USE_TQDM = False

    from beamsim.algorithms._dl.mlp_predictor import BeamPredictorMLP

    logger.info("Collecting training data (%d trajectories x %d steps)...", N_TRAJECTORIES, N_STEPS)
    X_all, y_all = _collect_data(N_TRAJECTORIES, N_STEPS, seed=seed)

    # Train/test split (80/20)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_all))
    split = int(0.8 * len(idx))
    X_tr, y_tr = X_all[idx[:split]], y_all[idx[:split]]
    X_te, y_te = X_all[idx[split:]], y_all[idx[split:]]

    logger.info("Training samples: %d  |  Test samples: %d", len(X_tr), len(X_te))

    device = torch.device("cpu")
    model = BeamPredictorMLP(INPUT_DIM, OUTPUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    ds_tr = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
    )
    loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

    epoch_iter = range(1, epochs + 1)
    if USE_TQDM:
        epoch_iter = tqdm(epoch_iter, desc="Training", unit="epoch")

    for epoch in epoch_iter:
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        avg_loss = total_loss / len(X_tr)
        if USE_TQDM and isinstance(epoch_iter, tqdm):
            epoch_iter.set_postfix(loss=f"{avg_loss:.4f}")
        elif epoch % 10 == 0:
            logger.info("Epoch %d/%d  loss=%.4f", epoch, epochs, avg_loss)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_te, dtype=torch.float32))
        preds = logits.argmax(dim=1).numpy()
    acc = float((preds == y_te).mean())
    logger.info("Test accuracy: %.2f%%", acc * 100)

    # Save checkpoint
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": INPUT_DIM,
            "output_dim": OUTPUT_DIM,
            "K": K_UE,
            "L": L_BS,
            "window": WINDOW,
        },
        output_path,
    )
    logger.info("Checkpoint saved to %s", output_path)
    print(f"Final test accuracy: {acc * 100:.2f}%")
    return acc


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Train BeamPredictorMLP (DL beam prediction baseline)"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("models/beam_predictor.pt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trajectories", type=int, default=N_TRAJECTORIES)
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
