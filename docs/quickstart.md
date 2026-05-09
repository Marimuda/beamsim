# Quickstart

A minimal end-to-end run that finishes in under a minute on a laptop.

## 1. Install

```bash
pip install -e ".[dev]"
```

See [Installation](installation.md) for details.

## 2. Run a tiny experiment

```bash
beamsim-run --config-name rotational run.n_trials=2 run.n_steps=100
```

Outputs land under `outputs/rotational/`:

- `rotational.npz` — every per-trial trace (selected indices, link budget,
  per-step SNR, etc.).
- `.hydra/config.yaml` — the resolved Hydra config used for the run.

## 3. Re-run from Python

```python
from pathlib import Path

import numpy as np

from beamsim import (
    Codebook,
    Experiment,
    FreeSpaceLosChannel,
    Track,
    run_experiment,
    save_experiment,
)

# Build a 0.5 s rotation track at 60 RPM, 1 ms step.
def track_factory(rng: np.random.Generator) -> Track:
    from beamsim.geometry import rotation_track
    return rotation_track(
        position_xy=(0.0, 0.0),
        rpm=60.0,
        n_steps=500,
        dt=1e-3,
        initial_orientation=0.0,
    )

def channel_factory(rng: np.random.Generator, bs_index: int) -> FreeSpaceLosChannel:
    return FreeSpaceLosChannel(
        bs_xy=np.array([100.0, 0.0]),
        bs_yaw=0.0,
        n_bs_elements=16,
        n_ue_elements=4,
    )

exp = Experiment(
    name="rotational_small",
    n_steps=500,
    dt=1e-3,
    n_trials=4,
    algorithms=["exhaustive", "nns", "mcmd"],
    bs_positions=[(100.0, 0.0)],
    bs_yaws=[0.0],
    track_factory=track_factory,
    channel_factory=channel_factory,
    noise_amplitude=1e-3,
    tx_amp=1.0,
    seed=0,
)

result = run_experiment(exp, n_workers=1, progress=True)
save_experiment(result, Path("outputs/rotational_small.npz"))
```

For a runnable version of the same script see
[`examples/minimal_example.py`](https://github.com/jakupsv/beamsim/blob/main/examples/minimal_example.py).

## 4. Browse the figures

Each `experiments/exp_*.py` script renders the publication figures from a
fresh `.npz`. Run any of them and inspect `results/<experiment>.pdf`.

## Next

- [Usage](usage.md) — Hydra config groups, sweeps, and how to add an algorithm.
- [Architecture](architecture.md) — module map and data flow.
- [API reference](api.md).
