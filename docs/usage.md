# Usage

`beamsim-run` is a thin Hydra adapter over `beamsim.run.run_from_config`.

## Configuration layout

Configs live under `configs/`:

```text
configs/
  rotational.yaml         # top-level experiment composing scenario + sweep + algo
  alpha.yaml
  snr.yaml
  handover.yaml
  bs_coordination.yaml
  scenario/               # scene, BS layout, channel kind, mobility model
    case_a.yaml
    case_b.yaml
    case_c.yaml
    case_d.yaml
  sweep/                  # variable being swept (rpm / alpha / snr_db / none)
    rotational.yaml
    snr.yaml
    alpha.yaml
  algo/                   # algorithm-specific options
    default.yaml
    sota_baselines.yaml
```

Every top-level YAML composes the right scenario, sweep, and algo via
Hydra's `defaults` list.

## Running an experiment

```bash
beamsim-run                                  # default = rotational
beamsim-run --config-name alpha
beamsim-run --config-name snr
beamsim-run --config-name handover
```

Override any field on the CLI:

```bash
beamsim-run --config-name rotational \
    run.n_trials=10 run.n_steps=1000 run.seed=42
```

## Sweep variables

| Variable   | Effect                                                      |
| ---------- | ----------------------------------------------------------- |
| `none`     | Single-point experiment.                                    |
| `rpm`      | Sweeps the UE rotation rate; one `.npz` per RPM.            |
| `alpha`    | Sweeps the measurement rate (`Δt = 1 / (α × 1000 Hz)`).      |
| `snr_db`   | Sweeps target per-element SNR via `tx_amp` calibration.     |

`run.algorithms` is a list of keys from
[`beamsim.algorithms.ALL_ALGORITHMS`](api.md#beamsim.algorithms).

## Output

Each run writes `<output_path>/<name>/<sweep_value>.npz`. The file contains
`TrialResult` arrays stacked along the trial axis:

| Key                      | Shape                          | Meaning                                |
| ------------------------ | ------------------------------ | -------------------------------------- |
| `selected_l`             | `(algorithms, trials, steps)`  | Per-step selected beam index.          |
| `selected_bs`            | `(algorithms, trials, steps)`  | Per-step selected BS (multi-BS only).  |
| `linear_snr`             | `(algorithms, trials, steps)`  | Linear SNR at the selected pair.       |
| `optimal_snr`            | `(trials, steps)`              | Genie-best SNR across the codebook.    |
| `seeds`                  | `(trials,)`                    | Per-trial seeds.                       |
| `metadata` (npz keys)    | scalars                        | `name`, `n_steps`, `dt`, etc.          |

Use `numpy.load` to reload:

```python
data = np.load("outputs/rotational/rotational.npz", allow_pickle=False)
data["linear_snr"].shape  # (algorithms, trials, steps)
```

## Logging

Library code uses the standard `logging` framework with the logger name
`beamsim.<module>`. Hydra's `--info` flag turns on verbose execution traces:

```bash
beamsim-run --config-name rotational --info
```

The CLI itself logs the resolved config, per-sweep progress, and the final
output directory at INFO level.

## Adding an algorithm

1. Implement a new class in `src/beamsim/algorithms/<name>.py` deriving from
   `Algorithm` (see `algorithms/base.py`).
2. Register it in `src/beamsim/algorithms/__init__.py` under `ALL_ALGORITHMS`
   and `__all__`.
3. Add a regression test under `tests/test_algorithms.py` that pins behaviour
   at a fixed seed.
4. Update [SOTA baselines](SOTA_BASELINES.md) if it ships as a baseline.
