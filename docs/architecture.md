# Architecture

This page explains the module layout and the dependency direction. The
contract is intentionally small: pure stateless physics layers feed a
stateful BPLM bookkeeping object that algorithms consume.

## Module map

```mermaid
flowchart LR
    subgraph io [I/O boundary]
        run[run.py — Hydra entry]
        runner[runner.py — orchestration]
    end

    subgraph core [Pure simulation core]
        geom[geometry.py — Track]
        cb[codebook.py — Codebook]
        ch[channel.py — ChannelRealisation]
        bplm[bplm.py — BPLMState]
        lb[link_budget.py]
        cfg[configs.py — typed config dataclasses]
    end

    subgraph algos [Beam-management policies]
        base[algorithms/base.py — Algorithm protocol]
        impls["exhaustive · NNS · tabu · MCMD · CI ·<br/>angular_prediction · UCB1 · Thompson · HBM ·<br/>OMP · DL-MLP · DL-LSTM · MAMBA · EKF · ..."]
    end

    metrics[metrics.py]
    plots[plotting.py]

    run --> runner
    runner --> base
    base --> impls
    impls --> bplm
    runner --> ch
    runner --> geom
    runner --> cb
    runner --> lb
    runner --> metrics
    metrics --> plots
    cfg --> run
```

## Dependency direction

- `geometry`, `codebook`, `link_budget` have no internal dependencies.
- `channel` depends on `geometry` and `codebook`.
- `bplm` depends on nothing in the package.
- `algorithms/*` depend on `bplm`, `codebook`, and (for some) `channel`.
- `runner` orchestrates every layer above and owns parallelism + I/O.
- `run` (the Hydra entry point) only depends on `runner` and `configs`.

There are **no circular imports** and **no import-time side effects**.

## Data flow per trial

```mermaid
sequenceDiagram
    participant R as runner
    participant T as Track
    participant C as ChannelRealisation
    participant K as Codebook
    participant B as BPLMState
    participant A as Algorithm

    R->>T: build track for trial t
    R->>C: build channel realisation
    R->>K: build codebooks (UE, BS)
    R->>B: BPLMState(K, L)
    loop for each step
        R->>C: H(step)
        A->>B: choose (k, l) given history
        R->>K: w_k, f_l
        R->>B: y = w_k^H · H · f_l + noise
    end
    R-->>R: collect TrialResult
```

## Determinism

Every algorithm receives an `np.random.Generator` via
`runner.run_experiment(..., seed=...)`. The runner derives a per-trial seed
sequence with `numpy.random.SeedSequence` and pairs algorithms via Common
Random Numbers — the channel realisation, the track, and the per-step noise
are bit-identical across algorithms within a single trial. This is what
makes ribbon-plot comparisons between algorithms statistically meaningful.

No code path calls the global `numpy.random` API. No code path reads the
system clock for randomness. Tests assert distinct seeds produce distinct
traces (`tests/test_runner.py::TestTrialResult::test_distinct_seeds_distinct_traces`).

## I/O boundary

- **Reads**: configs from `configs/` (Hydra), optional ML checkpoints from
  `models/`.
- **Writes**: `<output_path>/<name>/<sweep_value>.npz` from
  `runner.save_experiment`. PDFs are written by `experiments/exp_*.py`
  scripts via `plotting.py`.
- No network calls. No `os.environ` reads. No subprocess invocations.

## Where to extend

- **New algorithm**: see [Usage → Adding an algorithm](usage.md#adding-an-algorithm).
- **New scenario**: add a YAML under `configs/scenario/` and reference it in
  a top-level experiment YAML.
- **New metric**: add a function in `metrics.py`. Keep it pure; let
  `plotting.py` call it.
- **New channel kind**: extend `run._build_channel_factory` and add a
  matching `channel_kind` literal in the scenario schema (`configs.py`).
