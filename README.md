# beamsim — mmWave beam-alignment simulator

[![CI](https://github.com/jakupsv/beamsim/actions/workflows/ci.yml/badge.svg)](https://github.com/jakupsv/beamsim/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.10–3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type-checked: mypy](https://img.shields.io/badge/types-mypy-2a6db2.svg)](https://mypy-lang.org/)

`beamsim` provides a reproducible implementation of geometry-based mmWave
beam-alignment simulation and baseline comparison, grounded in the original
MSc thesis model (*Beam alignment methods for terminals in millimeter-wave
wireless networks*, Aalborg, 2018) and extended with modern reproducibility
and evaluation practices.

The repository is positioned within the broader **beam-management** problem
that 5G NR / 5G-Advanced has crystallised — see
[`docs/related_work.md`](docs/related_work.md) for how the field has evolved
from 2018-era beam alignment to the modern beam-management lifecycle, what
this simulator does cover, and what is deliberately out of scope (RIS,
near-field / sub-THz focusing, multi-TRP coordination, multimodal sensing,
ray-traced or measured channels).

## Status

**Active research code, version `0.1.0`.** Public API may change in minor
releases until the journal-paper figures and tables are locked.

## Intended users

Researchers and graduate students working on mmWave beam management who want
a small, reproducible, well-typed reference implementation of NNS, tabu,
angular-prediction, context-information, and MCMD policies — plus modern
SOTA baselines (Thompson, UCB1, HBM, OMP, DL-MLP, DL-LSTM, MAMBA, EKF,
PositionMAB, BAI) — under a faithful TR 38.901 cluster-delay-line channel,
all paired across algorithms by common random numbers for statistically
meaningful comparison.

## Scope

This is a **codebook-based beam-alignment simulator with mobility, blockage,
and reproducible baseline comparison**. It covers:

- 28 GHz, 100 MHz bandwidth, UMi (default) / UMa (selected runs);
- ULA: 4-element UE / 16-element BS, cosine-spaced linear-phase codebooks
  of size 8 and 32;
- azimuth-only, single polarisation, no random ray-coupling;
- 3GPP TR 38.901 cluster-delay-line generator simplified to 12 clusters /
  20 sub-rays per cluster, LOS, without spatial-consistency procedure;
- mobility tracks (rotation, straight-line) and Model A blockage;
- single-BS and multi-BS handover scenarios;
- common-random-numbers Monte Carlo orchestration (`runner.py`) so each
  algorithm sees an identical channel/noise sequence per trial;
- 3GPP TR 38.843–aligned beam-management metrics where applicable (Phase 4C);
- algorithm zoo: exhaustive, NNS, tabu (with aspiration), angular
  prediction, context information, MCMD, plus the SOTA baselines listed
  above (see [`docs/SOTA_BASELINES.md`](docs/SOTA_BASELINES.md)).

It deliberately does **not** cover RIS, near-field / sub-THz beam focusing,
multi-TRP joint transmission, multimodal sensing (camera / LiDAR / radar),
ray-traced or measured channels, or the full 3GPP beam-failure-recovery and
measurement-reporting signalling pipeline.
[`docs/related_work.md`](docs/related_work.md) explains why and points at
the projects that do.

## Methodological commitments

Beyond the algorithm zoo, `beamsim` is built around three evaluation
commitments that we argue are what makes a beam-management comparison
defensible:

- **Common random numbers (CRN) across algorithms.** Within a trial,
  the channel realisation, the mobility track, and the per-(k, l)
  noise sample are bit-identical across every algorithm under
  comparison. This is a *paired* evaluation: differences between
  algorithms come from policy choices, not from one algorithm getting
  an easier scenario than another. See `runner.py` for the seed-stream
  factoring (channel / track / per-algo noise) and
  [`docs/architecture.md`](docs/architecture.md) for the full
  determinism contract.
- **Codebook-oracle regret as a first-class diagnostic.** Raw SNR
  conflates scenario difficulty with policy quality.
  `metrics.oracle_snr_db` returns the strongest SNR achievable on the
  same codebook and channel; `metrics.snr_regret_db` returns the
  per-step gap. Two policies with similar mean SNR can have very
  different regret profiles, and the regret is what you should
  publish.
- **Overhead, switching, and outage as named metrics.** Modern
  evaluation reports a *budget*, not just a quality.
  [`docs/SOTA_BASELINES.md`](docs/SOTA_BASELINES.md) lists the metric
  surface — `probing_overhead`, `beam_switch_rate`,
  `outage_probability`, `top_k_accuracy`, `time_to_realign` — that
  papers built on `beamsim` are encouraged to adopt.

## Supported environments

- **OS**: Linux, macOS. Windows is not actively tested.
- **Python**: 3.10, 3.11, 3.12.
- Optional `[dl]` extras require PyTorch 2.2+.

## Installation

```bash
git clone https://github.com/jakupsv/beamsim.git
cd beamsim
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # includes test, docs, ruff, mypy, pre-commit
```

For the deep-learning baselines:

```bash
pip install -e ".[dev,dl]"
```

See [`docs/installation.md`](docs/installation.md) for the optional-extras matrix.

## Quickstart

```bash
beamsim-run --config-name rotational run.n_trials=2 run.n_steps=100
```

…or from Python:

```python
from beamsim import Experiment, run_experiment, save_experiment
```

A complete runnable script is in
[`examples/minimal_example.py`](examples/minimal_example.py); it finishes in
under two seconds and writes a tiny `.npz` archive.

## Layout

```
src/beamsim/         package source
  geometry.py        UE/BS positions, mobility tracks
  codebook.py        cosine-spaced linear-phase ULA codebooks
  channel.py         simplified TR 38.901 cluster-delay-line generator
  bplm.py            BPLM state, single-entry measurement, OBP rule
  algorithms/        MBP policies (one file each)
  metrics.py         coverage rate, mean SNR, L_BS
  runner.py          Monte Carlo orchestration with CRN pairing
  plotting.py        figures with 95 % bootstrap CI ribbons
  run.py             Hydra entry point (beamsim-run)
configs/             YAML experiment definitions
experiments/         legacy runnable scripts (one per figure)
examples/            minimal end-to-end example
docs/                MkDocs site sources
results/             simulation outputs (.npz, .pdf) — gitignored
tests/               pytest suite (185 tests, ~ 90 s full run)
```

## Development commands

| Target              | What it does                                              |
| ------------------- | --------------------------------------------------------- |
| `make install`      | Editable install with `[dev]` extras.                     |
| `make hooks`        | Install `pre-commit` and `pre-push` git hooks.            |
| `make format`       | Auto-format with ruff.                                    |
| `make lint`         | Lint with ruff (`--fix`).                                 |
| `make type`         | Type-check with mypy.                                     |
| `make test`         | Full pytest suite (slow tests included).                  |
| `make test-fast`    | Only the fast subset (no `@pytest.mark.slow`).            |
| `make cov`          | Tests with coverage reporting.                            |
| `make check`        | format-check + lint-check + type + fast tests (CI gate).  |
| `make docs`         | Build the MkDocs site (`mkdocs build --strict`).          |
| `make build`        | Build sdist + wheel.                                      |
| `make clean`        | Remove caches and build artefacts.                        |

`make help` lists every target.

## Testing

```bash
make test            # full suite
make test-fast       # skip @pytest.mark.slow tests
```

Tests are deterministic and use `tmp_path` for any filesystem I/O. CI runs
the full suite across Python 3.10/3.11/3.12 on Linux and macOS.

## Documentation

Browse the docs site after `make docs && python -m http.server -d site`,
or read the sources directly:

- [Installation](docs/installation.md)
- [Quickstart](docs/quickstart.md)
- [Usage](docs/usage.md)
- [Architecture](docs/architecture.md)
- [Development guide](docs/development.md)
- [SOTA baselines reference card](docs/SOTA_BASELINES.md)
- [Related work](docs/related_work.md) — where `beamsim` sits in the modern
  beam-management literature
- [API reference](docs/api.md) (auto-generated via mkdocstrings)

## Citation

If you use `beamsim` in published work, please cite via the
[`CITATION.cff`](CITATION.cff) metadata or use GitHub's "Cite this
repository" button. A DOI will be minted on the first tagged release.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). All contributions are released
under the [MIT license](LICENSE).

Bug reports and security disclosures: see [`SECURITY.md`](SECURITY.md).

## License

[MIT](LICENSE) © 2026 jakupsv.
