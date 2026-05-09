# Installation

## Requirements

- Python 3.10, 3.11, or 3.12.
- Linux or macOS. Windows is not actively tested.
- ~1 GB of disk for a working virtualenv with the deep-learning extras.

## From source (recommended for development)

```bash
git clone https://github.com/Marimuda/beamsim.git
cd beamsim
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

This installs the package in editable mode plus `[test]`, `[docs]`, ruff,
mypy, pre-commit, and `build`.

## Optional extras

| Extra   | What it adds                                       | Install                |
| ------- | -------------------------------------------------- | ---------------------- |
| `test`  | pytest, hypothesis, pytest-cov, pytest-benchmark   | `pip install -e ".[test]"`  |
| `docs`  | MkDocs Material + mkdocstrings                     | `pip install -e ".[docs]"`  |
| `dl`    | PyTorch (for `DLPredictor`, `DLLSTMPredictor`)     | `pip install -e ".[dl]"`    |
| `dev`   | All of the above plus ruff / mypy / pre-commit     | `pip install -e ".[dev]"`   |

## Verify the install

```bash
python -c "import beamsim; print(beamsim.__version__)"
beamsim-run --help
```

## Pre-commit hooks (contributors)

```bash
make hooks
```

Installs both the `pre-commit` and `pre-push` hooks. The pre-push hook runs
the fast subset of the test suite.
