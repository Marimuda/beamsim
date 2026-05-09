# beamsim — developer command surface
# All targets run from the repository root and assume an activated virtualenv.

.DEFAULT_GOAL := help
.PHONY: help install install-dl install-dev format format-check lint type test test-fast \
        cov check docs docs-serve build clean hooks

PY ?= python
PIP ?= $(PY) -m pip
PKG := beamsim
SRC := src/$(PKG)

help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install the package in editable mode with dev extras.
	$(PIP) install -e ".[dev]"

install-dl:  ## Add the deep-learning extras (torch).
	$(PIP) install -e ".[dev,dl]"

format:  ## Auto-format the codebase with ruff.
	ruff format $(SRC) tests experiments

format-check:  ## Check formatting without modifying files (CI gate).
	ruff format --check $(SRC) tests experiments

lint:  ## Run ruff lint with auto-fix.
	ruff check --fix $(SRC) tests experiments

lint-check:  ## Run ruff lint without modifications (CI gate).
	ruff check $(SRC) tests experiments

type:  ## Static type-check with mypy.
	mypy $(SRC)

test:  ## Run the full pytest suite.
	pytest -q

test-fast:  ## Run only the fast subset (no slow tests).
	pytest -q -m "not slow"

cov:  ## Run tests with coverage.
	pytest --cov=$(PKG) --cov-report=term-missing --cov-report=xml

check: format-check lint-check type test-fast  ## Quality gate (format + lint + type + fast tests).

docs:  ## Build the MkDocs site into site/.
	mkdocs build --strict

docs-serve:  ## Serve docs locally with live reload.
	mkdocs serve

build:  ## Build sdist + wheel into dist/.
	$(PY) -m build

hooks:  ## Install pre-commit + pre-push git hooks.
	pre-commit install
	pre-commit install --hook-type pre-push

clean:  ## Remove build, cache, and coverage artefacts.
	rm -rf build dist *.egg-info src/*.egg-info site
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis .benchmarks
	rm -f .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
