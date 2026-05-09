---
name: Bug report
about: Report unexpected behaviour, an incorrect result, or a regression
title: "[bug] "
labels: ["bug"]
assignees: []
---

## What happened?

A clear and concise description of the bug.

## Reproducer

The smallest possible script, command, or config that reproduces the issue.

```bash
# e.g.
beamsim-run --config-name rotational run.n_trials=1 run.seed=0
```

## Expected behaviour

What did you expect to happen?

## Observed behaviour

What actually happened? Paste tracebacks / numerical output verbatim.

## Environment

- `beamsim` version (`pip show beamsim` or git commit):
- Python version (`python --version`):
- OS:
- Relevant dependency versions (`pip freeze | grep -E 'numpy|scipy|hydra|torch'`):

## Additional context

Anything else (related issues, recent changes, hardware/cluster setup).
