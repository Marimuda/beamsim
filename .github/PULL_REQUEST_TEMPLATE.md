## Summary

<!-- Imperative one-liner: what does this PR change and why? -->

## Type of change

- [ ] Bug fix
- [ ] New algorithm / scenario / metric
- [ ] Refactor (no behavioural change)
- [ ] Documentation
- [ ] Tooling / CI / dev experience

## Related issue

Closes # <!-- or "Refs #" if not closing -->

## Testing

- [ ] `make check` passes locally.
- [ ] New / changed behaviour is covered by a test.
- [ ] Determinism preserved under common random numbers (no global RNG calls,
      seeds are accepted explicitly).
- [ ] Slow tests marked with `@pytest.mark.slow` if Monte-Carlo over many trials.

## Documentation

- [ ] `CHANGELOG.md` updated under **Unreleased**.
- [ ] `docs/` updated for user-visible changes.
- [ ] Public docstrings updated.

## Reviewer checklist

- [ ] Public API additions are exported via `__all__`.
- [ ] Logging used instead of `print` in library code.
- [ ] No new hidden filesystem or network I/O.
