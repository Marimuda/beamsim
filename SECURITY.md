# Security Policy

## Scope

`beamsim` is a research-grade simulator. It does not:

- make network calls,
- read user-controlled input over IPC or untrusted channels,
- handle credentials, tokens, or PII,
- execute user-supplied code at runtime.

The primary security surface is therefore the dependency tree
(`numpy`, `scipy`, `matplotlib`, `hydra-core`, `omegaconf`,
`torch` (optional)) and any input file the user passes to the CLI.

## Supported versions

| Version  | Supported |
| -------- | --------- |
| `0.1.x`  | ✅        |
| `< 0.1`  | ❌        |

## Reporting a vulnerability

Please **do not open a public issue** for vulnerabilities.

Email **jakupsv@setur.fo** with:

- a description of the issue,
- a minimal reproducer (input file, command, expected vs observed),
- the affected version (`pip show beamsim` or git commit),
- your platform and Python version.

You should expect an acknowledgement within seven days. Once the issue is
confirmed, a fix and disclosure timeline will be agreed before a public
advisory is filed.

## Hardening commitments

- `pre-commit` runs `detect-private-key` and `check-added-large-files` on
  every commit.
- CI enforces `ruff`, `mypy`, and the full test suite on every PR.
- Dependabot is enabled for both Python deps and GitHub Actions.
- Releases are tagged in Git; build artefacts are reproducible from the tag.
