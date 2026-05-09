# Contributing to qufin

Thanks for considering a contribution! By participating you agree to the
Contributor Covenant 2.1 Code of Conduct.

## Quickstart

```bash
git clone https://github.com/qufinance/qufin && cd qufin
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,all]"
pre-commit install
pytest -m "not hardware"
```

## Sign your commits (DCO)

We use the Developer Certificate of Origin. Use `git commit -s`.

## Style

- `ruff`, `black`, `mypy --strict` enforced via pre-commit.
- Google-style docstrings.
- One topic per PR.

## Tests

Every PR: unit tests; if you touch `algorithms/`, add a property-based test.
Regression tests in `tests/regression/` must pass; if you change a tolerance,
explain why in the PR.

## Good first issues

Filter the issue tracker by `good first issue`. Comment to claim.

## Release process

Maintainers cut tags `vMAJOR.MINOR.PATCH` after CHANGELOG update; CI does the rest.
