# Contributing to qufin

Thank you for your interest in contributing to qufin. This guide covers the workflow and standards for contributions.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork and create a feature branch:

```bash
git clone https://github.com/<your-username>/qufin.git
cd qufin
git checkout -b feature/your-feature-name
```

3. Install the package in development mode with all extras:

```bash
pip install -e ".[dev]"
```

## Development Workflow

### Running Tests

Run the fast test suite (excludes hardware and slow integration tests):

```bash
pytest -m "not hardware and not slow"
```

Run the full suite including slow tests:

```bash
pytest
```

### Linting and Formatting

Check for lint issues:

```bash
ruff check src/ tests/
```

Auto-format code:

```bash
ruff format src/ tests/
```

### Type Checking

Run mypy on modules that have strict typing enabled:

```bash
mypy src/qufin/
```

## Code Style

- **Formatter/linter**: ruff (configured in `pyproject.toml`)
- **Line length**: 100 characters
- **Variable naming**: Standard finance and mathematics conventions are acceptable for variable names (e.g., `S` for spot price, `K` for strike, `T` for time to maturity, `sigma` for volatility, `mu` for drift). Use descriptive names for everything else.
- **Docstrings**: Google style. All public functions and classes must have docstrings.
- **Type hints**: Required for all public function signatures.

## Pull Request Requirements

Before submitting a PR, verify the following:

- [ ] All tests pass (`pytest -m "not hardware and not slow"`)
- [ ] No lint errors (`ruff check src/ tests/`)
- [ ] Code is formatted (`ruff format --check src/ tests/`)
- [ ] Type checks pass on modified modules (`mypy`)
- [ ] New functionality includes tests
- [ ] Docstrings are present for all public APIs

## Submitting a Pull Request

1. Push your branch to your fork.
2. Open a pull request against the `main` branch of the upstream repository.
3. Fill out the PR template completely.
4. Ensure CI checks pass.

A maintainer will review your PR and may request changes. Please respond to feedback promptly.

## Reporting Issues

Use the GitHub issue templates for bug reports and feature requests. Provide as much detail as possible to help us reproduce and address the issue.
