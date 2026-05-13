# Contributing to qufin

Thanks for considering a contribution to qufin. Whether it's a bug fix, new algorithm, or documentation improvement, we appreciate the help.

## Setup

```bash
# Fork and clone
git clone https://github.com/<your-username>/qufin.git
cd qufin

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Development workflow

1. Create a feature branch from `master`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes. Follow the [code style](#code-style) guidelines below.

3. Run the checks:
   ```bash
   ruff check src/ tests/        # Lint
   ruff format --check src/ tests/  # Format check
   mypy src/qufin/               # Type check
   pytest -m "not hardware and not slow"  # Fast test suite
   ```

4. Push and open a PR against `master`.

## Code style

| Rule | Detail |
|---|---|
| **Formatter / linter** | ruff (config in `pyproject.toml`) |
| **Line length** | 100 characters |
| **Variable naming** | Finance/math conventions are fine: `S` (spot), `K` (strike), `T` (maturity), `sigma`, `mu`. Use descriptive names for everything else. |
| **Docstrings** | Google style. Required for all public functions and classes. |
| **Type hints** | Required for all public function signatures. |
| **Imports** | Sorted by ruff (`I` rules). Use lazy imports for optional dependencies. |

## Project structure

Every quantum algorithm in qufin ships alongside a classical baseline for the same problem. When adding a new algorithm:

```
src/qufin/<module>/
    classical/       # Classical implementation
    quantum_algo.py  # Quantum implementation
```

- Classical and quantum implementations should solve the **same problem** with the **same interface** where possible.
- Add tests in `tests/unit/` (fast, no simulator needed) and `tests/integration/` (Qiskit Aer required).
- Use `MockBackend` for unit tests so they run without a simulator.

## Testing

```bash
pytest                              # Full suite
pytest tests/unit/                  # Unit tests only
pytest tests/integration/           # Integration tests (needs Qiskit Aer)
pytest tests/property/              # Property-based tests (Hypothesis)
pytest tests/stress/                # Stress tests
pytest -m "not hardware and not slow"  # CI-friendly fast suite
pytest -k "test_black_scholes"      # Run specific tests
```

**Test requirements:**
- New features must include tests
- Bug fixes should include a regression test
- Aim for coverage on both happy paths and edge cases
- Use `@pytest.mark.slow` for tests taking >5s
- Use `@pytest.mark.hardware` for tests requiring IBM credentials

## Pull request checklist

Before submitting:

- [ ] Tests pass (`pytest -m "not hardware and not slow"`)
- [ ] No lint errors (`ruff check src/ tests/`)
- [ ] Code is formatted (`ruff format --check src/ tests/`)
- [ ] Type checks pass on modified modules (`mypy`)
- [ ] New code has tests
- [ ] Public APIs have docstrings
- [ ] `CHANGELOG.md` updated (if user-facing change)

## Reporting issues

Use the GitHub issue templates:
- **Bug reports**: Include a minimal reproduction, Python/qufin/Qiskit versions, and full traceback
- **Feature requests**: Describe the use case and reference relevant papers or algorithms if applicable

## Questions?

Open a [GitHub Discussion](https://github.com/anonymousAAK/qufin/discussions) for questions about architecture, design decisions, or "is this a good idea?" conversations.
