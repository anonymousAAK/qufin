# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x (latest) | Yes |

## Reporting a vulnerability

If you discover a security vulnerability in qufin, please report it responsibly:

1. **Do not** open a public GitHub issue.
2. Email **adarshkeshri027@gmail.com** with the subject line `[qufin security]`.
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Impact assessment (what can an attacker do?)
   - Suggested fix (if you have one)
4. You will receive an acknowledgment within **48 hours** and a fix timeline within **7 days**.

## Security practices

qufin follows these security principles:

- **No secrets in source code** -- verified via `bandit` and `ruff` in CI
- **Dependency scanning** -- `pip-audit` runs in CI to catch known vulnerabilities
- **No network calls in core algorithms** -- data fetching (Yahoo Finance, FRED) is explicit and opt-in; all quantum and classical algorithms work entirely offline
- **Input validation at API boundaries** -- user-facing functions validate parameters before processing
- **No dangerous builtins** -- no `eval()`, `exec()`, or `pickle.loads()` on untrusted data
- **Pinned dependency ranges** -- lower bounds enforced, upper bounds only where necessary for compatibility
