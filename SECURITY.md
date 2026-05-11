# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.x     | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in qufin, please report it responsibly:

1. **Do not** open a public issue
2. Email: adarsh@example.com with subject "qufin Security Issue"
3. Include: description, reproduction steps, impact assessment
4. We will acknowledge within 48 hours and provide a fix timeline

## Security Practices

- No secrets in source code (verified via `bandit` + `ruff`)
- Dependencies scanned with `pip-audit`
- No network calls in core algorithms (data fetching is explicit and opt-in)
- All user inputs validated at API boundaries
- No `eval()`, `exec()`, or `pickle.loads()` on untrusted data
