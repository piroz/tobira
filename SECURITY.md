# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.x.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in tobira, please report it responsibly.

### How to Report

1. **Do NOT open a public GitHub issue** for security vulnerabilities.
2. Send an email to the repository maintainer with the following information:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: We will acknowledge your report within 48 hours.
- **Assessment**: We will assess the vulnerability and provide an initial response within 5 business days.
- **Fix**: Critical vulnerabilities will be patched as soon as possible. Non-critical issues will be addressed in the next release cycle.
- **Disclosure**: We will coordinate with you on the disclosure timeline. We aim to disclose vulnerabilities within 90 days of the initial report.

## Security Practices

### Dependency Management

- Dependencies are automatically monitored via Dependabot for known vulnerabilities.
- `pip-audit` (Python) and `npm audit` (JS) are run in CI on every pull request.
- SBOM (Software Bill of Materials) in CycloneDX format is generated and attached to each GitHub Release.

### Code Security

- All pull requests require CI checks to pass before merging.
- Static analysis (ruff, mypy) is enforced in CI.

## Scope

The following are in scope for security reports:

- tobira Python package (`python/tobira/`)
- tobira JS SDK (`js/`)
- MTA integration plugins (`integrations/`)
- Docker deployment configurations (`docker/`)
- CI/CD pipeline security

The following are out of scope:

- Vulnerabilities in upstream dependencies (report these to the respective projects)
- Issues in development-only tooling that do not affect production deployments
