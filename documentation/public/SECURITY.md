# Security Policy

## Supported Versions

Security fixes are applied to the latest version on the default branch.

## Reporting a Vulnerability

Please do not open public issues for security vulnerabilities.

Report vulnerabilities privately via [GitHub Security Advisories](https://github.com/shovsOS/shovsOS/security/advisories) with:

- A clear description of the issue
- Reproduction steps or proof of concept
- Potential impact
- Suggested mitigation (if known)

You should receive an acknowledgment within 3 business days.
We aim to provide a remediation plan or status update within 7 business days.

## Disclosure Process

- We validate and triage the report
- We develop and test a fix
- We coordinate disclosure timing with the reporter
- We publish a security advisory when appropriate

## Security Best Practices for Contributors

- Never commit real secrets or API keys
- Use .env files locally and .env.example for templates
- Keep dependencies updated and pinned where practical
- Prefer least-privilege tokens for external integrations
