# Governance

This project is currently maintainer-led and evolving toward broader OSS collaboration.

## Roles

1. Maintainer:
- final decision authority on architecture and release
- security triage owner
- roadmap prioritization owner

2. Contributors:
- submit issues and pull requests
- propose design changes with tests and docs
- help improve runtime reliability and explainability

## Decision Policy

1. Runtime correctness and safety override stylistic preferences.
2. Breaking changes require migration notes.
3. Claims in docs must match observable behavior.
4. Significant architectural changes should include rationale in PR description.

## Release Quality Bar

A release should include:
1. passing CI and focused runtime tests
2. no known high-severity unresolved regressions
3. updated docs for behavior changes
4. clear known limitations list

## Conflict Resolution

For technical disagreements:
1. prefer evidence (tests, traces, reproductions) over opinion
2. prefer smaller reversible changes over large speculative rewrites
3. default to incremental strangler-style convergence

## Conduct and Security

- Conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Security: [SECURITY.md](SECURITY.md)
