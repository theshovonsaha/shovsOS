# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- Root-level OSS governance and policy docs:
  - `ARCHITECTURE.md`
  - `GOVERNANCE.md`
  - `ROADMAP.md`
  - `EVALUATION.md`
  - `CONTRIBUTING.md`
  - `CODE_OF_CONDUCT.md`
  - `SECURITY.md`
  - `SUPPORT.md`
- GitHub collaboration templates:
  - bug report template
  - feature request template
  - issue template config
  - pull request template

### Changed

- README reframed around human control center + autonomous agent body model.
- README public doc links fixed and expanded for OSS discoverability.
- Runtime profile defaults now converge to `runtime_kind=managed` (with explicit alias normalization for legacy/native compatibility values).
- Agent manager runtime fallback now resolves to managed by default when runtime kind is missing.
- CI workflow corrected to build real frontends (`frontend_shovs`, `frontend_consumer`).
- CI backend smoke tests now run on every PR; full backend suite remains optional behind `ENABLE_BACKEND_CI_TESTS=true`.
- Consumer frontend build strictness fixed (`unused state variables` in `frontend_consumer/src/App.tsx`).
- Public architecture/setup/developer docs now describe managed canonical runtime plus env-gated legacy compatibility mode.
