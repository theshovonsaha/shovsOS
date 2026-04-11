# Contributing to Shovs LLM OS

Thank you for your interest in contributing.

Shovs LLM OS is an open-source Language OS project focused on explicit runtime control, truthful state transitions, and strong local-first execution. Contributions that improve runtime clarity, tool reliability, small-model coherence, traceability, docs, and frontend usability are all valuable.

## How Can I Contribute?

### Reporting Bugs
- Use the GitHub Issue Tracker.
- Provide a clear description and steps to reproduce.

### Suggesting Enhancements
- Open an issue titled "[Enhancement] ..."
- Describe the feature and why it would be useful.

### Pull Requests
1. Fork the repo.
2. Create a new branch (`codex/your-feature` or `feature/your-feature`).
3. Ensure all tests pass (`pytest`).
4. Submit a PR.

## Development Setup
1. Standard installation (see [README](../../README.md) and [Setup](SETUP.md)).
2. Run tests: `pytest`.
3. Keep logic modular: add new LLM providers to `llm/` and new tools to `plugins/`.

## Coding Standards
- Use type hints wherever possible.
- Functional changes must include a corresponding test in `tests/`.
- Document new tools clearly in their docstrings.
- Keep the runtime honest: do not let the system claim work it did not actually complete.
