# Contributing to Shovs LLM OS

Thanks for contributing. This project is an active system build, not a static library, so contribution quality is judged by runtime behavior, not only code style.

## Scope

Shovs is a human control center over an autonomous agent runtime. Contributions are especially valuable in:
- runtime convergence (`run_engine` and legacy path parity)
- tool-calling reliability and safety
- trace/log explainability
- memory correctness and state integrity
- Nova and consumer UX clarity
- test and evaluation coverage

## Ground Rules

1. Prefer small, reviewable PRs with one clear objective.
2. Keep claims in docs tied to implemented behavior.
3. Add or update tests for all functional changes.
4. Do not remove existing behavior without migration notes.
5. Preserve owner/session isolation and traceability.

## Development Setup

1. Follow [Setup](documentation/public/SETUP.md).
2. Backend dev: `npm run dev:backend`
3. Nova dev: `npm run dev:frontend:nova`
4. Consumer dev: `npm run dev:frontend:consumer`

## Test Expectations

Run focused tests for touched areas before opening a PR.

Typical backend suite:

```bash
pytest tests/test_context_compiler.py \
       tests/test_run_engine_helpers.py \
       tests/test_run_engine_contract.py \
       tests/test_consumer_routes.py
```

If you change frontends, run build checks:

```bash
cd frontend_nova && npm run build
cd ../frontend_consumer && npm run build
```

## PR Checklist

1. Problem statement is explicit.
2. Behavior before/after is documented.
3. Tests added or updated.
4. Docs updated if user-visible behavior changed.
5. No secrets or local machine artifacts committed.

## Related Docs

- Vision: [documentation/public/VISION.md](documentation/public/VISION.md)
- Developer Guide: [documentation/public/DEVELOPER_GUIDE.md](documentation/public/DEVELOPER_GUIDE.md)
- Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- Governance: [GOVERNANCE.md](GOVERNANCE.md)
