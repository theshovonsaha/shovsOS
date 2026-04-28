# Evaluation and Evidence Policy

This project makes runtime-level claims. Those claims must be testable.

## What We Evaluate

1. Runtime correctness:

- phase transitions
- tool-call handling
- verification gating
- memory commit behavior

2. State integrity:

- deterministic fact correctness
- candidate signal containment
- correction/supersession behavior

3. UX explainability:

- trace coverage for critical decisions
- consistency of run summary and observable events

## Evidence Hierarchy

1. Reproducible tests.
2. Trace and run-store artifacts.
3. Manual UI validation notes.
4. Narrative claims in docs.

If narrative and tests conflict, tests and traces are source of truth.

## Baseline Test Surfaces

Backend examples:

- `tests/test_run_engine_contract.py`
- `tests/test_run_engine_helpers.py`
- `tests/test_context_compiler.py`
- `tests/test_consumer_routes.py`
- `tests/test_memory_benchmark.py`
- `tests/test_memory_retrieval.py`

Frontend examples:

- `frontend_shovs` build and critical interaction checks
- `frontend_consumer` build and streaming checks

## Release Evidence Checklist

1. Changed behavior has at least one targeted regression test.
2. Critical path events are present in traces.
3. Known limitations are explicitly documented.
4. Runtime convergence regressions are called out before release.

## Memory Benchmark Harness

- API run endpoint: `POST /memory/benchmark/run`
- API latest endpoint: `GET /memory/benchmark/latest`
- CLI runner: `python scripts/memory_benchmark.py --owner-id <id> --save-latest`

Current harness reports:

1. deterministic extraction precision/recall/F1 and void accuracy
2. direct-fact guard accuracy
3. semantic retrieval hit@3 and MRR@3
