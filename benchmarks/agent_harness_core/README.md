# Agent Harness Core Benchmark

This benchmark checks the smallest credibility surface of ShovsOS:

- source collection does not drift
- final answers do not leak tool JSON
- tool results cannot be orphaned
- memory replacement is rollback-safe

Run it with:

```bash
venv/bin/python -m pytest tests/test_agent_harness_core_benchmarks.py -q
```

The test writes a machine-readable benchmark report to its temporary pytest directory. The static example result in this folder shows the expected shape.

