# Results

This is a local validation snapshot, not a certification or broad benchmark claim. It should be updated when the benchmark suite or branch state changes.

## Recorded Local Snapshot

Date recorded: 2026-06-19

| Check | Result |
| --- | --- |
| Full Python test suite | `450 passed, 20 skipped` on the recorded local branch state |
| Response guard tests | Passed |
| Runtime E2E diagnostics | Passed |
| Shovs memory tests | Passed |
| Live Ollama trivial chat smoke | Passed |
| Live Ollama memory recall smoke | Passed |
| Live Ollama stock workflow trace smoke | Passed |

## Important Detail

Raw trace logs can retain the original model token for audit. The UI/runtime also receives retraction and clean replacement events when the response guard changes output.

For small local models, the managed runtime buffers final response text before showing it to the user in the tested managed path. The goal is to prevent raw tool JSON from flashing in chat; before citing this publicly, rerun the response-guard tests on the current branch.

## Baseline Core Results

See [benchmarks/agent_harness_core/baseline_results.example.json](benchmarks/agent_harness_core/baseline_results.example.json).

That file is an example result shape, not a substitute for running the tests.
