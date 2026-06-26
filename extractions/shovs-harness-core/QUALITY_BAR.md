# Quality Bar

This project is good only if it catches failures that normal chat traces hide.

## Minimum Passing Signals

| Signal | Pass condition |
| --- | --- |
| Tool truth | A result without a known tool call is rejected |
| Claim truth | A final answer cannot cite failed or missing result IDs |
| Source contract | "top N ... M URLs each" becomes `N * M` required fetches |
| Topic agnostic | Stocks, sushi, laptops, and papers compile to the same source workflow shape |
| Drift detection | A run that locks entities A/B/C and searches D fails |
| Continuation | The kernel keeps acting until required evidence exists |
| Small context | Phase attention ranks missing/risky items above stale completed items |

## What Counts As A Real Improvement

Do not compare prose quality first. Compare run behavior:

1. Did the plain loop stop early?
2. Did it invent or drift entities?
3. Did it cite tools that did not succeed?
4. Did it fetch the required number of sources?
5. Did the trace explain why the run continued or stopped?

If the harness improves those numbers on the same model, it is useful.

## What Does Not Count

- A prettier final answer.
- A longer plan.
- A trace full of labels but no invariants.
- A memory write with no provenance.
- A benchmark result that cannot replay the underlying run.
