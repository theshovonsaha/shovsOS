# Claims

This file separates locally validated claims from active work.

## Validated By Local Tests

| Claim | Evidence |
| --- | --- |
| Tool results must link to known tool calls. | `RunLedger.link_tool_result` rejects orphaned results. |
| Source-collection drift can be detected from trace state. | `SourceCollectionEval` catches forbidden query drift and missing fetch quotas. |
| Small/local final-answer JSON leaks can be guarded in tested paths. | `guard_final_response` converts tool-shaped JSON into clean user-facing text or a safe fallback. |
| Memory fact replacement is rollback-safe. | `SemanticGraph.replace_temporal_facts` keeps the old fact if the replacement fails. |
| The memory inspector reports real current fact counts, not only the last timeline page. | Inspector tests count facts beyond the timeline limit. |

## Supported Architecture Claims

- ShovsOS is best described as an agent harness/runtime.
- The central abstraction is structured run state, not a longer system prompt.
- The UI should inspect ledger events, not raw model text.
- Memory should expose current, superseded, candidate, and disputed state separately.

## Not Claimed Yet

- ShovsOS is not claiming broad benchmark leadership.
- ShovsOS is not claiming that prompts alone solve agent reliability.
- ShovsOS is not claiming full production security certification.
- ShovsOS is not claiming every model/provider behaves the same way.

## What Would Strengthen The Claims

- Public benchmark traces with anonymized run ledgers.
- More live-model evals across local and hosted providers.
- Third-party reproduction instructions.
- UI screenshots tied to the same trace IDs used in tests.
- A small hosted demo that solves one narrow workflow end to end.
