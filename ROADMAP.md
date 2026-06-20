# Roadmap

This roadmap tracks the path from a working managed runtime to a publishable, inspectable research prototype.

## Phase 1: Runtime Convergence

Goal:
- preserve useful legacy behavior while making `run_engine` and the run ledger the normal execution story

Milestones:
1. tool-calling behavior parity and canonical web-tool registration
2. memory commit and verification gating parity
3. frontend-visible phase/log event parity
4. consumer and delegation paths routed through canonical runtime contract
5. managed runtime as the default path, with legacy paths treated as compatibility/test substrate

## Phase 2: Explainability as Runtime Surface

Goal:
- make logs and traces understandable for non-experts and auditable for experts

Milestones:
1. canonical event schema with human summaries
2. run summary cards in UI (`what happened`, `why`, `what changed`)
3. memory diff and verification report surfaces
4. replayable run timelines
5. scenario eval cards that show whether the actual tool path matched the user goal

## Phase 3: Persistent Knowledge Layer

Goal:
- move from per-turn rediscovery to compounding knowledge

Milestones:
1. source layer + compiled wiki layer + schema layer
2. index/log maintenance flows
3. contradiction/staleness lint workflow
4. controlled ingest and query filing back into knowledge artifacts
5. memory inspector lanes for trusted facts, candidates, disputes, and conflict traces

## Phase 4: Workflow Contracts and Evals

Goal:
- make important workflows testable by state, not only by final text

Milestones:
1. source-collection scenario evals
2. shopping-advice scenario evals
3. coding-change scenario evals
4. research-report scenario evals
5. continuation/resume scenario evals

## Phase 5: OSS Maturity

Goal:
- sustain community scrutiny and contribution quality

Milestones:
1. stable contribution templates and governance
2. release checklist and known-limitations discipline
3. expanded evaluation suites and regression dashboards
4. stronger docs for deployment and operations

## Working Principle

Do not rewrite from scratch. Use staged convergence with measurable parity gates.
