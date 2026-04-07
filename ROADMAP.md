# Roadmap

This roadmap tracks the path from dual-runtime experimentation to a publish-grade, unified platform.

## Phase 1: Runtime Convergence

Goal:
- keep best legacy behavior while making `run_engine` canonical

Milestones:
1. tool-calling behavior parity and canonical web-tool registration
2. memory commit and verification gating parity
3. frontend-visible phase/log event parity
4. consumer and delegation paths routed through canonical runtime contract
5. managed runtime as profile default, with explicit legacy compatibility mode

## Phase 2: Explainability as Product Surface

Goal:
- make logs and traces understandable for non-experts and auditable for experts

Milestones:
1. canonical event schema with human summaries
2. run summary cards in UI (`what happened`, `why`, `what changed`)
3. memory diff and verification report surfaces
4. replayable run timelines

## Phase 3: Persistent Knowledge Layer

Goal:
- move from per-turn rediscovery to compounding knowledge

Milestones:
1. source layer + compiled wiki layer + schema layer
2. index/log maintenance flows
3. contradiction/staleness lint workflow
4. controlled ingest and query filing back into knowledge artifacts

## Phase 4: OSS Maturity

Goal:
- sustain community scrutiny and contribution quality

Milestones:
1. stable contribution templates and governance
2. release checklist and known-limitations discipline
3. expanded evaluation suites and regression dashboards
4. stronger docs for deployment and operations

## Working Principle

Do not rewrite from scratch. Use staged convergence with measurable parity gates.
