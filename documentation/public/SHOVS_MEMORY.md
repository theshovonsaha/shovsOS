# shovs-memory

`shovs-memory` is the deterministic fact + correction layer extracted from the
Shovs runtime, packaged for use inside other agent loops.

It is not a general-purpose agent memory system. It is a narrow guarantee:
explicit user statements get extracted by rule (not by LLM judgment), older
facts get voided when new ones supersede them, and unverified guesses are
quarantined as "candidates" instead of graduating into trusted state.

## Honest positioning vs mempalace

Before anything else: if you are looking for a broad, associative,
LLM-driven memory layer with rich narrative recall and a spatial/loci
metaphor — **use mempalace**. It is broader, more mature, and not tied to a
specific runtime. `shovs-memory` is not a replacement for it.

`shovs-memory` makes a different trade. It is narrower, more opinionated, and
exposes only what the Shovs runtime needs to keep facts honest:

| | mempalace | shovs-memory |
|---|---|---|
| Scope | broad agent memory model | narrow fact + correction layer |
| Recall | LLM-driven, associative | deterministic + semantic graph |
| Fact extraction | LLM judgment | rule-based, predicate-typed |
| Corrections | newer notes coexist with older | older facts voided as state transitions |
| Candidate vs trusted | not a primitive | first-class lane |
| Portability | agent-framework agnostic | wraps Shovs `SessionManager` + `SemanticGraph` |

If you want a memory system that *guesses well*, mempalace is the better
choice. If you want a memory system that *refuses to guess*, this is that.

## What it actually gives you

Three things that are not common in markdown-log or vector-only memory layers:

1. **Deterministic fact extraction.** A fixed set of predicates
   (`preferred_name`, `location`, `timezone`, `preferred_editor`,
   `package_manager`, `primary_language`, `environment_mode`,
   `scope_boundary`, `budget_limit`, `task_constraint`,
   `followup_directive`) are extracted by rule from explicit user statements.
   No LLM is asked to "decide what to remember." If the user did not say it
   plainly, it does not become a fact.

2. **Temporal invalidation.** Corrections are state transitions, not new
   notes appended next to old ones. "Actually, I moved to Berlin" voids the
   previous `location` fact and writes the new one. The timeline preserves
   both for audit, but `current_facts()` returns only the live one.

3. **Candidate-signal lane.** Compression-side paraphrases, model
   inferences, and other not-yet-verified claims are routed into a
   *candidate* lane instead of being committed as trusted facts. The
   inspector view surfaces both lanes separately so you can see what the
   system *believes* vs what it has merely *heard*.

## Install

```bash
pip install shovs-memory
```

## Use

```python
from orchestration.session_manager import SessionManager
from shovs_memory import ShovsMemory

sessions = SessionManager()
memory = ShovsMemory(
    session_id="user-123",
    owner_id="owner-123",
    session_manager=sessions,
)

memory.apply_user_message("My name is Shovon and I live in Toronto.", turn=1)
memory.apply_user_message("Actually, I moved to Berlin.", turn=2)

facts = memory.current_facts()       # [(User, preferred_name, Shovon), (User, location, Berlin)]
timeline = memory.fact_timeline()    # current + superseded, ordered by turn
inspection = memory.inspect()        # trusted + candidates + decision signals
```

## API surface

- `apply_user_message(message, turn)` — runs the deterministic extractors
  against a user turn, writes new facts, voids superseded ones, and routes
  any speculative paraphrases into the candidate lane.
- `store_fact(subject, predicate, object, *, supersede=False)` — direct
  insertion when you already know the fact (useful for migrating an existing
  profile into the graph).
- `retrieve(query, *, top_k=5)` — semantic retrieval over the memory graph
  (deterministic facts + candidates + indexed compressed history).
- `current_facts()` — only live, non-superseded facts.
- `fact_timeline()` — full lineage including superseded entries.
- `inspect()` — structured view: trusted facts, superseded facts, candidate
  signals, context preview, and recent memory decision signals when running
  inside the full Shovs runtime.

## Honest limits

- **Tied to Shovs runtime primitives.** `ShovsMemory` wraps
  `orchestration.session_manager.SessionManager` and
  `memory.semantic_graph.SemanticGraph`. You inherit those data models
  whether you want them or not. If you have your own session/graph layer,
  this package will not slot in cleanly.
- **Predicate set is fixed.** The deterministic extractors only cover the
  predicates listed above. Domain-specific predicates require either
  `store_fact(...)` (you classify, the layer stores) or extending the
  extractor module.
- **No narrative summarization.** This package will not produce a "what
  happened in this session" paragraph. That is a separate concern —
  intentionally outside its scope.
- **English-language extractor patterns.** Other languages will fall through
  to the candidate lane rather than producing structured facts.

## When to pick this

Pick `shovs-memory` when you want:

- hard guarantees about what counts as "remembered"
- corrections that actually invalidate, not just shadow
- a clear separation between trusted state and unverified guesses
- an inspectable view you can show a user or auditor

Pick something else when you want:

- broad associative recall without strict typing → mempalace
- runtime-agnostic memory you can drop into any framework → mempalace, mem0,
  or a vector store with your own schema
- pure narrative summaries of past sessions → a markdown log
