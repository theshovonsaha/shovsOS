# shovs-memory

`shovs-memory` is the smallest adoptable surface of the Shovs runtime.

It is not a separate memory engine. It wraps the same primitives the runtime
already uses:

- deterministic user-stated fact extraction
- temporal fact storage with invalidation
- semantic retrieval
- inspectable memory state

## Why This Exists

The full Shovs runtime is powerful, but it is a large first ask.

`shovs-memory` is the wedge:

- use it inside an existing agent loop
- keep your own orchestration
- gain deterministic fact writes, correction handling, and inspectable memory

## Current Shape

Install locally from this repo:

```bash
pip install -e .
```

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

facts = memory.current_facts()
timeline = memory.fact_timeline()
inspection = memory.inspect()
```

## What It Gives You

- `apply_user_message(...)`
  - deterministic extraction for explicit user-stated facts:
    - preferred name
    - location
    - timezone
    - preferred editor
    - package manager
    - primary language
  - correction handling through temporal voiding

- `store_fact(...)`
  - direct fact insertion with optional superseding behavior

- `retrieve(...)`
  - semantic retrieval over stored memory

- `inspect()`
  - trusted current facts
  - superseded facts
  - candidate signals
  - context preview
  - recent memory decision signals

## What It Really Is

`shovs-memory` is a typed, inspectable memory layer for agents.

It is not:

- a generic vector database wrapper
- a markdown log with fuzzy recall
- a second runtime separate from Shovs

It is:

- deterministic fact extraction for explicit user statements
- temporal invalidation when newer facts supersede older ones
- semantic retrieval over the memory graph
- an inspectable state view that separates trusted facts from candidates

## Why It Is Different

Compared with a typical agent memory add-on:

- explicit user facts do not depend on an LLM deciding to emit the right marker
- corrected facts void earlier facts instead of silently coexisting
- compression-side paraphrases can be blocked from hardening into truth
- the memory state is visible: current facts, superseded facts, candidates, and recent decisions

Compared with Markdown-only memory:

- retrieval is not limited to raw file scanning
- memory is typed instead of one undifferentiated text layer
- corrections are first-class state transitions, not just later notes in a log
- the inspectable view is backed by runtime state, not only by narrative summaries

## Positioning

The public story is:

- OpenClaw-style inspectability
- but with typed state, correction handling, and temporal invalidation

That is the difference between “agent remembers text files” and “agent has a
memory model you can inspect and trust.”
