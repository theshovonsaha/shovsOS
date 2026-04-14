"""
ContextEngineV2 — Convergent Graph Context Engine
--------------------------------------------------
DROP-IN REPLACEMENT for ContextEngine (V1).

HOW IT DIFFERS FROM V1:
  V1: Linear bullet compression. LLM compresses each exchange into
      a growing list of bullets. All bullets injected every turn.
      No awareness of which memory serves current goals.

  V2: Convergence-aware retrieval. Extracts Goals (Paths) and reusable
      Context Chunks (Modules) from each exchange. Scores every Module
      by how many active Goals it serves simultaneously. Injects ONLY
      the highest-convergence subset — the minimum context that serves
      the maximum number of active goals.

      efficiency = shared_module_hits / total_module_hits

      This directly solves Context Distraction (excess context hurting
      model quality) and reduces token spend on irrelevant memory.

ARCHITECTURE:
  - Goals (Paths)   → what the user is trying to achieve right now
  - Modules         → reusable context chunks (facts, decisions, preferences)
  - Convergence     → % of active Goals a Module serves
  - SemanticGraph   → SPO triplet store (reuses your existing infrastructure)
                      Subject=Goal, Predicate="requires", Object=Module key

INTERFACE:
  Identical to ContextEngine (V1). Same method signatures.
  AgentCore requires zero changes — just swap the instance.

PLACEMENT:
  Save as: engine/context_engine_v2.py

WIRING (in your main app factory / startup):
  from engine.context_engine_v2 import ContextEngineV2
  from memory.semantic_graph import SemanticGraph

  semantic_graph = SemanticGraph(db_path="memory_graph.db")
  ctx_v2 = ContextEngineV2(adapter=adapter, semantic_graph=semantic_graph)

SWITCHING (session-level toggle):
  # In your session route / API handler:
  session.context_mode = "v2"   # or "v1"
  agent_core.set_context_engine(session.context_mode)
"""

import re
import asyncio
import json
from typing import Optional
from llm.base_adapter import BaseLLMAdapter
from engine.context_schema import ContextItem, ContextKind, ContextPhase
from config.config import cfg
from engine.fact_guard import is_grounded_fact_record


# ── Prompts ────────────────────────────────────────────────────────────────────

GOAL_EXTRACTION_PROMPT = """\
You are a goal extraction engine for a Language OS memory system.

Analyze this exchange and identify what GOALS (objectives, tasks, intentions) 
the user is actively pursuing. A goal is something the user wants to accomplish,
learn, build, fix, or decide.

EXCHANGE:
User: {user_message}
Assistant: {assistant_response}

Extract 1-4 concise goal labels (3-6 words each).
Ignore internal execution chatter, tool schemas, and temporary plan boilerplate.
If no clear goal: output [none]

Output format — one per line, no bullets, no explanation:
goal: <label>
goal: <label>
"""

MODULE_EXTRACTION_PROMPT = """\
You are a memory extraction engine for a Language OS runtime.

Extract REUSABLE CONTEXT CHUNKS from this exchange — facts, preferences,
decisions, constraints, or established knowledge that would be useful
in future turns regardless of what topic comes next.
Ignore internal execution chatter, hidden packet names, raw tool JSON, and temporary planning boilerplate.

EXISTING MODULES (do not re-extract):
{existing_keys}

EXCHANGE:
User: {user_message}
Assistant: {assistant_response}

For each reusable chunk:
  module: <3-5 word key> | <one sentence summary of the fact/context>

VOID any module that is now outdated:
  void: <exact key of outdated module>

If nothing new: output [nothing new]
No headers. No explanation. No bullets.
"""

RECOMPRESSION_PROMPT = """\
The following context modules have been accumulated. Prune to the most 
important 15. Keep: active goals, user corrections, decisions made, 
durable preferences. Drop: one-time details, superseded facts.

MODULES:
{modules_json}

Output JSON array of kept module keys only:
["key1", "key2", ...]
No explanation.
"""

MAX_MODULES = 40          # Hard cap before pruning
TARGET_MODULES = 15       # Target after pruning
MAX_CONTEXT_CHARS = 3000  # Max chars injected into prompt
TOP_N_BY_CONVERGENCE = 12 # How many modules to inject per turn
MAX_ACTIVE_GOALS = 6
GOAL_RETIRE_AFTER_TURNS = 5
GOAL_MIN_ACTIVATION_WEIGHT = 0.2

TRIVIAL_PATTERN = re.compile(
    r"^(ok|okay|thanks|thank you|great|sure|yes|no|got it|understood|"
    r"alright|cool|nice|good|perfect|sounds good|makes sense)[\s.!?]*$",
    re.IGNORECASE,
)

PROTECTED_PATTERNS = re.compile(
    r"(user(?:'s)?\s+(?:preferred\s+)?(?:name|called|known as)|"
    r"preferred name|user (?:prefer|like|hate|use)|critical fact)",
    re.IGNORECASE,
)


class ContextEngineV2:
    """
    Convergent Graph Context Engine.
    
    Maintains a live Module Registry keyed by concept label.
    Each Module tracks which Goals it has been retrieved for.
    Context is built by ranking Modules by convergence score.
    """

    def __init__(
        self,
        adapter: BaseLLMAdapter,
        semantic_graph=None,           # Pass your SemanticGraph instance
        compression_model: Optional[str] = None,
    ):
        self.adapter           = adapter
        self.compression_model = compression_model or cfg.DEFAULT_MODEL
        self.graph             = semantic_graph  # Optional — used for cross-session persistence

        # In-memory state (rebuilt from SemanticGraph on restore if needed)
        # module_key → {content, goals: set, hit_count, created_turn}
        self._modules: dict[str, dict] = {}
        # goal_label → lifecycle metadata
        self._active_goals: dict[str, dict[str, int]] = {}
        self._turn: int = 0

    def set_adapter(self, adapter: BaseLLMAdapter):
        """Hot-swap adapter — same interface as V1."""
        self.adapter = adapter

    @property
    def model(self) -> str:
        return self.compression_model

    # ── Public Interface (identical to V1) ────────────────────────────────────

    async def compress_exchange(
        self,
        user_message:       str,
        assistant_response: str,
        current_context:    str,
        is_first_exchange:  bool = False,
        model:              str = None,
        grounding_text:     str = "",
    ) -> tuple[str, list[dict], list[dict]]:
        """
        Process a new exchange through the convergence engine.
        
        Returns:
          (updated_context_string, list_of_keyed_facts, list_of_voids)
          
        The context_string returned is a JSON snapshot of the module registry
        so it can be persisted to SessionManager.compressed_context exactly
        like V1's bullet string — no session schema changes needed.
        """
        self._turn += 1

        # Never skip first exchange
        is_trivial = (
            bool(TRIVIAL_PATTERN.match(user_message.strip()))
            and len(assistant_response.strip()) < 40
        )
        if is_trivial and not is_first_exchange:
            return self._serialize_context(), [], []

        # Restore state from persisted context if this is a resumed session
        if current_context and not self._modules:
            self._deserialize_context(current_context)

        use_model = model or self.compression_model

        # Run goal and module extraction concurrently
        goals, modules, voids = await self._extract_all(
            user_message, assistant_response, use_model
        )

        self._touch_goals(goals)
        self._retire_inactive_goals()

        # Register/update modules
        keyed_facts = []
        for m in modules:
            key     = m["key"]
            content = m["content"]
            candidate = {
                "key":       key,
                "fact":      content,
                "subject":   goals[0] if goals else "General",
                "predicate": "requires",
                "object":    key,
            }
            grounded, _ = is_grounded_fact_record(
                candidate,
                user_message=user_message,
                grounding_text=grounding_text,
            )
            if not grounded:
                continue
            protected = bool(PROTECTED_PATTERNS.search(f"{key} {content}"))
            if key not in self._modules:
                self._modules[key] = {
                    "content":      content,
                    "goals":        set(),
                    "hit_count":    0,
                    "created_turn": self._turn,
                    "protected":    protected,
                }
            elif protected:
                self._modules[key]["protected"] = True
            # Associate this module with all current active goals
            self._modules[key]["goals"].update(goals)
            self._modules[key]["last_seen_turn"] = self._turn
            self._modules[key]["hit_count"] += 1

            keyed_facts.append(candidate)

            # Persist to SemanticGraph for cross-session memory if available
            if self.graph and goals:
                try:
                    for goal in goals:
                        asyncio.create_task(
                            self.graph.add_triplet(goal, "requires", key)
                        )
                except Exception:
                    pass

        # Apply voids
        void_records = self._apply_voids(voids)

        # Handle first exchange forced storage
        if is_first_exchange and not keyed_facts:
            key = "Session Start"
            self._modules[key] = {
                "content":      f"First message: {user_message.strip()[:120]}",
                "goals":        set(goals) if goals else {"session"},
                "hit_count":    1,
                "created_turn": self._turn,
                "last_seen_turn": self._turn,
                "protected":    False,
            }
            keyed_facts.append({
                "key":       key,
                "fact":      self._modules[key]["content"],
                "subject":   "Session",
                "predicate": "Start",
                "object":    "First message",
            })

        # Prune if registry is too large
        if len(self._modules) > MAX_MODULES:
            await self._prune_modules(use_model)

        serialized = self._serialize_context()
        return serialized, keyed_facts, void_records

    def build_context_block(self, context: str) -> str:
        """
        Build the context block injected into the system prompt.
        
        Ranks modules by convergence score — modules serving the most
        active goals appear first. Caps total injected chars to prevent
        context explosion.
        """
        # Restore from serialized context if in-memory state is empty
        if context and not self._modules:
            self._deserialize_context(context)

        if not self._modules:
            return ""

        ranked = self._rank_by_convergence()

        lines = []
        total_chars = 0
        goal_summary = ", ".join(self._ordered_active_goal_labels()[:5]) or "none detected"

        lines.append(f"Active Goals: {goal_summary}")

        for key, module, score in ranked[:TOP_N_BY_CONVERGENCE]:
            content = module["content"]
            if len(content) > 200:
                content = content[:180] + "..."

            if score > 0:
                label = f"[{score:.0%} convergence] {key}"
            else:
                label = key

            line = f"- {label}: {content}"
            if total_chars + len(line) > MAX_CONTEXT_CHARS:
                omitted = len(ranked) - len(lines) + 1
                if omitted > 0:
                    lines.append(f"[...{omitted} lower-priority modules omitted]")
                break

            lines.append(line)
            total_chars += len(line)

        if not lines:
            return ""

        return (
            "--- Convergent Memory (V2) ---\n"
            + "\n".join(lines)
            + "\n--- End Convergent Memory ---"
        )

    def build_context_items(self, context: str) -> list[ContextItem]:
        block = self.build_context_block(context)
        if not block:
            return []
        return [
            ContextItem(
                item_id="context_engine_v2_memory",
                kind=ContextKind.MEMORY,
                title="",
                content=block,
                source="context_engine_v2",
                priority=55,
                max_chars=1800,
                phase_visibility=frozenset({
                    ContextPhase.PLANNING,
                    ContextPhase.ACTING,
                    ContextPhase.RESPONSE,
                    ContextPhase.VERIFICATION,
                }),
                trace_id="ctx:v2:memory",
                provenance={"engine": "v2"},
                formatted=True,
            )
        ]

    # ── Convergence Scoring ────────────────────────────────────────────────────

    def _convergence_score(self, module_key: str) -> float:
        """
        Score = weighted shared active-goal utility / total active-goal utility.
        Newer goals retain more activation weight than older goals so stale
        context can remain recoverable without dominating the prompt.
        """
        if not self._active_goals or module_key not in self._modules:
            return 0.0
        module_goals = self._modules[module_key]["goals"]
        total_weight = 0.0
        shared_weight = 0.0
        for goal, meta in self._active_goals.items():
            weight = self._goal_activation_weight(meta)
            total_weight += weight
            if goal in module_goals:
                shared_weight += weight
        if total_weight <= 0:
            return 0.0
        return shared_weight / total_weight

    def _goal_activation_weight(self, meta: dict[str, int]) -> float:
        last_seen_turn = int(meta.get("last_seen_turn", meta.get("first_seen_turn", self._turn)))
        age = max(0, self._turn - last_seen_turn)
        return max(GOAL_MIN_ACTIVATION_WEIGHT, 1.0 - (0.2 * age))

    def _ordered_active_goal_labels(self) -> list[str]:
        ranked = sorted(
            self._active_goals.items(),
            key=lambda item: (
                self._goal_activation_weight(item[1]),
                int(item[1].get("last_seen_turn", 0)),
                int(item[1].get("first_seen_turn", 0)),
            ),
            reverse=True,
        )
        return [goal for goal, _meta in ranked]

    def _rank_by_convergence(self) -> list[tuple[str, dict, float]]:
        """Return modules sorted by active-goal relevance, then durability/recency."""
        scored = []
        for key, mod in self._modules.items():
            score = self._convergence_score(key)
            shared_weight = sum(
                self._goal_activation_weight(self._active_goals[goal])
                for goal in mod.get("goals", set())
                if goal in self._active_goals
            )
            scored.append((key, mod, score, shared_weight))
        scored.sort(
            key=lambda x: (
                x[2],
                x[3],
                1 if x[1].get("protected") else 0,
                x[1]["hit_count"],
                x[1].get("last_seen_turn", x[1].get("created_turn", 0)),
            ),
            reverse=True,
        )
        return [(key, mod, score) for key, mod, score, _shared_weight in scored]

    def _touch_goals(self, goals: list[str]) -> None:
        for goal in goals:
            if goal not in self._active_goals:
                self._active_goals[goal] = {
                    "first_seen_turn": self._turn,
                    "last_seen_turn": self._turn,
                }
            else:
                self._active_goals[goal]["last_seen_turn"] = self._turn

    def _retire_inactive_goals(self) -> None:
        survivors: dict[str, dict[str, int]] = {}
        for goal, meta in self._active_goals.items():
            last_seen_turn = int(meta.get("last_seen_turn", meta.get("first_seen_turn", 0)))
            if self._turn - last_seen_turn <= GOAL_RETIRE_AFTER_TURNS:
                survivors[goal] = meta
        if len(survivors) > MAX_ACTIVE_GOALS:
            ranked = sorted(
                survivors.items(),
                key=lambda item: (
                    int(item[1].get("last_seen_turn", 0)),
                    int(item[1].get("first_seen_turn", 0)),
                ),
                reverse=True,
            )
            survivors = dict(ranked[:MAX_ACTIVE_GOALS])
        self._active_goals = survivors

    # ── Extraction ─────────────────────────────────────────────────────────────

    async def _extract_all(
        self,
        user_message: str,
        assistant_response: str,
        model: str,
    ) -> tuple[list[str], list[dict], list[str]]:
        """Run goal + module extraction concurrently."""
        from llm.adapter_factory import create_adapter, strip_provider_prefix

        current_adapter = create_adapter(provider=model) if ":" in model else self.adapter
        clean_model = strip_provider_prefix(model)

        # Truncate long responses before sending to extractor
        assistant_short = assistant_response
        if len(assistant_short) > 1200:
            assistant_short = assistant_short[:800] + "\n[...truncated...]\n" + assistant_short[-400:]

        existing_keys = ", ".join(list(self._modules.keys())[:30]) or "(none)"

        goal_prompt = GOAL_EXTRACTION_PROMPT.format(
            user_message=user_message.strip(),
            assistant_response=assistant_short,
        )
        module_prompt = MODULE_EXTRACTION_PROMPT.format(
            existing_keys=existing_keys,
            user_message=user_message.strip(),
            assistant_response=assistant_short,
        )

        async def _call(prompt):
            try:
                return await current_adapter.complete(
                    model=clean_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=300,
                )
            except Exception as e:
                print(f"[ContextEngineV2] Extraction call failed: {e}")
                return ""

        goal_raw, module_raw = await asyncio.gather(
            _call(goal_prompt),
            _call(module_prompt),
        )

        goals   = self._parse_goals(goal_raw)
        modules, voids = self._parse_modules(module_raw)
        return goals, modules, voids

    def _parse_goals(self, raw: str) -> list[str]:
        goals = []
        if not raw or "[none]" in raw.lower():
            return goals
        for line in raw.strip().split("\n"):
            line = line.strip()
            if line.lower().startswith("goal:"):
                label = line[5:].strip()
                if label and len(label) > 2:
                    goals.append(label[:60])
        return goals[:4]

    def _parse_modules(self, raw: str) -> tuple[list[dict], list[str]]:
        modules = []
        voids   = []
        if not raw or "[nothing new]" in raw.lower():
            return modules, voids
        for line in raw.strip().split("\n"):
            line = line.strip()
            if line.lower().startswith("module:"):
                parts = line[7:].split("|", 1)
                if len(parts) == 2:
                    key     = parts[0].strip()[:80]
                    content = parts[1].strip()[:300]
                    if key and content:
                        modules.append({"key": key, "content": content})
            elif line.lower().startswith("void:"):
                key = line[5:].strip()[:80]
                if key:
                    voids.append(key)
        return modules[:8], voids

    def _apply_voids(self, voids: list[str]) -> list[dict]:
        void_records: list[dict] = []
        for key in voids:
            if key in self._modules:
                if self._modules[key].get("protected"):
                    continue
                del self._modules[key]
                void_records.append({"subject": key, "predicate": "requires"})
        return void_records

    # ── Pruning ────────────────────────────────────────────────────────────────

    async def _prune_modules(self, model: str):
        """Ask the LLM to select which modules to keep. Falls back to recency."""
        from llm.adapter_factory import create_adapter, strip_provider_prefix
        current_adapter = create_adapter(provider=model) if ":" in model else self.adapter
        clean_model = strip_provider_prefix(model)

        modules_summary = [
            {"key": k, "content": v["content"][:100], "hits": v["hit_count"]}
            for k, v in self._modules.items()
        ]

        prompt = RECOMPRESSION_PROMPT.format(modules_json=json.dumps(modules_summary, indent=2))

        try:
            result = await current_adapter.complete(
                model=clean_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            raw = result.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            keep_keys = json.loads(raw)
            if isinstance(keep_keys, list):
                keep_set = set(keep_keys)
                self._modules = {k: v for k, v in self._modules.items() if k in keep_set}
                print(f"[ContextEngineV2] Pruned to {len(self._modules)} modules.")
                return
        except Exception as e:
            print(f"[ContextEngineV2] Prune LLM call failed: {e} — falling back to recency.")

        # Fallback: keep highest hit_count modules
        protected = {k: v for k, v in self._modules.items() if v.get("protected")}
        prunable = {k: v for k, v in self._modules.items() if not v.get("protected")}
        sorted_prunable = sorted(
            prunable.items(),
            key=lambda x: (self._convergence_score(x[0]), x[1]["hit_count"]),
            reverse=True
        )
        keep_prunable = dict(sorted_prunable[:max(0, TARGET_MODULES - len(protected))])
        self._modules = {**protected, **keep_prunable}

    # ── Serialization ──────────────────────────────────────────────────────────

    def _serialize_context(self) -> str:
        """
        Serialize engine state to a JSON string compatible with
        SessionManager.compressed_context (which stores a plain string).
        """
        exportable = {}
        for k, v in self._modules.items():
            exportable[k] = {
                "content":      v["content"],
                "goals":        list(v["goals"]),
                "hit_count":    v["hit_count"],
                "created_turn": v.get("created_turn", 0),
                "last_seen_turn": v.get("last_seen_turn", v.get("created_turn", 0)),
                "protected":    bool(v.get("protected", False)),
            }
        payload = {
            "__v2__":        True,
            "turn":          self._turn,
            "active_goals":  self._active_goals,
            "modules":       exportable,
        }
        return json.dumps(payload)

    def _deserialize_context(self, context: str):
        """Restore engine state from a serialized string."""
        if not context:
            return
        try:
            payload = json.loads(context)
            if not payload.get("__v2__"):
                # This is V1 bullet text — bootstrap V2 from it
                self._bootstrap_from_v1(context)
                return
            self._turn         = payload.get("turn", 0)
            raw_goals = payload.get("active_goals", {})
            raw_modules        = payload.get("modules", {})
            normalized_goals: dict[str, dict[str, int]] = {}
            if isinstance(raw_goals, dict):
                for key, value in raw_goals.items():
                    if isinstance(value, dict):
                        normalized_goals[str(key)] = {
                            "first_seen_turn": int(value.get("first_seen_turn", 0) or 0),
                            "last_seen_turn": int(value.get("last_seen_turn", value.get("first_seen_turn", 0)) or 0),
                        }
                    else:
                        turn = int(value or 0)
                        normalized_goals[str(key)] = {
                            "first_seen_turn": turn,
                            "last_seen_turn": turn,
                        }
            self._active_goals = normalized_goals
            self._modules = {
                k: {
                    "content":      v["content"],
                    "goals":        set(v.get("goals", [])),
                    "hit_count":    v.get("hit_count", 1),
                    "created_turn": v.get("created_turn", 0),
                    "last_seen_turn": v.get("last_seen_turn", v.get("created_turn", 0)),
                    "protected":    bool(v.get("protected", False)),
                }
                for k, v in raw_modules.items()
            }
        except Exception as e:
            print(f"[ContextEngineV2] Deserialize failed: {e} — starting fresh.")

    def _bootstrap_from_v1(self, v1_bullets: str):
        """
        Convert V1 bullet-list context into V2 modules on first switch.
        Each bullet becomes a module with no goal associations (score=0).
        They will gain associations as new exchanges are processed.
        """
        lines = [l.strip() for l in v1_bullets.split("\n") if l.strip().startswith("- ")]
        for i, line in enumerate(lines[:MAX_MODULES]):
            text = line[2:].strip()
            words = text.split()[:4]
            key = " ".join(w.capitalize() for w in words) or f"Legacy Context {i}"
            self._modules[key] = {
                "content":      text[:300],
                "goals":        {"legacy"},
                "hit_count":    1,
                "created_turn": 0,
                "last_seen_turn": 0,
                "protected":    False,
            }
        print(f"[ContextEngineV2] Bootstrapped {len(self._modules)} modules from V1 context.")
