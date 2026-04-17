from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from llm.base_adapter import BaseLLMAdapter
from engine.candidate_signals import parse_candidate_context, render_candidate_signals
from engine.context_schema import ContextItem, ContextKind, ContextPhase
from run_engine.memory_pipeline import (
    MemoryCommitOutcome,
    MemoryCommitPlan,
    apply_memory_commit as apply_governed_memory_commit,
    build_deterministic_memory_commit as build_governed_deterministic_commit,
    plan_memory_commit as plan_governed_memory_commit,
)


_MEMORY_PHASE_VISIBILITY = frozenset({
    ContextPhase.PLANNING,
    ContextPhase.ACTING,
    ContextPhase.RESPONSE,
    ContextPhase.VERIFICATION,
})


@dataclass(frozen=True)
class GovernedMemorySurface:
    candidate_context: str
    historical_context: str
    memory_items: list[ContextItem]
    provenance: dict[str, Any]


class ContextGovernor:
    """
    Shared context-engine resolver for managed and compatibility runtimes.

    This keeps V1/V2/V3 as policy modes behind one authority surface instead
    of letting each runtime own a separate engine-selection implementation.
    """

    def __init__(
        self,
        *,
        adapter: BaseLLMAdapter,
        v1_engine,
        semantic_graph=None,
    ):
        self.adapter = adapter
        self.graph = semantic_graph
        self._v1_engine = v1_engine
        self._v2_engine = None
        self._v3_engine = None

    def _ensure_v1_engine(self, compression_model: Optional[str] = None):
        if self._v1_engine is None:
            from engine.context_engine import ContextEngine

            self._v1_engine = ContextEngine(
                adapter=self.adapter,
                compression_model=compression_model,
            )
        return self._v1_engine

    def set_adapter(self, adapter: BaseLLMAdapter) -> None:
        self.adapter = adapter
        for engine in (self._v1_engine, self._v2_engine, self._v3_engine):
            if engine is not None and hasattr(engine, "set_adapter"):
                engine.set_adapter(adapter)

    def resolve(
        self,
        mode: Optional[str],
        *,
        compression_model: Optional[str] = None,
        ):
        normalized = str(mode or "v1").strip().lower()
        if normalized not in {"v1", "v2", "v3"}:
            normalized = "v1"

        self._ensure_v1_engine(compression_model)

        if normalized == "v2":
            if self._v2_engine is None:
                from engine.context_engine_v2 import ContextEngineV2

                self._v2_engine = ContextEngineV2(
                    adapter=self.adapter,
                    semantic_graph=self.graph,
                    compression_model=compression_model or "llama3.2",
                )
            engine = self._v2_engine
        elif normalized == "v3":
            if self._v3_engine is None:
                from engine.context_engine_v3 import ContextEngineV3

                self._v3_engine = ContextEngineV3(
                    adapter=self.adapter,
                    semantic_graph=self.graph,
                    compression_model=compression_model or "llama3.2",
                )
            engine = self._v3_engine
        else:
            engine = self._v1_engine

        if engine is not None:
            if hasattr(engine, "set_adapter"):
                engine.set_adapter(self.adapter)
            if compression_model and hasattr(engine, "compression_model"):
                engine.compression_model = compression_model
        return engine

    def mode_for_engine(self, engine: Optional[object]) -> str:
        if engine is None:
            return "v1"
        name = engine.__class__.__name__.lower()
        if "v3" in name:
            return "v3"
        if "v2" in name:
            return "v2"
        return "v1"

    def _build_standard_memory_item(
        self,
        *,
        item_id: str,
        title: str,
        content: str,
        trace_id: str,
        provenance: Optional[dict[str, Any]] = None,
        priority: int = 58,
        max_chars: int = 1800,
    ) -> ContextItem:
        return ContextItem(
            item_id=item_id,
            kind=ContextKind.MEMORY,
            title=title,
            content=content,
            source="context_governor",
            priority=priority,
            max_chars=max_chars,
            phase_visibility=_MEMORY_PHASE_VISIBILITY,
            trace_id=trace_id,
            provenance=provenance or {},
            formatted=True,
        )

    def _pattern_cues(
        self,
        *,
        engine: Optional[object],
        context: str,
        current_facts: Optional[list[tuple[str, str, str]]],
    ) -> str:
        cues: list[str] = []
        fact_predicates = [str(predicate or "").strip() for (_, predicate, _) in current_facts or [] if str(predicate or "").strip()]
        identity_predicates = [p for p in fact_predicates if p in {
            "preferred_name",
            "location",
            "timezone",
            "preferred_editor",
            "package_manager",
            "primary_language",
            "response_verbosity",
            "operating_system",
            "pronouns",
        }]
        task_predicates = [p for p in fact_predicates if p in {
            "environment_mode",
            "scope_boundary",
            "budget_limit",
            "task_constraint",
            "followup_directive",
        }]
        if identity_predicates:
            ordered = ", ".join(dict.fromkeys(identity_predicates))
            cues.append(f"- Identity anchors: {ordered}")
        if task_predicates:
            ordered = ", ".join(dict.fromkeys(task_predicates))
            cues.append(f"- Task frame: {ordered}")

        normalized_context = str(context or "").lower()
        if any(token in normalized_context for token in ("actually", "updated", "changed", "instead", "moved")):
            cues.append("- Correction lineage: prefer recent corrections over older paraphrases.")

        mode = self.mode_for_engine(engine)
        if mode in {"v2", "v3"} and engine is not None:
            goal_labels = []
            ordered_goals = getattr(engine, "_ordered_active_goal_labels", None)
            if callable(ordered_goals):
                try:
                    goal_labels = list(ordered_goals()[:4])
                except Exception:
                    goal_labels = []
            elif mode == "v3":
                inner = getattr(engine, "_v2", None)
                if inner is not None and callable(getattr(inner, "_ordered_active_goal_labels", None)):
                    try:
                        goal_labels = list(inner._ordered_active_goal_labels()[:4])
                    except Exception:
                        goal_labels = []
            if goal_labels:
                cues.append(f"- Goal convergence: {', '.join(goal_labels)}")

        if mode == "v1":
            cues.append("- Memory profile: continuity-biased recall with broad durable carryover.")
        elif mode == "v2":
            cues.append("- Memory profile: activation-biased recall that favors currently active goals.")
        else:
            cues.append("- Memory profile: hybrid recall balancing durable anchors with active-goal relevance.")

        return "\n".join(cues)

    def build_memory_items(
        self,
        *,
        engine: Optional[object],
        context: str,
        current_facts: Optional[list[tuple[str, str, str]]] = None,
        trace_prefix: str = "ctx",
    ) -> list[ContextItem]:
        if not str(context or "").strip() or engine is None:
            return []

        mode = self.mode_for_engine(engine)
        items: list[ContextItem] = []

        if mode == "v3" and hasattr(engine, "_split_context"):
            durable_context, convergent_context = engine._split_context(context)
            inner_v2 = getattr(engine, "_v2", None)
            convergent_block = ""
            if inner_v2 is not None and hasattr(inner_v2, "build_context_block"):
                try:
                    convergent_block = str(inner_v2.build_context_block(convergent_context) or "").strip()
                except Exception:
                    convergent_block = ""
            if convergent_block:
                items.append(
                    self._build_standard_memory_item(
                        item_id="context_governor_active_memory",
                        title="Active Memory",
                        content=convergent_block,
                        trace_id=f"{trace_prefix}:governor:active",
                        provenance={"engine": "v3", "profile": "active"},
                        priority=55,
                        max_chars=1500,
                    )
                )
            durable_lines = [line for line in str(durable_context or "").splitlines() if line.strip()]
            if durable_lines:
                chosen = engine._select_durable_lines(durable_lines, max_items=8) if hasattr(engine, "_select_durable_lines") else durable_lines[:8]
                durable_block = (
                    "--- Durable Anchors ---\n"
                    + "\n".join(chosen)
                    + ("\n[...additional durable memory omitted]" if len(durable_lines) > len(chosen) else "")
                    + "\n--- End Durable Anchors ---"
                )
                items.append(
                    self._build_standard_memory_item(
                        item_id="context_governor_durable_memory",
                        title="Durable Anchors",
                        content=durable_block,
                        trace_id=f"{trace_prefix}:governor:durable",
                        provenance={"engine": "v3", "profile": "durable"},
                        priority=57,
                        max_chars=1200,
                    )
                )
        else:
            build_block = getattr(engine, "build_context_block", None)
            if callable(build_block):
                try:
                    block = str(build_block(context) or "").strip()
                except Exception:
                    block = ""
                if block:
                    title = "Durable Continuity" if mode == "v1" else "Active Memory"
                    items.append(
                        self._build_standard_memory_item(
                            item_id=f"context_governor_{mode}_memory",
                            title=title,
                            content=block,
                            trace_id=f"{trace_prefix}:governor:{mode}",
                            provenance={"engine": mode},
                            priority=56 if mode == "v2" else 60,
                            max_chars=1800,
                        )
                    )

        pattern_content = self._pattern_cues(engine=engine, context=context, current_facts=current_facts)
        if pattern_content:
            items.append(
                ContextItem(
                    item_id="context_governor_pattern_cues",
                    kind=ContextKind.META,
                    title="Pattern Cues",
                    content=pattern_content,
                    source="context_governor",
                    priority=59,
                    max_chars=700,
                    phase_visibility=_MEMORY_PHASE_VISIBILITY,
                    trace_id=f"{trace_prefix}:governor:patterns",
                    provenance={"engine": mode},
                )
        )
        return items

    def build_memory_surface(
        self,
        *,
        engine: Optional[object],
        session: Optional[object],
        context: str,
        current_facts: Optional[list[tuple[str, str, str]]] = None,
        trace_prefix: str = "ctx",
        correction_turn: bool = False,
        direct_fact_memory_only: bool = False,
        # When True, suppress _build_historical_context so the v3 governor
        # items (active memory + durable anchors) are the sole older-history
        # lane and there is no duplication.
        suppress_historical_for_managed_engine: bool = False,
    ) -> GovernedMemorySurface:
        candidate_signals = list(getattr(session, "candidate_signals", []) or []) if session is not None else []
        legacy_candidate_context = str(getattr(session, "candidate_context", "") or "").strip() if session is not None else ""
        if candidate_signals:
            candidate_context = render_candidate_signals(candidate_signals)
            candidate_source = "structured_candidate_signals"
        else:
            candidate_context = legacy_candidate_context
            candidate_source = "legacy_candidate_context" if candidate_context else "none"
            if not candidate_context and legacy_candidate_context:
                candidate_signals = parse_candidate_context(legacy_candidate_context)

        # v3 governor emits its own active-memory + durable-anchor items which
        # fully cover older-session context.  Suppress the raw historical lane
        # to avoid injecting the same turns twice at different priorities.
        mode = self.mode_for_engine(engine)
        skip_historical = suppress_historical_for_managed_engine or mode == "v3"
        historical_context = (
            ""
            if skip_historical
            else self._build_historical_context(
                session=session,
                correction_turn=correction_turn,
                direct_fact_memory_only=direct_fact_memory_only,
            )
        )
        memory_items = self.build_memory_items(
            engine=engine,
            context=context,
            current_facts=current_facts,
            trace_prefix=trace_prefix,
        )
        return GovernedMemorySurface(
            candidate_context=candidate_context,
            historical_context=historical_context,
            memory_items=memory_items,
            provenance={
                "mode": mode,
                "candidate_source": candidate_source,
                "candidate_count": len(candidate_signals),
                "historical_segments": len([seg for seg in historical_context.split("\n\n---\n") if seg.strip()]),
                "memory_item_count": len(memory_items),
                "direct_fact_memory_only": bool(direct_fact_memory_only),
                "correction_turn": bool(correction_turn),
                "historical_suppressed": bool(skip_historical),
            },
        )

    def get_current_facts(
        self,
        session_id: str,
        *,
        owner_id: Optional[str] = None,
    ) -> list[tuple[str, str, str]]:
        if self.graph is None:
            return []
        try:
            return list(self.graph.get_current_facts(session_id, owner_id=owner_id) or [])
        except Exception:
            return []

    async def search_memory(
        self,
        query: str,
        *,
        owner_id: Optional[str],
        session_id: Optional[str] = None,
        locus_id: Optional[str] = None,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        from memory.retrieval import unified_memory_search

        return await unified_memory_search(
            query,
            owner_id=owner_id,
            session_id=session_id,
            locus_id=locus_id,
            top_k=top_k,
            threshold=threshold,
            graph=self.graph,
        )

    def build_memory_commit_plan(
        self,
        *,
        context_result: Optional[tuple[Any, ...]] = None,
        user_message: str,
        tool_results: list[dict[str, Any]],
        deterministic_keyed_facts: list[dict[str, Any]],
        deterministic_voids: list[dict[str, Any]],
        current_facts: Optional[list[tuple[str, str, str]]],
        existing_candidate_signals: Optional[list[dict[str, str]]],
        existing_candidate_context: str,
        new_candidate_signals: Optional[list[dict[str, Any]]] = None,
        current_turn: Optional[int] = None,
    ) -> MemoryCommitPlan:
        if context_result is None:
            return build_governed_deterministic_commit(
                deterministic_keyed_facts=deterministic_keyed_facts,
                deterministic_voids=deterministic_voids,
                existing_candidate_signals=existing_candidate_signals,
                existing_candidate_context=existing_candidate_context,
                user_message=user_message,
                new_candidate_signals=new_candidate_signals,
                current_turn=current_turn,
            )
        return plan_governed_memory_commit(
            context_result=context_result,
            user_message=user_message,
            tool_results=tool_results,
            deterministic_keyed_facts=deterministic_keyed_facts,
            deterministic_voids=deterministic_voids,
            current_facts=current_facts,
            existing_candidate_signals=existing_candidate_signals,
            existing_candidate_context=existing_candidate_context,
            new_candidate_signals=new_candidate_signals,
            current_turn=current_turn,
        )

    async def apply_memory_commit(
        self,
        *,
        sessions,
        session_id: str,
        owner_id: Optional[str],
        agent_id: str,
        turn: int,
        run_id: str,
        user_message: str,
        assistant_response: str,
        plan: MemoryCommitPlan,
        current_context: str,
    ) -> MemoryCommitOutcome:
        return await apply_governed_memory_commit(
            sessions=sessions,
            session_id=session_id,
            owner_id=owner_id,
            agent_id=agent_id,
            turn=turn,
            run_id=run_id,
            user_message=user_message,
            assistant_response=assistant_response,
            graph=self.graph,
            plan=plan,
            current_context=current_context,
        )

    def _build_historical_context(
        self,
        *,
        session: Optional[object],
        correction_turn: bool,
        direct_fact_memory_only: bool,
    ) -> str:
        if direct_fact_memory_only or session is None:
            return ""

        history = list(getattr(session, "full_history", []) or [])
        recent = list(getattr(session, "sliding_window", []) or [])
        if len(history) <= len(recent):
            return ""

        older_history = history[:-len(recent)] if recent else history[:-4]
        max_segments = 2 if correction_turn else 4
        segments: list[str] = []
        for entry in older_history[-max_segments:]:
            role = str(entry.get("role") or "")
            content = str(entry.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                clipped = content if len(content) <= 260 else content[:257].rstrip() + "..."
                segments.append(f"{role.upper()}: {clipped}")
        return "\n\n---\n".join(segments)
