from __future__ import annotations

from typing import Any, Optional

from llm.base_adapter import BaseLLMAdapter
from engine.context_schema import ContextItem, ContextKind, ContextPhase


_MEMORY_PHASE_VISIBILITY = frozenset({
    ContextPhase.PLANNING,
    ContextPhase.ACTING,
    ContextPhase.RESPONSE,
    ContextPhase.VERIFICATION,
})


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
