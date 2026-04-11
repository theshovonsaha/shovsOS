"""
ContextEngineV3 — Hybrid Durable + Convergent Context
-----------------------------------------------------
Experimental context engine that combines:
- V1 durable linear memory for stable facts/corrections
- V2 convergent module ranking for active-goal relevance

Storage format remains a single string in SessionManager.compressed_context.
Existing V1/V2 sessions are bootstrapped on first use.
"""

from __future__ import annotations

import json
from typing import Optional

from llm.base_adapter import BaseLLMAdapter
from engine.context_engine import ContextEngine
from engine.context_engine_v2 import ContextEngineV2
from engine.context_schema import ContextItem, ContextKind, ContextPhase
from config.config import cfg
from engine.fact_guard import is_grounded_fact_record


class ContextEngineV3:
    def __init__(
        self,
        adapter: BaseLLMAdapter,
        semantic_graph=None,
        compression_model: Optional[str] = None,
    ):
        self.adapter = adapter
        self.compression_model = compression_model or cfg.DEFAULT_MODEL
        self._v1 = ContextEngine(adapter=adapter, compression_model=compression_model)
        self._v2 = ContextEngineV2(
            adapter=adapter,
            semantic_graph=semantic_graph,
            compression_model=compression_model,
        )

    def set_adapter(self, adapter: BaseLLMAdapter):
        self.adapter = adapter
        self._v1.set_adapter(adapter)
        self._v2.set_adapter(adapter)

    @property
    def model(self) -> str:
        return self.compression_model

    def _split_context(self, current_context: str) -> tuple[str, str]:
        if not current_context:
            return "", ""
        try:
            payload = json.loads(current_context)
            if payload.get("__v3__"):
                return payload.get("durable_context", ""), payload.get("convergent_context", "")
            if payload.get("__v2__"):
                return "", current_context
        except Exception:
            pass
        return current_context, ""

    def _serialize_context(self, durable_context: str, convergent_context: str) -> str:
        payload = {
            "__v3__": True,
            "durable_context": durable_context or "",
            "convergent_context": convergent_context or "",
        }
        return json.dumps(payload)

    def _dedupe_fact_records(self, records: list[dict]) -> list[dict]:
        seen: set[tuple[str, str, str]] = set()
        deduped: list[dict] = []
        for record in records:
            key = (
                str(record.get("subject", "")),
                str(record.get("predicate", "")),
                str(record.get("object", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped

    async def compress_exchange(
        self,
        user_message: str,
        assistant_response: str,
        current_context: str,
        is_first_exchange: bool = False,
        model: str = None,
        grounding_text: str = "",
    ) -> tuple[str, list[dict], list[dict]]:
        durable_context, convergent_context = self._split_context(current_context)
        use_model = model or self.compression_model

        durable_updated, durable_facts, durable_voids = await self._v1.compress_exchange(
            user_message=user_message,
            assistant_response=assistant_response,
            current_context=durable_context,
            is_first_exchange=is_first_exchange,
            model=use_model,
            grounding_text=grounding_text,
        )
        convergent_updated, convergent_facts, convergent_voids = await self._v2.compress_exchange(
            user_message=user_message,
            assistant_response=assistant_response,
            current_context=convergent_context,
            is_first_exchange=is_first_exchange,
            model=use_model,
            grounding_text=grounding_text,
        )

        serialized = self._serialize_context(durable_updated, convergent_updated)
        facts = self._dedupe_fact_records(durable_facts + convergent_facts)
        voids = self._dedupe_fact_records(durable_voids + convergent_voids)
        return serialized, facts, voids

    def build_context_block(self, context: str) -> str:
        durable_context, convergent_context = self._split_context(context)

        parts: list[str] = []

        v2_block = self._v2.build_context_block(convergent_context)
        if v2_block:
            parts.append(v2_block)

        if durable_context:
            durable_lines = [line for line in durable_context.split("\n") if line.strip()]
            if durable_lines:
                # Keep durable memory concise: preserve only the strongest anchors.
                chosen = durable_lines[:8]
                parts.append(
                    "--- Durable Memory (V3) ---\n"
                    + "\n".join(chosen)
                    + ("\n[...additional durable memory omitted]" if len(durable_lines) > len(chosen) else "")
                    + "\n--- End Durable Memory ---"
                )

        return "\n\n".join(parts)

    def build_context_items(self, context: str) -> list[ContextItem]:
        durable_context, convergent_context = self._split_context(context)
        items: list[ContextItem] = []

        convergent_block = self._v2.build_context_block(convergent_context)
        if convergent_block:
            items.append(
                ContextItem(
                    item_id="context_engine_v3_convergent",
                    kind=ContextKind.MEMORY,
                    title="",
                    content=convergent_block,
                    source="context_engine_v3",
                    priority=55,
                    max_chars=1500,
                    phase_visibility=frozenset({
                        ContextPhase.PLANNING,
                        ContextPhase.ACTING,
                        ContextPhase.RESPONSE,
                        ContextPhase.VERIFICATION,
                    }),
                    trace_id="ctx:v3:convergent",
                    provenance={"engine": "v3", "memory_type": "convergent"},
                    formatted=True,
                )
            )

        if durable_context:
            durable_lines = [line for line in durable_context.split("\n") if line.strip()]
            if durable_lines:
                chosen = durable_lines[:8]
                durable_block = (
                    "--- Durable Memory (V3) ---\n"
                    + "\n".join(chosen)
                    + ("\n[...additional durable memory omitted]" if len(durable_lines) > len(chosen) else "")
                    + "\n--- End Durable Memory ---"
                )
                items.append(
                    ContextItem(
                        item_id="context_engine_v3_durable",
                        kind=ContextKind.MEMORY,
                        title="",
                        content=durable_block,
                        source="context_engine_v3",
                        priority=57,
                        max_chars=1200,
                        phase_visibility=frozenset({
                            ContextPhase.PLANNING,
                            ContextPhase.ACTING,
                            ContextPhase.RESPONSE,
                            ContextPhase.VERIFICATION,
                            ContextPhase.MEMORY_COMMIT,
                        }),
                        trace_id="ctx:v3:durable",
                        provenance={"engine": "v3", "memory_type": "durable"},
                        formatted=True,
                    )
                )

        return items
