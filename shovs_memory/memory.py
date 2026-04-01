from __future__ import annotations

from typing import Optional

from api.memory_inspector import build_session_memory_payload
from engine.deterministic_facts import extract_user_stated_fact_updates
from memory.semantic_graph import SemanticGraph
from orchestration.session_manager import SessionManager


def _simple_context_preview(raw_context: str) -> list[str]:
    return [line.strip() for line in (raw_context or "").splitlines() if line.strip()]


class ShovsMemory:
    """
    Minimal public facade over the existing Shovs memory stack.

    This intentionally wraps the current runtime primitives instead of inventing
    a parallel memory implementation.
    """

    def __init__(
        self,
        *,
        session_id: str,
        owner_id: Optional[str] = None,
        graph: Optional[SemanticGraph] = None,
        session_manager: Optional[SessionManager] = None,
        db_path: str = "memory_graph.db",
        embedding_model: str = "nomic-embed-text",
    ):
        self.session_id = session_id
        self.owner_id = owner_id
        self.graph = graph or SemanticGraph(db_path=db_path, embedding_model=embedding_model)
        self.session_manager = session_manager

    async def retrieve(self, query: str, *, top_k: int = 5, threshold: float = 0.5) -> list[dict]:
        return await self.graph.traverse(query, top_k=top_k, threshold=threshold, owner_id=self.owner_id)

    def current_facts(self) -> list[tuple[str, str, str]]:
        return self.graph.get_current_facts(self.session_id, owner_id=self.owner_id)

    def fact_timeline(self, *, limit: int = 50) -> list[dict]:
        return self.graph.list_temporal_facts(self.session_id, owner_id=self.owner_id, limit=limit)

    def store_fact(
        self,
        *,
        subject: str,
        predicate: str,
        object_: str,
        turn: int,
        run_id: Optional[str] = None,
        supersede_existing: bool = True,
    ) -> dict:
        if supersede_existing:
            self.graph.void_temporal_fact(
                self.session_id,
                subject=subject,
                predicate=predicate,
                turn=turn,
                owner_id=self.owner_id,
            )
        self.graph.add_temporal_fact(
            self.session_id,
            subject=subject,
            predicate=predicate,
            object_=object_,
            turn=turn,
            owner_id=self.owner_id,
            run_id=run_id,
        )
        return {
            "session_id": self.session_id,
            "subject": subject,
            "predicate": predicate,
            "object": object_,
            "turn": turn,
        }

    def apply_user_message(
        self,
        message: str,
        *,
        turn: int,
        run_id: Optional[str] = None,
    ) -> dict:
        facts, voids = extract_user_stated_fact_updates(
            message,
            current_facts=self.current_facts(),
        )

        for void in voids:
            self.graph.void_temporal_fact(
                self.session_id,
                subject=str(void.get("subject") or ""),
                predicate=str(void.get("predicate") or ""),
                turn=turn,
                owner_id=self.owner_id,
            )

        for fact in facts:
            self.graph.add_temporal_fact(
                self.session_id,
                subject=str(fact.get("subject") or ""),
                predicate=str(fact.get("predicate") or ""),
                object_=str(fact.get("object") or ""),
                turn=turn,
                owner_id=self.owner_id,
                run_id=run_id,
            )

        return {"facts": facts, "voids": voids}

    def inspect(self) -> dict:
        if self.session_manager is None:
            raise ValueError("inspect() requires a SessionManager instance")
        session = self.session_manager.get(self.session_id, owner_id=self.owner_id)
        if session is None:
            raise ValueError(f"Session not found: {self.session_id}")
        return build_session_memory_payload(
            session=session,
            owner_id=self.owner_id or "",
            graph=self.graph,
            context_preview=_simple_context_preview,
        )
