from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from config.logger import log
from engine.fact_guard import is_grounded_fact_record

_MANIFEST_RE = re.compile(r"\[MANIFEST:\s*([^\]]+)\]", re.IGNORECASE | re.DOTALL)
_LOAD_RE = re.compile(r"LOAD\s*=\s*([^\|]+?)(?:\s*\||$)", re.IGNORECASE)
_FACT_RE = re.compile(r"FACT\s*=\s*([^\|]+?)\|([^\|]+?)\|([^\|]+?)(?:\s*\||$)", re.IGNORECASE)
_VOIDS_RE = re.compile(r"VOIDS\s*=\s*([^\|]+?)\|([^\|]+?)(?:\s*\||$)", re.IGNORECASE)
_GOAL_RE = re.compile(r"GOAL\s*=\s*([^\|]+?)(?:\s*\||$)", re.IGNORECASE)


@dataclass
class Manifest:
    load: Optional[str] = None
    fact: Optional[tuple[str, str, str]] = None
    voids: Optional[tuple[str, str]] = None
    goal: Optional[str] = None
    raw: str = ""

    @property
    def is_trivial(self) -> bool:
        return (self.load or "").strip().lower() == "conversational"


class ManifestParser:
    def extract(self, response: str) -> Optional[Manifest]:
        if not response:
            return None
        match = _MANIFEST_RE.search(response)
        if not match:
            return None

        body = match.group(1)
        manifest = Manifest(raw=match.group(0))

        load_match = _LOAD_RE.search(body)
        if load_match:
            manifest.load = load_match.group(1).strip()

        fact_match = _FACT_RE.search(body)
        if fact_match:
            manifest.fact = (
                fact_match.group(1).strip(),
                fact_match.group(2).strip(),
                fact_match.group(3).strip(),
            )

        voids_match = _VOIDS_RE.search(body)
        if voids_match:
            manifest.voids = (
                voids_match.group(1).strip(),
                voids_match.group(2).strip(),
            )

        goal_match = _GOAL_RE.search(body)
        if goal_match:
            manifest.goal = goal_match.group(1).strip()

        return manifest

    def strip(self, response: str) -> str:
        return _MANIFEST_RE.sub("", response or "").rstrip()

    async def store(
        self,
        manifest: Manifest,
        session_id: str,
        turn: int,
        agent_id: str = "default",
        owner_id: Optional[str] = None,
        run_id: Optional[str] = None,
        user_message: str = "",
        grounding_text: str = "",
    ) -> None:
        if not manifest or manifest.is_trivial:
            return

        try:
            from memory.semantic_graph import SemanticGraph

            graph = SemanticGraph()
            if manifest.voids:
                graph.void_temporal_fact(
                    session_id=session_id,
                    subject=manifest.voids[0],
                    predicate=manifest.voids[1],
                    turn=turn,
                    owner_id=owner_id,
                )

            if manifest.fact:
                candidate = {
                    "subject": manifest.fact[0],
                    "predicate": manifest.fact[1],
                    "object": manifest.fact[2],
                    "fact": " ".join(manifest.fact).strip(),
                }
                grounded, _ = is_grounded_fact_record(
                    candidate,
                    user_message=user_message,
                    grounding_text=grounding_text,
                )
                if not grounded:
                    manifest = Manifest(
                        load=manifest.load,
                        fact=None,
                        voids=manifest.voids,
                        goal=manifest.goal,
                        raw=manifest.raw,
                    )
                else:
                    current_facts = set(graph.get_current_facts(session_id, owner_id=owner_id))
                    if manifest.fact not in current_facts:
                        graph.add_temporal_fact(
                            session_id=session_id,
                            subject=manifest.fact[0],
                            predicate=manifest.fact[1],
                            object_=manifest.fact[2],
                            turn=turn,
                            owner_id=owner_id,
                            run_id=run_id,
                        )
                        await graph.add_triplet(
                            subject=manifest.fact[0],
                            predicate=manifest.fact[1],
                            object_=manifest.fact[2],
                            owner_id=owner_id,
                            run_id=run_id,
                        )
        except Exception as exc:
            log("manifest", session_id, f"Failed to store semantic manifest data: {exc}", level="warn")

        if manifest.load:
            try:
                from memory.vector_engine import VectorEngine

                await VectorEngine(session_id=session_id, agent_id=agent_id, owner_id=owner_id).index(
                    key=manifest.load,
                    anchor=f"[turn:{turn}] {manifest.load}",
                    metadata={
                        "turn": turn,
                        "goal": manifest.goal or "",
                        "has_fact": bool(manifest.fact),
                        "source": "manifest",
                        "run_id": run_id or "",
                        "owner_id": owner_id or "",
                    },
                )
            except Exception as exc:
                log("manifest", session_id, f"Failed to index manifest load concept: {exc}", level="warn")


_parser: Optional[ManifestParser] = None


def get_manifest_parser() -> ManifestParser:
    global _parser
    if _parser is None:
        _parser = ManifestParser()
    return _parser
