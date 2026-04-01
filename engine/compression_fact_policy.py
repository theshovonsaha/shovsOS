from __future__ import annotations

from typing import Iterable, Optional

from engine.deterministic_facts import filter_redundant_user_alias_facts
from engine.fact_guard import filter_grounded_fact_records


def finalize_compression_fact_records(
    records: list[dict],
    *,
    user_message: str,
    grounding_text: Optional[str],
    deterministic_facts: Iterable[dict],
    current_facts: Optional[Iterable[tuple[str, str, str]]],
) -> tuple[list[dict], list[dict]]:
    allowed_records, blocked_records = filter_grounded_fact_records(
        records,
        user_message=user_message,
        grounding_text=grounding_text,
    )
    allowed_records, alias_blocked_records = filter_redundant_user_alias_facts(
        allowed_records,
        deterministic_facts=deterministic_facts,
        current_facts=current_facts,
    )
    return allowed_records, [*blocked_records, *alias_blocked_records]
