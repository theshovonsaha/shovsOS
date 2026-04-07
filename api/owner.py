from __future__ import annotations

from typing import Optional

from fastapi import HTTPException


def require_owner_id(owner_id: Optional[str]) -> str:
    normalized = (owner_id or "").strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="owner_id is required")
    return normalized
