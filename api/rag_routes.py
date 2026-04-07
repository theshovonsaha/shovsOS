"""
RAG API Routes
--------------
File upload and management endpoints for per-session RAG.

Mount in main.py:
    from api.rag_routes import make_rag_router
    app.include_router(make_rag_router(), prefix="/rag")

Endpoints:
    POST /rag/{session_id}/upload        — upload file(s) into session RAG
    GET  /rag/{session_id}/search        — search session RAG
    GET  /rag/{session_id}/stats         — count indexed chunks
    DELETE /rag/{session_id}             — wipe session RAG collection
"""

from __future__ import annotations

import io
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from memory.session_rag import get_session_rag, cleanup_session_rag
from api.owner import require_owner_id


# ── Supported file types ───────────────────────────────────────────────────────

SUPPORTED_TYPES = {
    ".txt":  "text/plain",
    ".md":   "text/markdown",
    ".pdf":  "application/pdf",
    ".csv":  "text/csv",
    ".json": "application/json",
    ".py":   "text/x-python",
    ".js":   "text/javascript",
    ".ts":   "text/typescript",
    ".html": "text/html",
    ".xml":  "text/xml",
    ".yaml": "text/yaml",
    ".yml":  "text/yaml",
}


def _extract_text(filename: str, raw: bytes) -> str:
    """Extract plain text from various file types."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                return "\n\n".join(
                    page.extract_text() or "" for page in pdf.pages
                ).strip()
        except ImportError:
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(raw))
                return "\n\n".join(
                    page.extract_text() or "" for page in reader.pages
                ).strip()
            except ImportError:
                return raw.decode("utf-8", errors="ignore")

    # All other text-based files
    return raw.decode("utf-8", errors="ignore")


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


def make_rag_router() -> APIRouter:
    router = APIRouter(tags=["rag"])
    _require_owner_id = require_owner_id

    @router.post("/{session_id}/upload")
    async def upload_to_rag(
        session_id: str,
        owner_id: Optional[str] = Query(default=None),
        files: List[UploadFile] = File(...),
    ):
        """
        Upload one or more files into a session's RAG store.
        Supported: .txt, .md, .pdf, .csv, .json, .py, .js, .ts, .html
        """
        owner_id = _require_owner_id(owner_id)
        rag = get_session_rag(session_id, owner_id=owner_id)
        results = []

        for upload in files:
            filename = upload.filename or "unknown"
            ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

            if ext not in SUPPORTED_TYPES:
                results.append({
                    "filename": filename,
                    "success": False,
                    "error": f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_TYPES.keys())}",
                })
                continue

            try:
                raw = await upload.read()
                text = _extract_text(filename, raw)

                if not text or len(text.strip()) < 20:
                    results.append({
                        "filename": filename,
                        "success": False,
                        "error": "File appears empty or could not be extracted.",
                    })
                    continue

                chunks_indexed = await rag.index_file(filename, text)
                results.append({
                    "filename":       filename,
                    "success":        True,
                    "chunks_indexed": chunks_indexed,
                    "chars_extracted": len(text),
                })

            except Exception as e:
                results.append({
                    "filename": filename,
                    "success": False,
                    "error": str(e),
                })

        total_chunks = sum(r.get("chunks_indexed", 0) for r in results)
        return {
            "session_id":   session_id,
            "files":        results,
            "total_chunks": total_chunks,
            "message": (
                f"Indexed {total_chunks} chunks into session RAG. "
                "The agent can now search this content with rag_search."
            ),
        }

    @router.get("/{session_id}/search")
    async def search_rag(
        session_id: str,
        query:      str   = Query(..., min_length=1),
        top_k:      int   = Query(5, ge=1, le=20),
        owner_id:   Optional[str] = Query(default=None),
    ):
        """Search a session's RAG store."""
        owner_id = _require_owner_id(owner_id)
        rag     = get_session_rag(session_id, owner_id=owner_id)
        results = await rag.query(query, top_k=top_k)
        return {
            "session_id": session_id,
            "query":      query,
            "results":    results,
            "count":      len(results),
        }

    @router.get("/{session_id}/stats")
    async def rag_stats(session_id: str, owner_id: Optional[str] = Query(default=None)):
        """Return stats about a session's RAG store."""
        owner_id = _require_owner_id(owner_id)
        rag   = get_session_rag(session_id, owner_id=owner_id)
        count = await rag.count()
        return {
            "session_id":     session_id,
            "chunks_indexed": count,
            "available":      rag._is_available(),
        }

    @router.delete("/{session_id}")
    async def clear_rag(session_id: str, owner_id: Optional[str] = Query(default=None)):
        """Wipe a session's RAG collection."""
        try:
            owner_id = _require_owner_id(owner_id)
            from memory.session_rag import _get_chroma_client
            client = _get_chroma_client()
            rag = get_session_rag(session_id, owner_id=owner_id)
            client.delete_collection(rag.collection_name)
            cleanup_session_rag(session_id, owner_id=owner_id)
            return {"session_id": session_id, "status": "cleared"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router
