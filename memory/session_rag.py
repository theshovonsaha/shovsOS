"""
Session RAG
-----------
Per-session vector store using ChromaDB.
Each session gets its own isolated collection.

Every tool result is auto-indexed. The LLM can then call
`rag_search` to retrieve information from earlier in the
conversation — avoiding redundant web fetches and building
a growing knowledge base within the session.

Also supports file uploads: PDFs, text, markdown are chunked
and indexed into the session collection on upload.

Requirements:
    pip install chromadb sentence-transformers

Architecture:
    - One Chroma collection per session: "session_{session_id}"
    - Each document has metadata: {source, tool_name, timestamp, chunk_index}
    - Embedding model: all-MiniLM-L6-v2 (local, fast, no API key)
    - Collections are persisted to ./data/chroma/
    - Collections are automatically cleaned up after 7 days of inactivity
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

CHROMA_DIR    = Path(os.getenv("CHROMA_DIR", "./data/chroma")).resolve()
EMBED_MODEL   = os.getenv("SESSION_RAG_EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE    = int(os.getenv("SESSION_RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("SESSION_RAG_CHUNK_OVERLAP", str(max(120, CHUNK_SIZE // 5))))
MAX_RESULTS   = int(os.getenv("SESSION_RAG_MAX_RESULTS", "5"))

CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def _get_chroma_client():
    """Lazy-loaded Chroma client (persistent)."""
    import chromadb
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _get_embedding_fn():
    """Lazy-loaded sentence-transformers embedding function."""
    from chromadb.utils import embedding_functions
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks for better retrieval."""
    if len(text) <= chunk_size:
        return [text]

    overlap = max(0, min(overlap, max(1, chunk_size // 3)))
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Prefer paragraph breaks, then sentence boundaries, before falling back.
        if end < len(text):
            last_paragraph = text.rfind("\n\n", start, end)
            last_period = text.rfind(".", start, end)
            last_newline = text.rfind("\n", start, end)
            break_at = max(last_paragraph, last_period, last_newline)
            if break_at > start + (chunk_size // 2):
                end = break_at + (2 if break_at == last_paragraph else 1)
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c.strip()]


class SessionRAG:
    """
    Per-session RAG store. One instance per session.
    Thread-safe for async use via executor pattern.
    """

    def __init__(self, session_id: str, owner_id: Optional[str] = None):
        self.session_id    = session_id
        self.owner_id      = owner_id
        owner_key = (owner_id or "global").replace("-", "_")
        self.collection_name = f"owner_{owner_key}_session_{session_id.replace('-', '_')}"
        self._collection   = None
        self._available    = None   # None = untested, True/False = cached

    def _is_available(self) -> bool:
        """Check if ChromaDB is installed."""
        if self._available is None:
            try:
                import chromadb
                import sentence_transformers
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def _get_collection(self):
        """Get or create the Chroma collection for this session."""
        if self._collection is None:
            client = _get_chroma_client()
            embed_fn = _get_embedding_fn()
            self._collection = client.get_or_create_collection(
                name               = self.collection_name,
                embedding_function = embed_fn,
                metadata           = {
                    "session_id": self.session_id,
                    "owner_id": self.owner_id or "",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        return self._collection

    async def index(
        self,
        content:   str,
        source:    str,
        tool_name: Optional[str] = None,
        filename:  Optional[str] = None,
        content_type: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> int:
        """
        Index content into this session's RAG store.
        Returns number of chunks indexed.
        Content is chunked automatically.
        """
        if not self._is_available() or not content or len(content.strip()) < 50:
            return 0

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(
                None,
                lambda: self._index_sync(content, source, tool_name, filename, content_type, run_id)
            )
            return count
        except Exception as e:
            print(f"[SessionRAG] Index error for {self.session_id}: {e}")
            return 0

    def _index_sync(
        self,
        content:   str,
        source:    str,
        tool_name: Optional[str],
        filename:  Optional[str],
        content_type: Optional[str],
        run_id: Optional[str],
    ) -> int:
        collection = self._get_collection()
        chunks     = _chunk_text(content)
        now        = datetime.now(timezone.utc).isoformat()

        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            # Stable ID: prevents re-indexing identical content
            chunk_id = hashlib.md5(f"{self.session_id}:{source}:{i}:{chunk[:64]}".encode()).hexdigest()
            ids.append(chunk_id)
            docs.append(chunk)
            metas.append({
                "source":      source,
                "tool_name":   tool_name or "",
                "filename":    filename or "",
                "content_type": content_type or "",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "indexed_at":  now,
                "owner_id":    self.owner_id or "",
                "run_id":      run_id or "",
            })

        if ids:
            # upsert prevents duplicates on re-index
            collection.upsert(ids=ids, documents=docs, metadatas=metas)

        return len(ids)

    async def query(
        self,
        query:   str,
        top_k:   int = MAX_RESULTS,
        source_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Search this session's RAG store.
        Returns list of {content, source, tool_name, filename, score}.
        """
        if not self._is_available():
            return []

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._query_sync(query, top_k, source_filter)
            )
            return results
        except Exception as e:
            print(f"[SessionRAG] Query error for {self.session_id}: {e}")
            return []

    def _query_sync(
        self,
        query:         str,
        top_k:         int,
        source_filter: Optional[str],
    ) -> list[dict]:
        collection = self._get_collection()

        # Check if collection has any documents
        if collection.count() == 0:
            return []

        where = {"source": source_filter} if source_filter else None

        results = collection.query(
            query_texts    = [query],
            n_results      = min(top_k, collection.count()),
            where          = where,
            include        = ["documents", "metadatas", "distances"],
        )

        output = []
        docs   = results.get("documents", [[]])[0]
        metas  = results.get("metadatas", [[]])[0]
        dists  = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            # Convert distance to similarity score (0-1, higher is better)
            score = max(0.0, 1.0 - (dist / 2.0))
            output.append({
                "content":   doc,
                "source":    meta.get("source", ""),
                "tool_name": meta.get("tool_name", ""),
                "filename":  meta.get("filename", ""),
                "content_type": meta.get("content_type", ""),
                "indexed_at": meta.get("indexed_at", ""),
                "score":     round(score, 3),
            })

        try:
            from memory.reranker import rerank
            output = rerank(query, output, text_key="content", top_n=top_k)
        except Exception:
            pass

        return output

    async def index_file(self, filename: str, content: str) -> int:
        """Index an uploaded file into this session's RAG store."""
        return await self.index(
            content   = content,
            source    = f"uploaded_file:{filename}",
            tool_name = "file_upload",
            filename  = filename,
            content_type = "uploaded_file",
        )

    async def count(self) -> int:
        """Return number of indexed chunks in this session."""
        if not self._is_available():
            return 0
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self._get_collection().count()
            )
        except Exception:
            return 0

    async def clear(self) -> None:
        if not self._is_available():
            return
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._clear_sync)
        except Exception:
            pass

    def _clear_sync(self) -> None:
        try:
            client = _get_chroma_client()
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = None

    def format_results_for_llm(self, results: list[dict], query: str) -> str:
        """Format RAG results into a clean string for LLM injection."""
        if not results:
            return f"No relevant information found in session memory for: '{query}'"

        parts = [f"Session memory search results for: '{query}'\n"]
        for i, r in enumerate(results, 1):
            source_label = r.get("filename") or r.get("source") or r.get("tool_name") or "unknown"
            content = str(r.get("content", "")).strip()
            if len(content) > 500:
                content = content[:500].rstrip() + "..."
            content_type = r.get("content_type") or "evidence"
            parts.append(
                f"[{i}] Source: {source_label} type={content_type} (relevance: {r['score']:.0%})\n"
                f"{content}\n"
            )
        return "\n---\n".join(parts)


# ── Global registry of active SessionRAG instances ────────────────────────────

_session_rag_registry: dict[tuple[str, str], SessionRAG] = {}


def get_session_rag(session_id: str, owner_id: Optional[str] = None) -> SessionRAG:
    """Get or create a SessionRAG instance for a session."""
    key = (owner_id or "", session_id)
    if key not in _session_rag_registry:
        _session_rag_registry[key] = SessionRAG(session_id, owner_id=owner_id)
    return _session_rag_registry[key]


def cleanup_session_rag(session_id: str, owner_id: Optional[str] = None):
    """Remove a session's RAG instance from memory (data persists on disk)."""
    _session_rag_registry.pop((owner_id or "", session_id), None)


async def clear_session_rag(session_id: str, owner_id: Optional[str] = None):
    """Delete a session's persisted RAG collection and drop its cached instance."""
    rag = get_session_rag(session_id, owner_id=owner_id)
    await rag.clear()
    cleanup_session_rag(session_id, owner_id=owner_id)


def reset_session_rag_storage(storage_dir: Optional[Path] = None):
    """Clear all session-RAG collections and recreate the backing directory."""
    _session_rag_registry.clear()
    target_dir = Path(storage_dir) if storage_dir is not None else CHROMA_DIR
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)
