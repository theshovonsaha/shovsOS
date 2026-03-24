"""
Semantic Graph DB
-----------------
A lightweight Hybrid SQLite + Vector Knowledge Graph.
Stores Subject-Predicate-Object triplets and their vector embeddings
for fuzzy retrieval ("Deep Memory").

Uses Ollama 'nomic-embed-text' for embedding generation.
"""

import sqlite3
import json
import asyncio
import threading
from collections import OrderedDict
import numpy as np
import httpx
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from config.config import cfg


class SemanticGraph:
    _http_clients: dict[tuple[str, float], httpx.AsyncClient] = {}
    _embedding_cache: "OrderedDict[str, List[float]]" = OrderedDict()
    _cache_lock = threading.RLock()

    def __init__(self, db_path: str = "memory_graph.db", embedding_model: str = "nomic-embed-text"):
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.embedding_timeout = float(getattr(cfg, "EMBEDDING_HTTP_TIMEOUT", 20.0))
        self.embedding_retries = max(0, int(getattr(cfg, "EMBEDDING_HTTP_RETRIES", 2)))
        self.embedding_cache_size = max(64, int(getattr(cfg, "EMBEDDING_CACHE_SIZE", 512)))
        self._init_db()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").strip().split())

    @classmethod
    def _get_http_client(cls, base_url: str, timeout: float) -> httpx.AsyncClient:
        key = (base_url.rstrip("/"), float(timeout))
        client = cls._http_clients.get(key)
        if client is None or client.is_closed:
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=100, keepalive_expiry=30.0)
            timeout_cfg = httpx.Timeout(
                timeout=float(timeout),
                connect=min(5.0, float(timeout)),
                read=float(timeout),
                write=float(timeout),
                pool=min(5.0, float(timeout)),
            )
            client = httpx.AsyncClient(base_url=key[0], limits=limits, timeout=timeout_cfg)
            cls._http_clients[key] = client
        return client

    @classmethod
    def _cache_get(cls, key: str) -> Optional[List[float]]:
        with cls._cache_lock:
            value = cls._embedding_cache.get(key)
            if value is not None:
                cls._embedding_cache.move_to_end(key)
            return value

    @classmethod
    def _cache_set(cls, key: str, value: List[float], max_size: int):
        with cls._cache_lock:
            cls._embedding_cache[key] = value
            cls._embedding_cache.move_to_end(key)
            while len(cls._embedding_cache) > max_size:
                cls._embedding_cache.popitem(last=False)

    async def _post_with_retry(
        self,
        base_url: str,
        path: str,
        payload: dict,
        headers: Optional[dict] = None,
    ) -> dict:
        client = self._get_http_client(base_url, self.embedding_timeout)
        last_error: Optional[Exception] = None

        for attempt in range(self.embedding_retries + 1):
            try:
                res = await client.post(path, json=payload, headers=headers)
                res.raise_for_status()
                return res.json()
            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code if e.response is not None else 0
                if status not in (429, 500, 502, 503, 504):
                    break
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.PoolTimeout, httpx.RemoteProtocolError) as e:
                last_error = e
            except Exception as e:
                last_error = e
                break

            if attempt < self.embedding_retries:
                await asyncio.sleep(0.2 * (2 ** attempt))

        raise RuntimeError(f"Embedding request failed after retries: {last_error}")

    def _init_db(self):
        """Initialize the SQLite schema."""
        with sqlite3.connect(self.db_path) as conn:
            # We store the embedding as a JSON string for simplicity,
            # since a personal agent DB will easily fit in memory for numpy cosine sim.
            # In a production scaled system, we would use sqlite-vec or chroma,
            # but for a portable agent OS, native SQLite + numpy is 0-dependency exact math.
            conn.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id TEXT,
                    run_id TEXT,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    owner_id TEXT,
                    run_id TEXT,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    valid_from INTEGER NOT NULL,
                    valid_to INTEGER,
                    created_at TEXT NOT NULL
                )
            ''')
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN owner_id TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN run_id TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE facts ADD COLUMN owner_id TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE facts ADD COLUMN run_id TEXT")
            except sqlite3.OperationalError:
                pass
            conn.commit()

    async def _get_embedding(self, text: str) -> List[float]:
        """Fetch an embedding from the provider."""
        normalized = self._normalize_text(text)
        model_name = str(self.embedding_model or "").strip()
        provider = ""
        clean_model = model_name
        if ":" in model_name:
            provider, clean_model = model_name.split(":", 1)
            provider = provider.strip().lower()
        elif "/" in model_name:
            maybe_provider, maybe_model = model_name.split("/", 1)
            if maybe_provider.strip().lower() in {"openai", "local_openai", "lmstudio", "llamacpp", "ollama"}:
                provider = maybe_provider.strip().lower()
                clean_model = maybe_model

        if not provider:
            llm_provider = str(cfg.LLM_PROVIDER or "").strip().lower()
            if llm_provider == "auto":
                if getattr(cfg, "LMSTUDIO_BASE_URL", ""):
                    provider = "lmstudio"
                elif getattr(cfg, "LLAMACPP_BASE_URL", ""):
                    provider = "llamacpp"
                elif getattr(cfg, "OPENAI_BASE_URL", ""):
                    provider = "local_openai"
                elif getattr(cfg, "OPENAI_API_KEY", ""):
                    provider = "openai"
                else:
                    provider = "ollama"
            else:
                provider = llm_provider or "ollama"

        cache_key = f"{provider}:{self.embedding_model}:{normalized}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if provider in {"openai", "local_openai", "lmstudio", "llamacpp"}:
            if provider == "lmstudio":
                base_url = getattr(cfg, "LMSTUDIO_BASE_URL", "") or "http://127.0.0.1:1234/v1"
                api_key = getattr(cfg, "LMSTUDIO_API_KEY", "") or "lm-studio"
            elif provider == "llamacpp":
                base_url = getattr(cfg, "LLAMACPP_BASE_URL", "") or "http://127.0.0.1:8080/v1"
                api_key = getattr(cfg, "LLAMACPP_API_KEY", "") or "llama.cpp"
            elif provider == "local_openai":
                base_url = getattr(cfg, "OPENAI_BASE_URL", "") or "http://127.0.0.1:1234/v1"
                api_key = getattr(cfg, "OPENAI_API_KEY", "") or "local"
            else:
                base_url = "https://api.openai.com"
                api_key = getattr(cfg, "OPENAI_API_KEY", "")
            headers = {"Authorization": f"Bearer {api_key}"}
            payload = {"model": clean_model or "text-embedding-3-small", "input": normalized}
            data = await self._post_with_retry(base_url.rstrip("/"), "/v1/embeddings", payload, headers=headers)
            embedding = data.get("data", [{}])[0].get("embedding")
        else:
            base_url = cfg.OLLAMA_BASE_URL or "http://localhost:11434"
            try:
                payload = {"model": clean_model or self.embedding_model, "input": normalized}
                data = await self._post_with_retry(base_url, "/api/embed", payload)
                embeddings = data.get("embeddings")
                embedding = embeddings[0] if isinstance(embeddings, list) and embeddings else None
            except RuntimeError as e:
                if "404" not in str(e):
                    raise
                payload = {"model": clean_model or self.embedding_model, "prompt": normalized}
                data = await self._post_with_retry(base_url, "/api/embeddings", payload)
                embedding = data.get("embedding")

        if not isinstance(embedding, list):
            raise RuntimeError("Embedding payload missing vector data.")

        self._cache_set(cache_key, embedding, self.embedding_cache_size)
        return embedding

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    async def add_triplet(
        self,
        subject: str,
        predicate: str,
        object_: str,
        owner_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> int:
        """
        Embed the relationship and store the triplet.
        We embed the string: "subject predicate object"
        """
        text_to_embed = f"{subject} {predicate} {object_}"
        try:
            vector = await self._get_embedding(text_to_embed)
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {e}")

        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO memories (owner_id, run_id, subject, predicate, object, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (owner_id, run_id, subject.strip(), predicate.strip(), object_.strip(), json.dumps(vector), now))
            conn.commit()
            return cursor.lastrowid

    async def traverse(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5,
        owner_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Perform a semantic traversal:
        1. Embed the query.
        2. Calculate cosine similarity against all stored memories in memory.
        3. Return the top_k matching relationships.
        """
        try:
            query_vector = await self._get_embedding(query)
        except Exception as e:
            print(f"[SemanticGraph] Embedding error: {e}")
            return []

        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if owner_id is None:
                cursor.execute("SELECT id, subject, predicate, object, embedding, created_at FROM memories")
            else:
                cursor.execute(
                    "SELECT id, subject, predicate, object, embedding, created_at FROM memories "
                    "WHERE COALESCE(owner_id, '') = COALESCE(?, '')",
                    (owner_id,),
                )
            all_memories = cursor.fetchall()

            for row in all_memories:
                m_id, sub, pred, obj, emb_json, created = row
                try:
                    db_vector = json.loads(emb_json)
                    sim = self._cosine_similarity(query_vector, db_vector)
                    if sim >= threshold:
                        results.append({
                            "id": m_id,
                            "subject": sub,
                            "predicate": pred,
                            "object": obj,
                            "similarity": round(sim, 3),
                            "created_at": created
                        })
                except Exception:
                    continue  # Skip corrupted rows

        # Sort by most similar first
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def list_all(self, limit: int = 100, owner_id: Optional[str] = None) -> list[dict]:
        """Return all stored memories, newest first."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if owner_id is None:
                cursor.execute(
                    "SELECT id, subject, predicate, object, created_at FROM memories ORDER BY id DESC LIMIT ?",
                    (limit,)
                )
            else:
                cursor.execute(
                    "SELECT id, subject, predicate, object, created_at FROM memories "
                    "WHERE COALESCE(owner_id, '') = COALESCE(?, '') ORDER BY id DESC LIMIT ?",
                    (owner_id, limit),
                )
            rows = cursor.fetchall()
        return [
            {"id": r[0], "subject": r[1], "predicate": r[2], "object": r[3], "created_at": r[4]}
            for r in rows
        ]

    def delete_by_id(self, memory_id: int, owner_id: Optional[str] = None) -> bool:
        """Delete a single memory by its ID. Returns True if deleted."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if owner_id is None:
                cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            else:
                cursor.execute(
                    "DELETE FROM memories WHERE id = ? AND COALESCE(owner_id, '') = COALESCE(?, '')",
                    (memory_id, owner_id),
                )
            conn.commit()
            return cursor.rowcount > 0

    def count(self, owner_id: Optional[str] = None) -> int:
        """Return total number of stored memories."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if owner_id is None:
                cursor.execute("SELECT COUNT(*) FROM memories")
            else:
                cursor.execute(
                    "SELECT COUNT(*) FROM memories WHERE COALESCE(owner_id, '') = COALESCE(?, '')",
                    (owner_id,),
                )
            return cursor.fetchone()[0]

    def clear(self, owner_id: Optional[str] = None):
        """Wipe the graph."""
        with sqlite3.connect(self.db_path) as conn:
            if owner_id is None:
                conn.execute("DELETE FROM memories")
                conn.execute("DELETE FROM facts")
            else:
                conn.execute(
                    "DELETE FROM memories WHERE COALESCE(owner_id, '') = COALESCE(?, '')",
                    (owner_id,),
                )
                conn.execute(
                    "DELETE FROM facts WHERE COALESCE(owner_id, '') = COALESCE(?, '')",
                    (owner_id,),
                )
            conn.commit()

    # ── Temporal Fact Logic (Deterministic Memory) ─────────────────────────
    def add_temporal_fact(
        self,
        session_id: str,
        subject: str,
        predicate: str,
        object_: str,
        turn: int,
        owner_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        """Insert a new fact valid from exactly this turn forward."""
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO facts (session_id, owner_id, run_id, subject, predicate, object, valid_from, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, owner_id, run_id, subject.strip(), predicate.strip(), object_.strip(), turn, now))
            conn.commit()

    def void_temporal_fact(
        self,
        session_id: str,
        subject: str,
        predicate: str,
        turn: int,
        owner_id: Optional[str] = None,
    ):
        """Invalidate the most recent fact matching 'Subject Predicate' starting exactly on this turn."""
        with sqlite3.connect(self.db_path) as conn:
            if owner_id is None:
                conn.execute('''
                    UPDATE facts 
                    SET valid_to = ? 
                    WHERE session_id = ? AND subject = ? AND predicate = ? AND valid_to IS NULL
                ''', (turn, session_id, subject.strip(), predicate.strip()))
            else:
                conn.execute('''
                    UPDATE facts 
                    SET valid_to = ? 
                    WHERE session_id = ? AND COALESCE(owner_id, '') = COALESCE(?, '')
                      AND subject = ? AND predicate = ? AND valid_to IS NULL
                ''', (turn, session_id, owner_id, subject.strip(), predicate.strip()))
            conn.commit()

    def get_current_facts(self, session_id: str, owner_id: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """Return all currently valid facts (un-voided) for this session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if owner_id is None:
                cursor.execute('''
                    SELECT subject, predicate, object 
                    FROM facts 
                    WHERE session_id = ? AND valid_to IS NULL
                    ORDER BY valid_from ASC
                ''', (session_id,))
            else:
                cursor.execute('''
                    SELECT subject, predicate, object 
                    FROM facts 
                    WHERE session_id = ? AND COALESCE(owner_id, '') = COALESCE(?, '') AND valid_to IS NULL
                    ORDER BY valid_from ASC
                ''', (session_id, owner_id))
            return cursor.fetchall()
