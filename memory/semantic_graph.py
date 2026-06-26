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
import hashlib
from collections import OrderedDict
import numpy as np
import httpx
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from config.config import cfg


class SemanticGraph:
    _http_clients: dict[tuple[str, float, int | None], httpx.AsyncClient] = {}
    _embedding_cache: "OrderedDict[str, List[float]]" = OrderedDict()
    _cache_lock = threading.RLock()

    def __init__(self, db_path: str = "memory_graph.db", embedding_model: Optional[str] = None):
        self.db_path = db_path
        self.embedding_model = embedding_model or cfg.EMBED_MODEL
        self.embedding_timeout = float(getattr(cfg, "EMBEDDING_HTTP_TIMEOUT", 20.0))
        self.embedding_retries = max(0, int(getattr(cfg, "EMBEDDING_HTTP_RETRIES", 2)))
        self.embedding_cache_size = max(64, int(getattr(cfg, "EMBEDDING_CACHE_SIZE", 512)))
        self._embed_available: bool = True
        self._embed_failure_count: int = 0
        self._init_db()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").strip().split())

    @staticmethod
    def _fact_memory_type(subject: str, predicate: str) -> str:
        pred = (predicate or "").strip().lower()
        if pred in {
            "preferred_name",
            "location",
            "timezone",
            "preferred_editor",
            "package_manager",
            "primary_language",
            "response_verbosity",
            "operating_system",
            "pronouns",
        }:
            return "preference" if pred in {"preferred_editor", "package_manager", "response_verbosity"} else "fact"
        if pred in {
            "environment_mode",
            "scope_boundary",
            "budget_limit",
            "task_constraint",
            "followup_directive",
        }:
            return "policy" if pred in {"task_constraint", "scope_boundary", "budget_limit"} else "episodic"
        if (subject or "").strip().lower() in {"policy", "rule", "constraint"}:
            return "policy"
        return "fact"

    @staticmethod
    def _content_hash(subject: str, predicate: str, object_: str, owner_id: Optional[str], session_id: str) -> str:
        payload = "\x1f".join([
            str(owner_id or ""),
            str(session_id or ""),
            (subject or "").strip().lower(),
            (predicate or "").strip().lower(),
            (object_ or "").strip(),
        ])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def _get_http_client(cls, base_url: str, timeout: float) -> httpx.AsyncClient:
        try:
            loop_id = id(asyncio.get_running_loop())
        except RuntimeError:
            loop_id = None
        key = (base_url.rstrip("/"), float(timeout), loop_id)
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
            conn.execute('''
                CREATE TABLE IF NOT EXISTS loci (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT,
                    name TEXT NOT NULL,
                    description TEXT,
                    compiled_drawer TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id TEXT,
                    run_id TEXT,
                    locus_id TEXT,
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
                    locus_id TEXT,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'fact',
                    memory_status TEXT NOT NULL DEFAULT 'active',
                    confidence REAL,
                    source_turn_id INTEGER,
                    superseded_by INTEGER,
                    expires_at TEXT,
                    content_hash TEXT,
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
                conn.execute("ALTER TABLE memories ADD COLUMN locus_id TEXT")
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
            try:
                conn.execute("ALTER TABLE facts ADD COLUMN locus_id TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE facts ADD COLUMN conflict_trace INTEGER NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE facts ADD COLUMN prior_value_disputed INTEGER NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE facts ADD COLUMN conflict_reason TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE facts ADD COLUMN memory_type TEXT NOT NULL DEFAULT 'fact'")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE facts ADD COLUMN memory_status TEXT NOT NULL DEFAULT 'active'")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE facts ADD COLUMN confidence REAL")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE facts ADD COLUMN source_turn_id INTEGER")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE facts ADD COLUMN superseded_by INTEGER")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE facts ADD COLUMN expires_at TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE facts ADD COLUMN content_hash TEXT")
            except sqlite3.OperationalError:
                pass
            conn.execute('''
                CREATE TABLE IF NOT EXISTS locus_edges (
                    src_id TEXT NOT NULL,
                    dst_id TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (src_id, dst_id)
                )
            ''')
            # M1: hot-path indexes on facts. Every context build calls
            # get_current_facts (session-scoped) and get_owner_current_facts
            # (cross-session). Without these, a 50-turn session does N full
            # table scans. The predicate-history index also covers M3's
            # get_predicate_history reader.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_owner_session_valid "
                "ON facts(owner_id, session_id, valid_to)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_owner_predicate "
                "ON facts(owner_id, subject, predicate, valid_to)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_owner_valid_from "
                "ON facts(owner_id, valid_from DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_owner_type_status "
                "ON facts(owner_id, memory_type, memory_status, valid_to)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_content_hash "
                "ON facts(content_hash)"
            )
            conn.commit()

    def _record_embedding_failure(self, exc: Exception) -> List[float]:
        self._embed_failure_count += 1
        if self._embed_failure_count >= 2:
            if self._embed_available:
                print(f"[SemanticGraph] Embedding disabled after repeated failures: {exc}")
            self._embed_available = False
        elif self._embed_failure_count == 1:
            print(f"[SemanticGraph] Embedding failed: {exc}")
        return []

    async def _get_embedding(self, text: str) -> List[float]:
        if not self._embed_available:
            return []
        try:
            embedding = await self._get_embedding_uncached(text)
            self._embed_failure_count = 0
            self._embed_available = True
            return embedding
        except Exception as exc:
            return self._record_embedding_failure(exc)

    async def _get_embedding_uncached(self, text: str) -> List[float]:
        """Fetch an embedding from the provider."""
        normalized = self._normalize_text(text)
        model_name = str(self.embedding_model or cfg.EMBED_MODEL).strip()
        provider = ""
        clean_model = model_name

        # Detect Provider Prefix (e.g., "openai:text-embedding-3")
        known_providers = {"openai", "local_openai", "lmstudio", "llamacpp", "ollama"}
        if ":" in model_name:
            prefix, rest = model_name.split(":", 1)
            if prefix.lower() in known_providers:
                provider = prefix.lower()
                clean_model = rest
            else:
                # This was likely a model tag (e.g., "nomic-embed-text:latest")
                pass
        
        if not provider:
            provider = cfg.LLM_PROVIDER.lower() if cfg.LLM_PROVIDER != "auto" else "ollama"

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
                # 1. Try stable /api/embeddings logic with 'prompt' key
                payload = {"model": clean_model or self.embedding_model, "prompt": normalized}
                data = await self._post_with_retry(base_url, "/api/embeddings", payload)
                
                if "embedding" in data:
                    embedding = data["embedding"]
                elif "embeddings" in data:
                    embedding = data["embeddings"]
                else:
                    embedding = None

                if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
                    embedding = embedding[0]
            except RuntimeError as e:
                # 2. Fallback to /api/embed with 'input' if 404 or other failure
                payload = {"model": clean_model or self.embedding_model, "input": [normalized]}
                data = await self._post_with_retry(base_url, "/api/embed", payload)
                
                if "embeddings" in data:
                    embedding = data["embeddings"]
                elif "embedding" in data:
                    embedding = data["embedding"]
                else:
                    embedding = None

                if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
                    embedding = embedding[0]

        if not isinstance(embedding, list):
            raise RuntimeError(f"Embedding failed. Model: {clean_model}. Data: {data}")

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
        locus_id: Optional[str] = None,
    ) -> int:
        """
        Embed the relationship and store the triplet.
        We embed the string: "subject predicate object"
        """
        text_to_embed = f"search_document: {subject} {predicate} {object_}"
        vector = await self._get_embedding(text_to_embed)

        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO memories (owner_id, run_id, locus_id, subject, predicate, object, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (owner_id, run_id, locus_id, subject.strip(), predicate.strip(), object_.strip(), json.dumps(vector), now))
            conn.commit()
            return cursor.lastrowid

    async def traverse(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5,
        owner_id: Optional[str] = None,
        locus_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Perform a semantic traversal:
        1. Embed the query.
        2. Calculate cosine similarity against all stored memories in memory.
        3. Return the top_k matching relationships.
        """
        # For retrieval, we use 'search_query: ' prefix
        query_to_embed = f"search_query: {query}" if query else "search_query: "
        query_vector = await self._get_embedding(query_to_embed)
        if not query_vector:
            return []

        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query_str = "SELECT id, subject, predicate, object, embedding, created_at FROM memories WHERE 1=1"
            params = []
            if owner_id is not None:
                query_str += " AND COALESCE(owner_id, '') = COALESCE(?, '')"
                params.append(owner_id)
            if locus_id is not None:
                query_str += " AND COALESCE(locus_id, '') = COALESCE(?, '')"
                params.append(locus_id)
            
            cursor.execute(query_str, tuple(params))
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

    def list_triplets(
        self,
        *,
        owner_id: Optional[str] = None,
        locus_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return raw semantic triplets without requiring embeddings.

        This backs exact/locus fallback retrieval when the embedding provider is
        offline. It does not replace vector traversal; it preserves recall for
        scoped triplets that are already structurally known.
        """
        safe_limit = max(1, int(limit))
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = (
                "SELECT id, owner_id, run_id, locus_id, subject, predicate, object, created_at "
                "FROM memories WHERE 1=1"
            )
            params: list[object] = []
            if owner_id is not None:
                query += " AND COALESCE(owner_id, '') = COALESCE(?, '')"
                params.append(owner_id)
            if locus_id is not None:
                query += " AND COALESCE(locus_id, '') = COALESCE(?, '')"
                params.append(locus_id)
            query += " ORDER BY id DESC LIMIT ?"
            params.append(safe_limit)
            rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def delete_triplets(
        self,
        subject: str,
        predicate: str,
        owner_id: Optional[str] = None,
    ) -> int:
        """Hard-delete vector triplets matching (subject, predicate). Returns rows deleted.

        Used by the void/update path so retrieval (`traverse`) stops surfacing
        embeddings whose deterministic fact has been superseded — preventing
        the annotation-style ghost where old triplets still rank in similarity.
        """
        sub = (subject or "").strip()
        pred = (predicate or "").strip()
        if not sub or not pred:
            return 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if owner_id is None:
                cursor.execute(
                    "DELETE FROM memories WHERE subject = ? AND predicate = ?",
                    (sub, pred),
                )
            else:
                cursor.execute(
                    "DELETE FROM memories WHERE subject = ? AND predicate = ? "
                    "AND COALESCE(owner_id, '') = COALESCE(?, '')",
                    (sub, pred, owner_id),
                )
            conn.commit()
            return cursor.rowcount

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

    def clear_session_facts(self, session_id: str, owner_id: Optional[str] = None):
        """Delete deterministic fact history for one session without touching global memory."""
        with sqlite3.connect(self.db_path) as conn:
            if owner_id is None:
                conn.execute("DELETE FROM facts WHERE session_id = ?", (session_id,))
            else:
                conn.execute(
                    "DELETE FROM facts WHERE session_id = ? AND COALESCE(owner_id, '') = COALESCE(?, '')",
                    (session_id, owner_id),
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
        locus_id: Optional[str] = None,
        conflict_trace: bool = False,
        prior_value_disputed: bool = False,
        conflict_reason: Optional[str] = None,
        memory_type: Optional[str] = None,
        memory_status: str = "active",
        confidence: Optional[float] = None,
        expires_at: Optional[str] = None,
    ):
        """Insert a new fact valid from exactly this turn forward."""
        now = datetime.now().isoformat()
        subject_clean = subject.strip()
        predicate_clean = predicate.strip()
        object_clean = object_.strip()
        fact_type = str(memory_type or self._fact_memory_type(subject_clean, predicate_clean)).strip() or "fact"
        content_hash = self._content_hash(subject_clean, predicate_clean, object_clean, owner_id, session_id)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO facts (
                    session_id, owner_id, run_id, locus_id, subject, predicate,
                    object, memory_type, memory_status, confidence, source_turn_id,
                    expires_at, content_hash, valid_from, created_at, conflict_trace,
                    prior_value_disputed, conflict_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                owner_id,
                run_id,
                locus_id,
                subject_clean,
                predicate_clean,
                object_clean,
                fact_type,
                str(memory_status or "active"),
                confidence,
                turn,
                expires_at,
                content_hash,
                turn,
                now,
                1 if conflict_trace else 0,
                1 if prior_value_disputed else 0,
                conflict_reason,
            ))
            conn.commit()

    def replace_temporal_facts(
        self,
        session_id: str,
        *,
        facts: List[Dict[str, object]],
        voids: List[Dict[str, object]],
        turn: int,
        owner_id: Optional[str] = None,
        run_id: Optional[str] = None,
        locus_id: Optional[str] = None,
    ) -> None:
        """Atomically void old facts and insert replacements.

        `add_temporal_fact()` and `void_temporal_fact()` each commit their own
        connection. Replacement flows must use this batch method so a failed
        insert cannot leave the old fact voided with no current replacement.
        """
        now = datetime.now().isoformat()
        prepared_facts: list[dict] = []
        for fact in facts:
            subject = str(fact.get("subject") or "").strip()
            predicate = str(fact.get("predicate") or "").strip()
            raw_object = fact.get("object") if "object" in fact else fact.get("object_")
            if not isinstance(raw_object, (str, int, float, bool)):
                raise AttributeError("temporal fact object must be a scalar value")
            object_ = str(raw_object).strip()
            if not subject or not predicate:
                continue
            prepared_facts.append({**fact, "subject": subject, "predicate": predicate, "object": object_})

        with sqlite3.connect(self.db_path) as conn:
            for void in voids:
                subject = str(void.get("subject") or "").strip()
                predicate = str(void.get("predicate") or "").strip()
                if not subject or not predicate:
                    continue
                if owner_id is None:
                    conn.execute(
                        '''
                        UPDATE facts
                        SET valid_to = ?, memory_status = 'superseded'
                        WHERE session_id = ? AND subject = ? AND predicate = ? AND valid_to IS NULL
                        ''',
                        (turn, session_id, subject, predicate),
                    )
                else:
                    conn.execute(
                        '''
                        UPDATE facts
                        SET valid_to = ?, memory_status = 'superseded'
                        WHERE session_id = ? AND COALESCE(owner_id, '') = COALESCE(?, '')
                          AND subject = ? AND predicate = ? AND valid_to IS NULL
                        ''',
                        (turn, session_id, owner_id, subject, predicate),
                    )

            for fact in prepared_facts:
                subject = str(fact.get("subject") or "").strip()
                predicate = str(fact.get("predicate") or "").strip()
                object_ = str(fact.get("object") or "").strip()
                if not subject or not predicate:
                    continue
                fact_locus_id = str(fact.get("locus_id") or "").strip() or locus_id
                fact_run_id = str(fact.get("run_id") or "").strip() or run_id
                fact_type = str(fact.get("memory_type") or self._fact_memory_type(subject, predicate)).strip() or "fact"
                memory_status = str(fact.get("memory_status") or "active").strip() or "active"
                confidence = fact.get("confidence")
                expires_at = str(fact.get("expires_at") or "").strip() or None
                content_hash = str(fact.get("content_hash") or "").strip() or self._content_hash(subject, predicate, object_, owner_id, session_id)
                conn.execute(
                    '''
                    INSERT INTO facts (
                        session_id, owner_id, run_id, locus_id, subject, predicate,
                        object, memory_type, memory_status, confidence, source_turn_id,
                        expires_at, content_hash, valid_from, created_at, conflict_trace,
                        prior_value_disputed, conflict_reason
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        session_id,
                        owner_id,
                        fact_run_id,
                        fact_locus_id,
                        subject,
                        predicate,
                        object_,
                        fact_type,
                        memory_status,
                        confidence,
                        turn,
                        expires_at,
                        content_hash,
                        turn,
                        now,
                        1 if bool(fact.get("conflict_trace")) else 0,
                        1 if bool(fact.get("prior_value_disputed")) else 0,
                        str(fact.get("conflict_reason") or "") or None,
                    ),
                )
                new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                if owner_id is None:
                    conn.execute(
                        '''
                        UPDATE facts
                        SET superseded_by = ?
                        WHERE session_id = ? AND subject = ? AND predicate = ?
                          AND valid_to = ? AND superseded_by IS NULL AND id != ?
                        ''',
                        (new_id, session_id, subject, predicate, turn, new_id),
                    )
                else:
                    conn.execute(
                        '''
                        UPDATE facts
                        SET superseded_by = ?
                        WHERE session_id = ? AND COALESCE(owner_id, '') = COALESCE(?, '')
                          AND subject = ? AND predicate = ?
                          AND valid_to = ? AND superseded_by IS NULL AND id != ?
                        ''',
                        (new_id, session_id, owner_id, subject, predicate, turn, new_id),
                    )
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
                    SET valid_to = ?, memory_status = 'superseded'
                    WHERE session_id = ? AND subject = ? AND predicate = ? AND valid_to IS NULL
                ''', (turn, session_id, subject.strip(), predicate.strip()))
            else:
                conn.execute('''
                    UPDATE facts 
                    SET valid_to = ?, memory_status = 'superseded'
                    WHERE session_id = ? AND COALESCE(owner_id, '') = COALESCE(?, '')
                      AND subject = ? AND predicate = ? AND valid_to IS NULL
                ''', (turn, session_id, owner_id, subject.strip(), predicate.strip()))
            conn.commit()

    def get_owner_current_facts(
        self,
        owner_id: Optional[str],
        limit: int = 40,
    ) -> List[Tuple[str, str, str]]:
        """Return currently valid facts scoped to an owner across every session.

        Used by the L0 identity brief so a fresh session still carries forward
        who the owner is without needing an explicit locus query. Deduplicates
        on (subject, predicate) — the most recently valid_from wins.
        """
        safe_limit = max(1, int(limit))
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''
                SELECT subject, predicate, object
                FROM facts
                WHERE COALESCE(owner_id, '') = COALESCE(?, '') AND valid_to IS NULL
                  AND subject IS NOT NULL AND predicate IS NOT NULL
                ORDER BY valid_from DESC, id DESC
                ''',
                (owner_id,),
            )
            seen: set[tuple[str, str]] = set()
            out: List[Tuple[str, str, str]] = []
            for subject, predicate, obj in cursor.fetchall():
                key = ((subject or "").strip().lower(), (predicate or "").strip().lower())
                if not key[0] or not key[1] or key in seen:
                    continue
                seen.add(key)
                out.append((subject, predicate, obj))
                if len(out) >= safe_limit:
                    break
            return out

    def get_predicate_history(
        self,
        owner_id: Optional[str],
        subject: str,
        predicate: str,
        limit: int = 10,
    ) -> List[Dict[str, Optional[str]]]:
        """Return the full timeline for one (subject, predicate) pair across
        all sessions for this owner — including superseded (voided) facts.

        This is the **archaeology** reader. It answers questions like
        "what did I originally say about Berlin?" or "when did I correct my
        budget?". The 50-turn shovs scenario (turns 36, 38, 41, 50) requires
        this surface; without it the agent can only see current facts and
        will give half-answers to history questions.

        Returns rows ordered by ``valid_from DESC`` (most recent first):

            [
              {subject, predicate, object, valid_from, valid_to, created_at,
               run_id, status: "current"|"superseded"},
              ...
            ]

        Status is derived (``"current"`` when ``valid_to IS NULL``,
        ``"superseded"`` otherwise) so callers don't need to interpret the
        raw column.
        """
        safe_limit = max(1, int(limit))
        clean_subject = (subject or "").strip()
        clean_predicate = (predicate or "").strip()
        if not clean_subject or not clean_predicate:
            return []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                '''
                SELECT subject, predicate, object, valid_from, valid_to,
                       created_at, run_id, session_id, memory_type, memory_status,
                       confidence, source_turn_id, superseded_by, expires_at, content_hash
                FROM facts
                WHERE COALESCE(owner_id, '') = COALESCE(?, '')
                  AND lower(subject) = lower(?)
                  AND lower(predicate) = lower(?)
                ORDER BY valid_from DESC, id DESC
                LIMIT ?
                ''',
                (owner_id, clean_subject, clean_predicate, safe_limit),
            )
            out: List[Dict[str, Optional[str]]] = []
            for row in cursor.fetchall():
                d = dict(row)
                d["status"] = "current" if d.get("valid_to") is None else "superseded"
                out.append(d)
            return out

    def get_current_fact_records(
        self,
        session_id: str,
        owner_id: Optional[str] = None,
        *,
        memory_types: Optional[List[str]] = None,
        limit: Optional[int] = 100,
    ) -> List[Dict[str, object]]:
        """Return current facts with typed-memory provenance.

        This is the richer reader for prompt assembly and inspectors. The older
        get_current_facts() tuple API intentionally remains unchanged.
        """
        safe_limit = max(1, int(limit)) if limit is not None else None
        normalized_types = [str(t).strip().lower() for t in (memory_types or []) if str(t).strip()]
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            params: list[object] = [session_id]
            owner_filter = ""
            if owner_id is not None:
                owner_filter = " AND COALESCE(owner_id, '') = COALESCE(?, '')"
                params.append(owner_id)
            type_filter = ""
            if normalized_types:
                type_filter = " AND lower(memory_type) IN (" + ",".join("?" for _ in normalized_types) + ")"
                params.extend(normalized_types)
            limit_clause = ""
            if safe_limit is not None:
                limit_clause = " LIMIT ?"
                params.append(safe_limit)
            cursor = conn.execute(
                f'''
                SELECT id, session_id, owner_id, run_id, locus_id, subject, predicate, object,
                       memory_type, memory_status, confidence, source_turn_id,
                       superseded_by, expires_at, content_hash, valid_from, valid_to,
                       created_at, conflict_trace, prior_value_disputed, conflict_reason
                FROM facts
                WHERE session_id = ?{owner_filter} AND valid_to IS NULL{type_filter}
                ORDER BY valid_from ASC, id ASC{limit_clause}
                ''',
                tuple(params),
            )
            rows = cursor.fetchall()
        out: List[Dict[str, object]] = []
        for row in rows:
            item = dict(row)
            item["status"] = "current"
            item["conflict_trace"] = bool(item.get("conflict_trace"))
            item["prior_value_disputed"] = bool(item.get("prior_value_disputed"))
            out.append(item)
        return out

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

    def list_temporal_facts(
        self,
        session_id: str,
        owner_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Optional[str]]]:
        """Return the recent fact timeline for a session, including superseded facts."""
        safe_limit = max(1, int(limit))
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if owner_id is None:
                cursor.execute(
                    '''
                    SELECT id, subject, predicate, object, valid_from, valid_to,
                           created_at, run_id, conflict_trace,
                           prior_value_disputed, conflict_reason, memory_type,
                           memory_status, confidence, source_turn_id,
                           superseded_by, expires_at, content_hash
                    FROM facts
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    ''',
                    (session_id, safe_limit),
                )
            else:
                cursor.execute(
                    '''
                    SELECT id, subject, predicate, object, valid_from, valid_to,
                           created_at, run_id, conflict_trace,
                           prior_value_disputed, conflict_reason, memory_type,
                           memory_status, confidence, source_turn_id,
                           superseded_by, expires_at, content_hash
                    FROM facts
                    WHERE session_id = ? AND COALESCE(owner_id, '') = COALESCE(?, '')
                    ORDER BY id DESC
                    LIMIT ?
                    ''',
                    (session_id, owner_id, safe_limit),
                )
            rows = cursor.fetchall()

        return [
            {
                "id": row["id"],
                "subject": row["subject"],
                "predicate": row["predicate"],
                "object": row["object"],
                "valid_from": row["valid_from"],
                "valid_to": row["valid_to"],
                "created_at": row["created_at"],
                "run_id": row["run_id"],
                "memory_type": row["memory_type"],
                "memory_status": row["memory_status"],
                "confidence": row["confidence"],
                "source_turn_id": row["source_turn_id"],
                "superseded_by": row["superseded_by"],
                "expires_at": row["expires_at"],
                "content_hash": row["content_hash"],
                "conflict_trace": bool(row["conflict_trace"]),
                "prior_value_disputed": bool(row["prior_value_disputed"]),
                "conflict_reason": row["conflict_reason"],
                "status": "current" if row["valid_to"] is None else "superseded",
            }
            for row in rows
        ]

    # ── Locus / Spatial Registry Logic ──────────────────────────────────────
    def register_locus(
        self,
        locus_id: str,
        name: str,
        description: str = "",
        owner_id: Optional[str] = None,
    ):
        """Register a new spatial room (Locus) with the graph."""
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO loci (id, owner_id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, COALESCE((SELECT created_at FROM loci WHERE id = ?), ?), ?)
            ''', (locus_id, owner_id, name, description, locus_id, now, now))
            conn.commit()

    def get_locus(self, locus_id: str, owner_id: Optional[str] = None) -> Optional[Dict]:
        """Fetch details of a specific Locus."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if owner_id is None:
                cursor.execute("SELECT * FROM loci WHERE id = ?", (locus_id,))
            else:
                cursor.execute(
                    "SELECT * FROM loci WHERE id = ? AND COALESCE(owner_id, '') = COALESCE(?, '')",
                    (locus_id, owner_id),
                )
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_compiled_drawer(self, locus_id: str, compiled_markdown: str):
        """Update the Karpathy-style compiled context (Executable Wiki) for a Locus."""
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE loci SET compiled_drawer = ?, updated_at = ? WHERE id = ?
            ''', (compiled_markdown, now, locus_id))
            conn.commit()

    def get_compiled_drawer(self, locus_id: str) -> Optional[str]:
        """Get the compiled document for a specific Locus."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT compiled_drawer FROM loci WHERE id = ?", (locus_id,))
            row = cursor.fetchone()
            if row and row[0]:
                return row[0]
            return None

    def compile_locus_drawer(
        self,
        locus_id: str,
        owner_id: Optional[str] = None,
        max_facts: int = 40,
    ) -> Optional[str]:
        """Render a dense markdown summary of a locus's current facts and write
        it to the `compiled_drawer` column.

        The drawer is what `unified_memory_search` slams to score=1.0 when a
        locus is detected — without a writer the column was always NULL, so
        the high-priority lane was dead. This is the mempalace closet pattern
        adapted natively: a per-locus index doc that retrieval can pin-hit.
        """
        meta = self.get_locus(locus_id, owner_id=owner_id)
        if meta is None:
            return None
        facts = self.list_temporal_facts_by_locus(
            locus_id, owner_id=owner_id, limit=max_facts
        ) or []
        current = [f for f in facts if f.get("status") == "current"]

        lines: list[str] = []
        name = str(meta.get("name") or locus_id).strip()
        description = str(meta.get("description") or "").strip()
        lines.append(f"# {name}")
        if description:
            lines.append(description)
        if current:
            lines.append("")
            lines.append("## Current facts")
            for fact in current:
                subject = str(fact.get("subject") or "").strip()
                predicate = str(fact.get("predicate") or "").strip()
                object_ = str(fact.get("object") or "").strip()
                if not (subject and predicate and object_):
                    continue
                lines.append(f"- {subject} — {predicate}: {object_}")
        compiled = "\n".join(lines).strip()
        if not compiled:
            return None
        self.update_compiled_drawer(locus_id, compiled)
        return compiled

    def list_locus_ids(self, owner_id: Optional[str] = None) -> list[str]:
        """Distinct locus ids known to the graph, scoped by owner if given."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if owner_id is None:
                cursor.execute("SELECT id FROM loci")
            else:
                cursor.execute(
                    "SELECT id FROM loci WHERE COALESCE(owner_id, '') = COALESCE(?, '')",
                    (owner_id,),
                )
            return [row[0] for row in cursor.fetchall() if row and row[0]]

    def add_locus_edge(
        self,
        src_id: str,
        dst_id: str,
        weight: float = 1.0,
    ) -> None:
        """Record an undirected association between two loci.

        Stored as a directed pair so callers can express asymmetric weights,
        but `get_locus_neighbors` follows edges in both directions.
        """
        if not src_id or not dst_id or src_id == dst_id:
            return
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                '''
                INSERT INTO locus_edges (src_id, dst_id, weight, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(src_id, dst_id) DO UPDATE SET weight = excluded.weight
                ''',
                (src_id, dst_id, float(weight), now),
            )
            conn.commit()

    def get_locus_neighbors(
        self,
        src_id: str,
        max_depth: int = 2,
        max_fanout: int = 5,
    ) -> List[Tuple[str, int, float]]:
        """Return (neighbor_id, depth, score) for loci reachable from `src_id`.

        BFS bounded by `max_depth` (≤2) and per-level fanout (`max_fanout` ≤5).
        Score decays geometrically per hop (0.85 ** depth) and multiplies edge
        weight; nearer / heavier neighbors rank first.
        """
        if not src_id:
            return []
        depth_cap = max(1, min(int(max_depth or 1), 3))
        fanout_cap = max(1, min(int(max_fanout or 5), 10))
        seen = {src_id}
        frontier: list[tuple[str, float]] = [(src_id, 1.0)]
        out: list[tuple[str, int, float]] = []
        with sqlite3.connect(self.db_path) as conn:
            for depth in range(1, depth_cap + 1):
                next_frontier: list[tuple[str, float]] = []
                for node, parent_score in frontier:
                    cursor = conn.execute(
                        '''
                        SELECT dst_id AS other, weight FROM locus_edges WHERE src_id = ?
                        UNION
                        SELECT src_id AS other, weight FROM locus_edges WHERE dst_id = ?
                        ORDER BY weight DESC
                        LIMIT ?
                        ''',
                        (node, node, fanout_cap),
                    )
                    for other, weight in cursor.fetchall():
                        if not other or other in seen:
                            continue
                        seen.add(other)
                        score = parent_score * float(weight or 1.0) * (0.85 ** depth)
                        out.append((other, depth, score))
                        next_frontier.append((other, score))
                if not next_frontier:
                    break
                frontier = next_frontier
        out.sort(key=lambda t: (-t[2], t[1]))
        return out

    def list_loci(self, owner_id: Optional[str] = None) -> List[Dict]:
        """List all registered spatial Loci."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if owner_id is None:
                cursor.execute("SELECT * FROM loci ORDER BY name ASC")
            else:
                cursor.execute(
                    "SELECT * FROM loci WHERE COALESCE(owner_id, '') = COALESCE(?, '') ORDER BY name ASC",
                    (owner_id,),
                )
            return [dict(r) for r in cursor.fetchall()]

    def list_temporal_facts_by_locus(
        self,
        locus_id: str,
        owner_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Return the fact timeline for a specific Locus."""
        safe_limit = max(1, int(limit))
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if owner_id is None:
                cursor.execute(
                    '''
                    SELECT subject, predicate, object, valid_from, valid_to, created_at, run_id
                    FROM facts
                    WHERE locus_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    ''',
                    (locus_id, safe_limit),
                )
            else:
                cursor.execute(
                    '''
                    SELECT subject, predicate, object, valid_from, valid_to, created_at, run_id
                    FROM facts
                    WHERE locus_id = ? AND COALESCE(owner_id, '') = COALESCE(?, '')
                    ORDER BY id DESC
                    LIMIT ?
                    ''',
                    (locus_id, owner_id, safe_limit),
                )
            rows = cursor.fetchall()
        
        return [
            {
                "subject": row["subject"],
                "predicate": row["predicate"],
                "object": row["object"],
                "valid_from": row["valid_from"],
                "valid_to": row["valid_to"],
                "created_at": row["created_at"],
                "run_id": row["run_id"],
                "status": "current" if row["valid_to"] is None else "superseded",
            }
            for row in rows
        ]
