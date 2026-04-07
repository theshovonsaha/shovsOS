import asyncio
import hashlib
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
import httpx
import chromadb
from typing import List, Optional
import shutil

OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
DB_PATH     = "./chroma_db"
OPENAI_COMPAT_PROVIDERS = {"openai", "local_openai", "lmstudio", "llamacpp"}

class VectorEngine:
    _http_clients: dict[tuple[str, float, int | None], httpx.AsyncClient] = {}
    _chroma_clients: dict[str, chromadb.PersistentClient] = {}
    _embedding_cache: "OrderedDict[str, List[float]]" = OrderedDict()
    _cache_lock = threading.RLock()
    _client_lock = threading.RLock()

    def __init__(
        self,
        session_id: str,
        agent_id: str = "default",
        model: str = "nomic-embed-text",
        owner_id: Optional[str] = None,
    ):
        from config.config import cfg
        self.session_id = session_id
        self.agent_id   = agent_id
        self.owner_id   = owner_id
        self.model      = model
        self.base_url   = cfg.OLLAMA_BASE_URL.rstrip("/") if cfg.OLLAMA_BASE_URL else OLLAMA_BASE
        self.embedding_timeout = float(getattr(cfg, "EMBEDDING_HTTP_TIMEOUT", 20.0))
        self.embedding_retries = max(0, int(getattr(cfg, "EMBEDDING_HTTP_RETRIES", 2)))
        self.embedding_cache_size = max(64, int(getattr(cfg, "EMBEDDING_CACHE_SIZE", 512)))
        chroma_path = getattr(cfg, "CHROMA_DB_PATH", DB_PATH) or DB_PATH
        self.client = self._get_chroma_client(chroma_path)
        self._ensure_collection()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").strip().split())

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

    @classmethod
    def _get_chroma_client(cls, chroma_path: str) -> chromadb.PersistentClient:
        key = str(chroma_path)
        with cls._client_lock:
            client = cls._chroma_clients.get(key)
            if client is None:
                client = chromadb.PersistentClient(path=key)
                cls._chroma_clients[key] = client
            return client

    @classmethod
    def reset_storage(cls, chroma_path: str):
        with cls._client_lock:
            cls._chroma_clients.clear()
        path = Path(chroma_path)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)

    def _ensure_collection(self):
        # Isolation: owner_{owner_id}_agent_{agent_id}_session_{session_id}
        safe_owner = (self.owner_id or "global").replace("-", "_")
        safe_agent = self.agent_id.replace("-", "_")
        safe_sid   = self.session_id.replace("-", "_")
        self.collection = self.client.get_or_create_collection(
            name=f"owner_{safe_owner}_agent_{safe_agent}_session_{safe_sid}"
        )

    def _resolve_embedding_transport(self, normalized: str) -> tuple[str, str, str, dict, dict]:
        from config.config import cfg

        model_name = str(self.model or "").strip()
        provider = ""
        clean_model = model_name
        if ":" in model_name:
            provider, clean_model = model_name.split(":", 1)
            provider = provider.strip().lower()
        elif "/" in model_name:
            maybe_provider, maybe_model = model_name.split("/", 1)
            if maybe_provider.strip().lower() in OPENAI_COMPAT_PROVIDERS | {"groq", "gemini", "anthropic", "nvidia", "ollama"}:
                provider = maybe_provider.strip().lower()
                clean_model = maybe_model

        if not provider:
            llm_provider = str(getattr(cfg, "LLM_PROVIDER", "") or "").strip().lower()
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

        if provider in OPENAI_COMPAT_PROVIDERS:
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
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"model": clean_model, "input": normalized}
            return provider, base_url.rstrip("/"), "/v1/embeddings", headers, payload

        headers = {"Content-Type": "application/json"}
        payload = {"model": clean_model.replace("ollama:", ""), "input": normalized}
        base_url = (getattr(cfg, "OLLAMA_BASE_URL", "") or OLLAMA_BASE).rstrip("/")
        return "ollama", base_url, "/api/embed", headers, payload

    async def _get_embedding(self, text: str) -> List[float]:
        normalized = self._normalize_text(text)
        cache_key = f"{self.model}::{normalized}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        provider, base_url, endpoint, headers, payload = self._resolve_embedding_transport(normalized)
        client = self._get_http_client(base_url, self.embedding_timeout)

        last_error: Optional[Exception] = None

        for attempt in range(self.embedding_retries + 1):
            try:
                resp = await client.post(endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                
                if provider in OPENAI_COMPAT_PROVIDERS:
                    embedding = data.get("data", [{}])[0].get("embedding")
                elif endpoint == "/api/embed":
                    embeddings = data.get("embeddings")
                    embedding = embeddings[0] if isinstance(embeddings, list) and embeddings else None
                else:
                    embedding = data.get("embedding")
                    
                if not isinstance(embedding, list):
                    raise ValueError(f"Embedding missing from response: {data}")
                    
                self._cache_set(cache_key, embedding, self.embedding_cache_size)
                return embedding
            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code if e.response is not None else 0
                if provider == "ollama" and endpoint == "/api/embed" and status == 404:
                    endpoint = "/api/embeddings"
                    payload = {"model": payload.get("model"), "prompt": normalized}
                    continue
                # Retry only transient classes and throttles.
                if status not in (429, 500, 502, 503, 504):
                    break
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.PoolTimeout, httpx.RemoteProtocolError) as e:
                last_error = e
            except Exception as e:
                last_error = e
                break

            if attempt < self.embedding_retries:
                await asyncio.sleep(0.2 * (2 ** attempt))

        raise RuntimeError(f"Failed to generate embedding after retries: {last_error}")

    def _generate_id(self, key: str, anchor: str) -> str:
        return hashlib.sha256(f"{key}\n{anchor}".encode()).hexdigest()

    async def index(self, key: str, anchor: str, metadata: Optional[dict] = None):
        doc_id = self._generate_id(key, anchor)
        embedding = await self._get_embedding(key)
        meta = metadata or {}
        meta["key"] = key
        meta["anchor"] = anchor
        meta.setdefault("memory_class", "recall_anchor")
        meta.setdefault("source_type", "compressed_exchange")
        meta.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[meta],
            documents=[anchor]
        )
        try:
            from memory.bm25_engine import BM25Engine
            BM25Engine(session_id=self.session_id, agent_id=self.agent_id, owner_id=self.owner_id).index(
                doc_id=doc_id,
                key=key,
                content=anchor,
            )
        except Exception:
            pass

    async def query(self, text: str, limit: int = 3) -> List[dict]:
        embedding = await self._get_embedding(text)
        results = self.collection.query(query_embeddings=[embedding], n_results=limit)
        parsed = []
        if not results or not results["ids"]: return parsed
        for i in range(len(results["ids"][0])):
            parsed.append({
                "id": results["ids"][0][i],
                "key": results["metadatas"][0][i].get("key"),
                "anchor": results["metadatas"][0][i].get("anchor"),
                "metadata": results["metadatas"][0][i]
            })
        try:
            from memory.reranker import rerank
            parsed = rerank(text, parsed, text_key="anchor", top_n=limit)
        except Exception:
            pass
        return parsed

    async def count(self) -> int:
        try:
            return self.collection.count()
        except:
            self._ensure_collection()
            return self.collection.count()

    async def clear(self):
        try:
            self.client.delete_collection(self.collection.name)
        except:
            pass
        self._ensure_collection()
