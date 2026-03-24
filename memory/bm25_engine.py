from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DB_DIR = Path("./chroma_db")
DB_DIR.mkdir(parents=True, exist_ok=True)

_STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for", "of", "and",
    "or", "but", "not", "with", "by", "from", "as", "this", "that", "was",
    "be", "are", "were", "been", "have", "has", "had", "do", "does", "did",
    "will", "would", "can", "could", "should", "may", "might", "shall", "must",
    "am",
}


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return [token for token in cleaned.split() if len(token) > 2 and token not in _STOPWORDS]


class BM25Engine:
    K1 = 1.5
    B = 0.75

    def __init__(self, session_id: str, agent_id: str = "default", owner_id: str | None = None):
        safe_owner = (owner_id or "global").replace("-", "_")
        safe_name = f"bm25_{safe_owner}_{agent_id}_{session_id}".replace("-", "_")[:96]
        self.db_path = DB_DIR / f"{safe_name}.db"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    key TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS inverted_index (
                    term TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    frequency INTEGER NOT NULL,
                    PRIMARY KEY (term, doc_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_inverted_index_term ON inverted_index(term)")
            conn.commit()

    def index(self, doc_id: str, key: str, content: str) -> None:
        tokens = _tokenize(f"{key} {content}")
        if not tokens:
            return

        counts = Counter(tokens)
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (id, content, key, token_count, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (doc_id, (content or "")[:4000], key or "", len(tokens), now),
            )
            for term, freq in counts.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO inverted_index (term, doc_id, frequency)
                    VALUES (?, ?, ?)
                    """,
                    (term, doc_id, freq),
                )
            conn.commit()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        with sqlite3.connect(self.db_path) as conn:
            total_docs, avg_len = conn.execute(
                "SELECT COUNT(*), AVG(token_count) FROM documents"
            ).fetchone()
            total_docs = total_docs or 0
            avg_len = avg_len or 1.0
            if total_docs == 0:
                return []

            placeholders = ",".join("?" for _ in query_tokens)
            candidate_rows = conn.execute(
                f"SELECT DISTINCT doc_id FROM inverted_index WHERE term IN ({placeholders})",
                query_tokens,
            ).fetchall()
            candidate_ids = [row[0] for row in candidate_rows]
            if not candidate_ids:
                return []

            id_placeholders = ",".join("?" for _ in candidate_ids)
            doc_lengths = dict(
                conn.execute(
                    f"SELECT id, token_count FROM documents WHERE id IN ({id_placeholders})",
                    candidate_ids,
                ).fetchall()
            )
            doc_info = {
                row[0]: {"key": row[1], "content": row[2]}
                for row in conn.execute(
                    f"SELECT id, key, content FROM documents WHERE id IN ({id_placeholders})",
                    candidate_ids,
                ).fetchall()
            }

            scores = {doc_id: 0.0 for doc_id in candidate_ids}
            for term in query_tokens:
                df_row = conn.execute(
                    "SELECT COUNT(DISTINCT doc_id) FROM inverted_index WHERE term = ?",
                    (term,),
                ).fetchone()
                df = df_row[0] if df_row else 0
                if df == 0:
                    continue

                idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)
                for doc_id in candidate_ids:
                    freq_row = conn.execute(
                        "SELECT frequency FROM inverted_index WHERE term = ? AND doc_id = ?",
                        (term, doc_id),
                    ).fetchone()
                    tf = freq_row[0] if freq_row else 0
                    if tf == 0:
                        continue
                    doc_len = doc_lengths.get(doc_id, avg_len)
                    tf_norm = tf * (self.K1 + 1) / (
                        tf + self.K1 * (1 - self.B + self.B * doc_len / avg_len)
                    )
                    scores[doc_id] += idf * tf_norm

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        results: list[dict] = []
        for doc_id, score in ranked:
            if score <= 0:
                continue
            info = doc_info.get(doc_id, {})
            results.append(
                {
                    "id": doc_id,
                    "key": info.get("key", ""),
                    "anchor": info.get("content", ""),
                    "score": round(score, 4),
                    "source": "bm25",
                    "metadata": {"bm25_score": score, "key": info.get("key", "")},
                }
            )
        return results

    def clear(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM documents")
            conn.execute("DELETE FROM inverted_index")
            conn.commit()


def reciprocal_rank_fusion(*result_lists: list[dict], k: int = 60, top_n: int = 5) -> list[dict]:
    scores: dict[str, float] = {}
    merged: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list):
            doc_id = result.get("id") or result.get("key") or str(rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
            if doc_id not in merged:
                merged[doc_id] = dict(result)
                merged[doc_id]["id"] = doc_id
                merged[doc_id]["metadata"] = dict(result.get("metadata", {}))
            else:
                merged[doc_id]["metadata"].update(result.get("metadata", {}))

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_n]
    fused: list[dict] = []
    for doc_id in ranked_ids:
        item = dict(merged[doc_id])
        item["metadata"] = dict(item.get("metadata", {}))
        item["metadata"]["rrf_score"] = round(scores[doc_id], 6)
        fused.append(item)
    return fused
