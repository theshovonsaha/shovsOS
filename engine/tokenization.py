"""Tokenizer access used across engine + run_engine.

Lifted out of ``engine.core`` so ``run_engine.engine`` can import it without
pulling the rest of ``engine.core`` (which itself imports ``run_engine`` via
the context governor — the cycle that broke ``test_memory_plumbing``).
"""

from __future__ import annotations

import tiktoken


def get_token_encoding():
    """Resolve a safe tokenizer without raising.

    Falls back to ``o200k_base`` if ``cl100k_base`` is unavailable, then to
    ``None`` so callers can drop to a char-count approximation.
    """
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return None
