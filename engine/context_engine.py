"""
Context Engine v2
-----------------
Manages compressed session memory across long conversations.

Changes vs v1:
- compress_exchange() now accepts model= parameter so the main model is used
  instead of tinyllama. Small models (1b) are unreliable for structured output.
- Compression prompt simplified: plain bullet points, no [KEY:] syntax required.
  The vector indexing now extracts keys separately instead of relying on the
  compressor to format them perfectly.
- Recompression threshold raised: 80 lines → compress to 40. Same logic, cleaner.
- TRIVIAL_PATTERN still skips useless exchanges but never skips first exchange.
"""

import re
from typing import Optional
from llm.base_adapter import BaseLLMAdapter
from llm.adapter_factory import create_adapter, strip_provider_prefix
from engine.context_schema import ContextItem, ContextKind, ContextPhase
from engine.fact_guard import is_grounded_fact_record
from engine.context_hygiene import should_skip_memory_compression
from config.config import cfg


COMPRESSION_PROMPT = """\
You are a memory compression engine for a Language OS runtime.
Extract what must be remembered from this exchange.

EXISTING MEMORY (do not re-extract what's already here):
{existing_context}

NEW EXCHANGE:
User: {user_message}
Assistant: {assistant_response}

Extract ONLY genuinely new information not already in memory.
Ignore:
- hidden execution chatter, tool schemas, and packet labels
- temporary plan boilerplate about what the assistant intends to do
- raw tool JSON and tool-result formatting noise
Include:
- Topics the user asked about
- Names, goals, preferences, constraints, corrections
- Decisions made or tasks agreed on
- If this is the first exchange: note "First message"

CRITICAL: If an exchange establishes a durable fact about the user or system, output it as:
[FACT: <subject> | <predicate> | <object>]

CRITICAL: If a new fact contradicts or updates a prior belief, you MUST void the old one:
[VOIDS: <subject> | <predicate>]

Output: one fact per line starting with "- " or the exact [FACT:/VOIDS:] markers.
If nothing new: output exactly: [nothing new]
No headers. No commentary.\
"""

RECOMPRESSION_PROMPT = """\
Compress this memory list. Keep the most important facts. Target: half the current length.

Never drop:
- The first message note
- All topics discussed
- Active goals or tasks
- User corrections

MEMORY:
{context}

Output: plain bullet points starting with "- ", most important first.\
"""

TRIVIAL_PATTERN = re.compile(
    r"^(ok|okay|thanks|thank you|great|sure|yes|no|got it|understood|"
    r"alright|cool|nice|good|perfect|sounds good|makes sense)[\s.!?]*$",
    re.IGNORECASE,
)

TOOL_RESULT_RE = re.compile(
    r"(?:\[Tool result from [^\]]+\].*?(?=\n\n|\Z))|(<SYSTEM_TOOL_RESULT[^>]*>.*?</SYSTEM_TOOL_RESULT>)",
    re.DOTALL | re.IGNORECASE,
)

MAX_CONTEXT_LINES  = 80
TARGET_LINES_AFTER = 40


class ContextEngine:

    def __init__(
        self,
        adapter: BaseLLMAdapter,
        compression_model: Optional[str] = None,
    ):
        self.adapter           = adapter
        self.compression_model = compression_model or cfg.DEFAULT_MODEL

    def set_adapter(self, adapter: BaseLLMAdapter):
        """Hot-swap the underlying LLM adapter (called when user switches providers)."""
        self.adapter = adapter

    def is_trivial(self, user_message: str, assistant_response: str) -> bool:
        return should_skip_memory_compression(user_message, assistant_response)

    def _clean_response(self, response: str) -> str:
        """Strip tool result blocks before compression — model needs prose only."""
        cleaned = TOOL_RESULT_RE.sub("", response)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if len(cleaned) > 1500:
            omitted = len(cleaned) - 1200
            cleaned = f"{cleaned[:800]}\n\n[...{omitted} chars omitted...]\n\n{cleaned[-400:]}"
        return cleaned

    async def compress_exchange(
        self,
        user_message:      str,
        assistant_response: str,
        current_context:   str,
        is_first_exchange: bool = False,
        model:             str = None,  # caller can pass main model
        grounding_text:    str = "",
    ) -> tuple[str, list[dict], list[dict]]:
        """
        Compress a new exchange into the existing context.
        Returns (updated_context_string, list_of_keyed_facts, list_of_voids).
        """
        # Skip low-value social turns even when the assistant was verbose. They
        # do not carry durable user facts and should not poison context.
        if self.is_trivial(user_message, assistant_response):
            return current_context, [], []

        use_model = model or self.compression_model
        cleaned_response = self._clean_response(assistant_response)

        if ":" in use_model:
            provider_prefix = use_model.split(":", 1)[0].lower()
            current_adapter = create_adapter(provider=provider_prefix)
            clean_model = strip_provider_prefix(use_model)
        else:
            current_adapter = self.adapter
            clean_model = use_model

        prompt = COMPRESSION_PROMPT.format(
            existing_context=current_context or "(empty — first exchange)",
            user_message=user_message.strip(),
            assistant_response=cleaned_response,
        )

        try:
            print(f"[ContextEngine] Compressing with {use_model} (adapter: {current_adapter.__class__.__name__})...")
            compressed = await current_adapter.complete(
                model=clean_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400,
            )
        except Exception as e:
            print(f"[ContextEngine] Compression failed: {e} — skipping")
            return current_context, [], []

        compressed = compressed.strip()

        # Handle "nothing new" response
        if re.search(r"^\[nothing( new)?\]", compressed, re.IGNORECASE | re.MULTILINE):
            if is_first_exchange:
                # Force-store the first message even if model said nothing new
                line = f"- First message: \"{user_message.strip()[:100]}\""
                existing = [l for l in current_context.split("\n") if l.strip()]
                fact = {"key": "Session Start", "fact": f"First message: {user_message.strip()[:100]}", "subject": "Session", "predicate": "Start", "object": "User message"}
                return "\n".join(existing + [line]), [fact], []
            return current_context, [], []

        # Parse bullet points and structural markers
        new_lines = []
        keyed_facts = []
        voids = []

        void_re = re.compile(r"\[\s*VOIDS\s*:(.*?)\]", re.IGNORECASE)
        fact_re = re.compile(r"\[\s*FACT\s*:(.*?)\]", re.IGNORECASE)

        for line in compressed.split("\n"):
            line = line.strip()

            # 1) Parse VOIDS
            void_match = void_re.search(line)
            if void_match:
                parts = [p.strip() for p in void_match.group(1).split("|")]
                if len(parts) >= 2:
                    voids.append({"subject": parts[0], "predicate": parts[1]})
                continue

            # 2) Parse FACT
            fact_match = fact_re.search(line)
            if fact_match:
                parts = [p.strip() for p in fact_match.group(1).split("|")]
                if len(parts) >= 2:
                    subj = parts[0]
                    pred = parts[1]
                    obj = parts[2] if len(parts) > 2 else ""
                    key = f"{subj} {pred}".strip()
                    fact_str = f"{subj} {pred} {obj}".strip()
                    candidate = {
                        "key": key,
                        "fact": fact_str,
                        "subject": subj,
                        "predicate": pred,
                        "object": obj,
                    }
                    grounded, _ = is_grounded_fact_record(
                        candidate,
                        user_message=user_message,
                        grounding_text=grounding_text,
                    )
                    if not grounded:
                        continue
                    keyed_facts.append({
                        "key": key,
                        "fact": fact_str,
                        "subject": subj,
                        "predicate": pred,
                        "object": obj
                    })
                    new_lines.append(f"- {fact_str}")
                continue

            # 3) Plain bullet points
            if not line.startswith("- ") or len(line) < 4:
                continue
            new_lines.append(line)
            # Derive a simple key from the first few words for vector indexing
            fact_text = line[2:].strip()
            words = fact_text.split()[:4]
            key = " ".join(w.capitalize() for w in words) if words else "General"
            candidate = {
                "key": key,
                "fact": fact_text,
                "subject": "General",
                "predicate": "Context",
                "object": fact_text,
            }
            grounded, _ = is_grounded_fact_record(
                candidate,
                user_message=user_message,
                grounding_text=grounding_text,
            )
            if not grounded:
                new_lines.pop()
                continue
            keyed_facts.append({
                "key": key, 
                "fact": fact_text,
                "subject": "General",
                "predicate": "Context",
                "object": fact_text
            })

        if not new_lines:
            if is_first_exchange:
                line = f"- First message: \"{user_message.strip()[:100]}\""
                existing = [l for l in current_context.split("\n") if l.strip()]
                return "\n".join(existing + [line]), [{"key": "Session Start", "fact": line[2:], "subject": "Session", "predicate": "Start", "object": "First message"}], []
            return current_context, [], []

        existing_lines = [l for l in current_context.split("\n") if l.strip()]
        # Deterministic dedup BEFORE the line cap: each turn the compressor may
        # re-extract an already-known fact (e.g. "User name is Vonny"). Appending
        # blindly produced duplicate bullets that bloated context and forced an
        # extra LLM recompression. Deduping here keeps the summary tight and cuts
        # recompression frequency — no model call needed.
        merged = self._dedupe_lines(existing_lines + new_lines)

        if len(merged) > MAX_CONTEXT_LINES:
            print(f"[ContextEngine] Context at {len(merged)} lines — recompressing...")
            merged = await self._recompress("\n".join(merged), use_model)

        return "\n".join(merged), keyed_facts, voids

    @staticmethod
    def _dedupe_lines(lines: list[str]) -> list[str]:
        """Drop duplicate durable bullets, comparing on normalized content so
        "- User name is Vonny" and "User name is Vonny." collapse to one. Keeps
        the first occurrence to preserve the "First message" anchor's position."""
        seen: set[str] = set()
        out: list[str] = []
        for line in lines:
            if not str(line).strip():
                continue
            key = re.sub(r"[^a-z0-9]+", " ", str(line).lower()).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(line)
        return out

    async def _recompress(self, context: str, model: str) -> list[str]:
        if ":" in model:
            provider_prefix = model.split(":", 1)[0].lower()
            current_adapter = create_adapter(provider=provider_prefix)
            clean_model = strip_provider_prefix(model)
        else:
            current_adapter = self.adapter
            clean_model = model

        try:
            result = await current_adapter.complete(
                model=clean_model,
                messages=[{
                    "role": "user",
                    "content": RECOMPRESSION_PROMPT.format(context=context),
                }],
                temperature=0.1,
                max_tokens=500,
            )
            lines = [
                l.strip() for l in result.strip().split("\n")
                if l.strip().startswith("- ") and len(l.strip()) > 3
            ]
            if lines:
                return lines
        except Exception as e:
            print(f"[ContextEngine] Recompression failed: {e}")

        # Hard fallback: keep most recent lines
        all_lines = [l for l in context.split("\n") if l.strip()]
        return all_lines[-TARGET_LINES_AFTER:]

    def build_context_block(self, context: str) -> str:
        lines = [l for l in context.split("\n") if l.strip()] if context else []
        if not lines:
            return ""
        return (
            "--- Session Memory ---\n"
            + "\n".join(lines)
            + "\n--- End Session Memory ---"
        )

    def build_context_items(self, context: str) -> list[ContextItem]:
        block = self.build_context_block(context)
        if not block:
            return []
        return [
            ContextItem(
                item_id="context_engine_v1_memory",
                kind=ContextKind.MEMORY,
                title="",
                content=block,
                source="context_engine_v1",
                priority=60,
                max_chars=1800,
                phase_visibility=frozenset({
                    ContextPhase.PLANNING,
                    ContextPhase.ACTING,
                    ContextPhase.RESPONSE,
                    ContextPhase.VERIFICATION,
                }),
                trace_id="ctx:v1:memory",
                provenance={"engine": "v1"},
                formatted=True,
            )
        ]

    @property
    def model(self) -> str:
        return self.compression_model
