import json
import re
from typing import List, Dict, Any, Optional
from llm.base_adapter import BaseLLMAdapter
from config.logger import log

PLANNING_PROMPT = """\
You are the [Shovs Orchestrator]. Choose the smallest reliable next-step plan for the SAME run.

Available Tools:
{tools_docs}

Session Signals:
- session_has_history: {session_has_history}
- current_fact_count: {current_fact_count}
- recently_failed_tools: {failed_tools}

Phase Context:
{compiled_context}

Rules:
- Preserve exact user entities, domains, URLs, file names, tickers, and keywords unless the context clearly justifies normalization.
- Prefer the minimum tool set needed for the next useful step, not every possible step.
- Do not broaden or weaken the user's query with filler like "related to", "about", or "similar to" unless that expansion is clearly necessary.
- For exact-domain product or company research, prefer first-party evidence before third-party commentary.
- If the user asks what a product costs or what plans exist, prefer fetching the exact pricing page before concluding.
- If the user asks whether a product is good, trustworthy, or recommendable, gather first-party feature/pricing evidence before finalizing.
- Conversational queries ("hi", "how are you", opinions) → return []
- Factual/current data → ["web_search"]
- URL reading → ["web_fetch"]
- File creation/editing → ["file_create"] or ["file_str_replace"]
- Image lookup → ["image_search"]
- Weather → ["weather_fetch"]
- Memory recall → ["query_memory"]
- Delegation → ["delegate_to_agent"]
- Use `todo_write` only when multi-step task tracking will materially help several later steps. Do not use task tools as a substitute for substantive work.
- For direct commands like "web search X" or "research X", prefer the substantive research tool first. Add task tools only when they clearly help the run stay organized.
- Do not include tools that already failed unless a changed query/source makes retrying reasonable.

Return ONLY JSON (no markdown). Preferred format:
{{
  "strategy": "one-line plan",
  "tools": [
    {{"name": "tool_name", "priority": "high|medium|low", "reason": "short reason", "target_argument_clue": "specific clue for the executor to use (e.g. exact URL or search terms)"}}
  ],
  "force_memory": true/false,
  "memory_topic": "topic or empty",
  "confidence": 0.0-1.0
}}

Legacy fallback format allowed: ["web_search", "web_fetch"]

User Query: "{query}"
"""

OBSERVATION_PROMPT = """\
You are the [Shovs Loop Manager]. Review the latest tool results and decide the next step for the SAME run.

User Objective:
{query}

Available Tools:
{tools_docs}

Recent Tool Results:
{tool_results}

Phase Context:
{compiled_context}

Rules:
- Prefer staying inside one run. Do not create a swarm.
- If the gathered evidence is enough to answer, choose "finalize".
- If one more concrete tool step is clearly needed, choose "continue" and name the tools.
- If tools failed and no better tool exists, choose "finalize" so the actor explains what is missing.
- Do not repeat the same failed tool unless the reason is explicit.
- After substantive evidence exists, avoid admin tools unless task state genuinely changed. Do not ask for `todo_write` again.
- Preserve exact user entities/domains in follow-up tools. Do not silently rewrite or broaden the target.
- If search results are noisy or off-topic, only continue if you can name a sharper next query/source.
- For exact-domain product research, prefer first-party pages like pricing/privacy/FAQ before relying on noisy review searches.
- If pricing, plans, trust, or recommendation are requested and that evidence is still missing, continue and fetch the first-party page that closes the gap.
- `notes` must be short, actor-facing execution guidance. Do not include protocol chatter or mention hidden packets.

Return ONLY JSON:
{{
  "status": "continue|finalize",
  "strategy": "one-line controller decision",
  "tools": [
    {{"name": "tool_name", "priority": "high|medium|low", "reason": "short reason"}}
  ],
  "notes": "short instruction to the actor",
  "confidence": 0.0-1.0
}}
"""

VERIFICATION_PROMPT = """\
You are the [Shovs Verification Layer]. Decide if the final answer is supported by the current turn evidence.

User Objective:
{query}

Final Answer:
{response}

Evidence From This Turn:
{tool_results}

Phase Context:
{compiled_context}

Rules:
- Mark supported=false if the answer introduces a concrete factual claim not grounded in the evidence above or the user request.
- Mark supported=true for synthesis or conversational framing that stays within the evidence.
- Be strict about ticker/company swaps, invented files, invented actions, and invented completed work.
- Mark supported=false if the answer echoes internal execution chatter, hidden packet text, or unexecuted tool intentions.
- Be strict about mutated domains, renamed products, and widened search targets that are not supported by evidence.

Return ONLY JSON:
{{
  "supported": true/false,
  "issues": ["short issue", "short issue"],
  "confidence": 0.0-1.0
}}
"""

MEMORY_SIGNAL = re.compile(
    r"\b(remember|recall|earlier|before|last time|you said|i told|i said|"
    r"my name|preferred|preference|always|usually|i like|i hate|"
    r"what did i|did i tell|as i mentioned|we discussed|previously|"
    r"my .{1,25} is|you know that|don't forget)\b",
    re.IGNORECASE,
)
TRIVIAL_SIGNAL = re.compile(
    r"^(?:(?:hi|hello|hey|yo)(?:\s+\w{1,20}){0,2}|ok|okay|thanks|thank you|thx|cool|sure)[!. ]*$",
    re.IGNORECASE,
)
URL_SIGNAL = re.compile(r"https?://", re.IGNORECASE)
MULTISTEP_SIGNAL = re.compile(
    r"\b(then|after that|afterwards|step by step|plan|research|summarize|save|write|create|build|compare|analyze)\b",
    re.IGNORECASE,
)
CURRENT_INFO_SIGNAL = re.compile(
    r"\b(current|latest|news|today|recent|price|who is|what is|when is|where is|which)\b",
    re.IGNORECASE,
)
DOMAIN_SIGNAL = re.compile(r"\b[a-z0-9][a-z0-9.-]*\.[a-z]{2,}\b", re.IGNORECASE)
PRICING_SIGNAL = re.compile(r"\b(price|pricing|cost|costs|plan|plans|tier|tiers)\b", re.IGNORECASE)
TRUST_SIGNAL = re.compile(r"\b(trust|trustworthy|legit|legitimate|safe|secure|privacy|security)\b", re.IGNORECASE)
COMPARISON_SIGNAL = re.compile(r"\b(compare|comparison|competitor|competitors|alternative|alternatives|vs)\b", re.IGNORECASE)
PRODUCT_RESEARCH_SIGNAL = re.compile(r"\b(research|investigate|evaluate|assess|recommend|good|worth it)\b", re.IGNORECASE)


def _extract_exact_domains(text: str) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for match in DOMAIN_SIGNAL.findall(text or ""):
        normalized = match.lower().strip()
        if normalized and normalized not in seen:
            ordered.append(normalized)
            seen.add(normalized)
    return ordered


def _tool_result_mentions(item: Dict[str, Any], needle: str) -> bool:
    haystack = (
        str(item.get("content") or "")
        + " "
        + json.dumps(item.get("arguments") or {}, ensure_ascii=False)
    ).lower()
    return needle.lower() in haystack


def _infer_observation_gap_actions(
    *,
    query: str,
    tool_results: List[Dict[str, Any]],
    known_tools: set[str],
    failed_tools: set[str],
) -> Optional[Dict[str, Any]]:
    domains = _extract_exact_domains(query)
    if not domains:
        return None

    domain = domains[0]
    homepage_fetch = any(
        bool(item.get("success"))
        and str(item.get("tool_name") or item.get("tool") or "") == "web_fetch"
        and _tool_result_mentions(item, domain)
        for item in tool_results or []
    )
    if not homepage_fetch:
        return None

    lower_query = (query or "").lower()

    if (
        PRICING_SIGNAL.search(lower_query)
        and "web_fetch" in known_tools
        and "web_fetch" not in failed_tools
        and not any(_tool_result_mentions(item, f"{domain}/pricing") for item in tool_results or [])
    ):
        return {
            "status": "continue",
            "strategy": "Fetch first-party pricing evidence before concluding on cost or recommendation.",
            "tools": [{"name": "web_fetch", "priority": "high", "reason": "Missing exact pricing evidence."}],
            "notes": f"Fetch the exact pricing page for {domain} before concluding pricing, value, or recommendation.",
            "confidence": 0.9,
        }

    if (
        COMPARISON_SIGNAL.search(lower_query)
        and "web_search" in known_tools
        and "web_search" not in failed_tools
        and not any(
            str(item.get("tool_name") or item.get("tool") or "") == "web_search"
            and any(term in str(item.get("content") or "").lower() for term in ["alternative", "alternatives", "competitor", "competitors", "vs "])
            for item in tool_results or []
        )
    ):
        return {
            "status": "continue",
            "strategy": "Gather competitor evidence before final comparison or recommendation.",
            "tools": [{"name": "web_search", "priority": "high", "reason": "Missing competitor/alternative evidence."}],
            "notes": f"Search for exact competitors or alternatives to {domain} without broadening the product identity.",
            "confidence": 0.82,
        }

    if (
        TRUST_SIGNAL.search(lower_query)
        and PRODUCT_RESEARCH_SIGNAL.search(lower_query)
        and "web_fetch" in known_tools
        and "web_fetch" not in failed_tools
        and not any(
            _tool_result_mentions(item, f"{domain}/privacy")
            or _tool_result_mentions(item, "privacy policy")
            or _tool_result_mentions(item, "terms of service")
            for item in tool_results or []
        )
    ):
        return {
            "status": "continue",
            "strategy": "Fetch a first-party trust or privacy page before final trust assessment.",
            "tools": [{"name": "web_fetch", "priority": "medium", "reason": "Trust/privacy evidence is still weak."}],
            "notes": f"Fetch a first-party privacy or trust-related page for {domain} before making a strong trust recommendation.",
            "confidence": 0.76,
        }

    return None

class AgenticOrchestrator:
    def __init__(self, adapter: BaseLLMAdapter):
        self.adapter = adapter

    def set_adapter(self, adapter: BaseLLMAdapter):
        """Hot-swap the underlying adapter when user switches providers."""
        self.adapter = adapter

    def classify_route(self, query: str, session_has_history: bool = False, current_fact_count: int = 0) -> str:
        q = (query or "").strip()
        if not q:
            return "trivial_chat"
        if TRIVIAL_SIGNAL.fullmatch(q):
            return "trivial_chat"
        if URL_SIGNAL.search(q):
            return "url_fetch"
        if MEMORY_SIGNAL.search(q) and (session_has_history or current_fact_count > 0):
            return "memory_recall"
        if MULTISTEP_SIGNAL.search(q):
            return "multi_step"
        if CURRENT_INFO_SIGNAL.search(q):
            return "direct_fact"
        return "open_ended"

    async def plan_with_context(
        self,
        query: str,
        tools_list: List[Dict],
        model: str = "llama3.1:8b",
        session_has_history: bool = False,
        current_fact_count: int = 0,
        failed_tools: Optional[List[str]] = None,
        compiled_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze query and return structured execution guidance."""
        tools_docs = "\n".join([f"- {t['name']}: {t['description']}" for t in tools_list])
        known_tools = {t["name"] for t in tools_list if isinstance(t, dict) and t.get("name")}
        failed_set = set(failed_tools or [])
        route_type = self.classify_route(
            query,
            session_has_history=session_has_history,
            current_fact_count=current_fact_count,
        )

        deterministic_tools: list[dict[str, str]] = []
        if route_type == "trivial_chat":
            return {
                "strategy": "No tools needed for a trivial conversational turn.",
                "tools": [],
                "force_memory": False,
                "memory_topic": "",
                "confidence": 0.98,
                "route_type": route_type,
                "should_plan": False,
            }
        if route_type == "url_fetch" and "web_fetch" in known_tools and "web_fetch" not in failed_set:
            deterministic_tools.append({"name": "web_fetch", "priority": "high", "reason": "Direct URL detected."})
        elif route_type == "memory_recall" and "query_memory" in known_tools and "query_memory" not in failed_set:
            deterministic_tools.append({"name": "query_memory", "priority": "high", "reason": "Memory recall intent detected."})
        elif route_type == "direct_fact" and "web_search" in known_tools and "web_search" not in failed_set:
            deterministic_tools.append({"name": "web_search", "priority": "high", "reason": "Deterministic factual/current query route."})

        if deterministic_tools and route_type in {"url_fetch", "direct_fact"}:
            return {
                "strategy": "Use deterministic route-selected tools before final answer.",
                "tools": deterministic_tools,
                "force_memory": route_type == "memory_recall",
                "memory_topic": query[:80],
                "confidence": 0.9,
                "route_type": route_type,
                "should_plan": False,
            }

        prompt = PLANNING_PROMPT.format(
            tools_docs=tools_docs,
            query=query,
            session_has_history=str(bool(session_has_history)).lower(),
            current_fact_count=current_fact_count,
            failed_tools=", ".join(sorted(failed_set)) if failed_set else "none",
            compiled_context=(compiled_context.strip() if compiled_context else "none"),
        )
        
        from llm.adapter_factory import create_adapter, strip_provider_prefix
        current_adapter = create_adapter(provider=model) if ":" in model else self.adapter
        clean_model = strip_provider_prefix(model)
        
        try:
            response = await current_adapter.complete(
                model=clean_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            def _normalize_tools(entries: List[Any]) -> List[Dict[str, str]]:
                normalized = []
                for entry in entries:
                    if isinstance(entry, str):
                        name = entry
                        priority = "medium"
                        reason = "Planner-selected tool"
                        clue = ""
                    elif isinstance(entry, dict):
                        name = entry.get("name")
                        priority = str(entry.get("priority", "medium")).lower()
                        reason = str(entry.get("reason", "Planner-selected tool"))
                        clue = str(entry.get("target_argument_clue", "")).strip()
                    else:
                        continue
                    if not isinstance(name, str) or name not in known_tools or name in failed_set:
                        continue
                    if priority not in {"high", "medium", "low"}:
                        log("orch", "plan", f"Invalid planner priority '{priority}' for tool '{name}'. Using medium.", level="warn")
                        priority = "medium"
                    normalized.append({"name": name, "priority": priority, "reason": reason, "target_argument_clue": clue})
                priority_rank = {"high": 0, "medium": 1, "low": 2}
                normalized.sort(key=lambda item: (priority_rank[item["priority"]], item["name"]))
                return normalized

            payload: Dict[str, Any] = {}
            obj_match = re.search(r'\{.*\}', response, re.DOTALL)
            if obj_match:
                maybe_obj = json.loads(obj_match.group(0))
                if isinstance(maybe_obj, dict):
                    payload = maybe_obj
            else:
                arr_match = re.search(r'\[.*\]', response, re.DOTALL)
                if arr_match:
                    maybe_list = json.loads(arr_match.group(0))
                    if isinstance(maybe_list, list):
                        payload = {"tools": maybe_list}

            tools = _normalize_tools(payload.get("tools", []))

            query_norm = re.sub(r"[^a-z0-9\s]", " ", query.lower())
            query_norm = re.sub(r"\s+", " ", query_norm).strip()

            greeting_set = {
                "hi",
                "hello",
                "hey",
                "yo",
                "how are you",
                "how are you today",
                "hello how are you",
                "hello how are you today",
            }
            if query_norm in greeting_set:
                tools = []

            if (
                re.search(r"\b(weather|temperature|forecast|rain|snow|wind)\b", query_norm)
                and "weather_fetch" in known_tools
                and "weather_fetch" not in failed_set
                and not any(t["name"] == "weather_fetch" for t in tools)
            ):
                tools.insert(0, {
                    "name": "weather_fetch",
                    "priority": "high",
                    "reason": "Deterministic routing for weather queries.",
                })

            if (
                not tools
                and "web_search" in known_tools
                and "web_search" not in failed_set
                and re.search(r"\b(current|latest|news|price|best|top|what|who|when|where|which)\b", query_norm)
                and query_norm not in greeting_set
            ):
                tools.append({
                    "name": "web_search",
                    "priority": "medium",
                    "reason": "Deterministic fallback for factual queries.",
                })

            memory_warranted = (
                (session_has_history or current_fact_count > 0)
                and bool(MEMORY_SIGNAL.search(query))
            )

            if memory_warranted and not any(t["name"] == "query_memory" for t in tools):
                if "query_memory" in known_tools and "query_memory" not in failed_set:
                    tools.insert(0, {
                        "name": "query_memory",
                        "priority": "high",
                        "reason": "Query signals memory dependency.",
                    })

            structured = {
                "strategy": str(payload.get("strategy", "Use selected tools to gather evidence before final answer.")),
                "tools": tools,
                "force_memory": bool(payload.get("force_memory", session_has_history or current_fact_count > 0)),
                "memory_topic": str(payload.get("memory_topic", query[:80])).strip(),
                "confidence": float(payload.get("confidence", 0.5)),
                "route_type": route_type,
                "should_plan": True,
            }
            log("orch", "plan", f"Orchestrator strategy: {[t['name'] for t in tools]}")
            return structured
        except Exception as e:
            log("orch", "plan", f"Orchestrator failed: {e}", level="error")
            return {
                "strategy": "Planner failed; continue with direct reasoning.",
                "tools": [],
                "force_memory": False,
                "memory_topic": "",
                "confidence": 0.0,
                "route_type": route_type,
                "should_plan": False,
            }

    async def plan(self, query: str, tools_list: List[Dict], model: str = "llama3.1:8b") -> List[str]:
        """
        Backward-compatible planning API: returns tool names only.
        """
        structured = await self.plan_with_context(query=query, tools_list=tools_list, model=model)
        return [t["name"] for t in structured.get("tools", []) if isinstance(t, dict) and isinstance(t.get("name"), str)]

    async def observe_with_context(
        self,
        query: str,
        tools_list: List[Dict],
        tool_results: List[Dict[str, Any]],
        model: str = "llama3.1:8b",
        compiled_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        tools_docs = "\n".join([f"- {t['name']}: {t['description']}" for t in tools_list])
        known_tools = {t["name"] for t in tools_list if isinstance(t, dict) and t.get("name")}

        formatted_results = []
        failed_tools = set()
        successful_count = 0
        for item in tool_results or []:
            name = str(item.get("tool_name") or item.get("tool") or "unknown")
            success = bool(item.get("success"))
            preview = str(item.get("content") or "").strip()
            if len(preview) > 300:
                preview = preview[:297].rstrip() + "..."
            formatted_results.append(
                f"- {name} | success={str(success).lower()} | preview={preview or '[empty]'}"
            )
            if success:
                successful_count += 1
            else:
                failed_tools.add(name)

        if not tool_results:
            return {
                "status": "finalize",
                "strategy": "No tool results to review; continue with direct reasoning.",
                "tools": [],
                "notes": "",
                "confidence": 0.2,
            }

        prompt = OBSERVATION_PROMPT.format(
            query=query,
            tools_docs=tools_docs,
            tool_results="\n".join(formatted_results) or "none",
            compiled_context=(compiled_context.strip() if compiled_context else "none"),
        )

        from llm.adapter_factory import create_adapter, strip_provider_prefix
        current_adapter = create_adapter(provider=model) if ":" in model else self.adapter
        clean_model = strip_provider_prefix(model)

        try:
            response = await current_adapter.complete(
                model=clean_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            obj_match = re.search(r'\{.*\}', response, re.DOTALL)
            payload = json.loads(obj_match.group(0)) if obj_match else {}
        except Exception as e:
            log("orch", "observe", f"Observation failed: {e}", level="warn")
            payload = {}

        tools: list[dict[str, str]] = []
        for entry in payload.get("tools", []) if isinstance(payload, dict) else []:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            priority = str(entry.get("priority", "medium")).lower()
            reason = str(entry.get("reason", "Manager-selected tool"))
            if not isinstance(name, str) or name not in known_tools or name in failed_tools:
                continue
            if priority not in {"high", "medium", "low"}:
                priority = "medium"
            tools.append({"name": name, "priority": priority, "reason": reason})

        status = str(payload.get("status", "")).strip().lower()
        if status not in {"continue", "finalize"}:
            status = "finalize" if successful_count > 0 else "continue"

        if status == "continue" and not tools and successful_count > 0:
            status = "finalize"

        if status == "continue" and not tools:
            for candidate in ("web_search", "web_fetch", "query_memory"):
                if candidate in known_tools and candidate not in failed_tools:
                    tools.append({
                        "name": candidate,
                        "priority": "medium",
                        "reason": "Fallback continuation after incomplete observation.",
                    })
                    break

        heuristic = _infer_observation_gap_actions(
            query=query,
            tool_results=tool_results,
            known_tools=known_tools,
            failed_tools=failed_tools,
        )
        if heuristic:
            return heuristic

        return {
            "status": status,
            "strategy": str(payload.get("strategy", "Manager reviewed the latest step.")),
            "tools": tools,
            "notes": str(payload.get("notes", "")).strip(),
            "confidence": float(payload.get("confidence", 0.5)),
        }

    async def verify_with_context(
        self,
        query: str,
        response: str,
        tool_results: List[Dict[str, Any]],
        model: str = "llama3.1:8b",
        compiled_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        formatted_results = []
        for item in tool_results or []:
            name = str(item.get("tool_name") or item.get("tool") or "unknown")
            success = bool(item.get("success"))
            preview = str(item.get("content") or "").strip()
            if len(preview) > 300:
                preview = preview[:297].rstrip() + "..."
            formatted_results.append(
                f"- {name} | success={str(success).lower()} | preview={preview or '[empty]'}"
            )

        if not formatted_results:
            return {"supported": True, "issues": [], "confidence": 0.3}

        prompt = VERIFICATION_PROMPT.format(
            query=query,
            response=response,
            tool_results="\n".join(formatted_results),
            compiled_context=(compiled_context.strip() if compiled_context else "none"),
        )

        from llm.adapter_factory import create_adapter, strip_provider_prefix
        current_adapter = create_adapter(provider=model) if ":" in model else self.adapter
        clean_model = strip_provider_prefix(model)

        try:
            raw = await current_adapter.complete(
                model=clean_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            obj_match = re.search(r'\{.*\}', raw, re.DOTALL)
            payload = json.loads(obj_match.group(0)) if obj_match else {}
        except Exception as e:
            log("orch", "verify", f"Verification failed: {e}", level="warn")
            payload = {}

        supported = payload.get("supported")
        if not isinstance(supported, bool):
            supported = True
        issues = payload.get("issues", [])
        if not isinstance(issues, list):
            issues = []
        confidence = payload.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.5
        return {
            "supported": supported,
            "issues": [str(item) for item in issues[:5] if str(item).strip()],
            "confidence": confidence,
        }
