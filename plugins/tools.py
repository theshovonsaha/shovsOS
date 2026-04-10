"""
Built-in Tools
--------------
Real implementations for:
  - web_search       canonical web search from tools_web.py
  - web_fetch        canonical web fetch from tools_web.py
  - image_search     DuckDuckGo images (no API key required)
    - bash             sandboxed Docker command execution
  - file_create      write file inside sandbox dir
  - file_view        read file / list directory
  - file_str_replace in-place string replacement
  - weather_fetch    open-meteo.com (free, no API key)
  - places_search    Google Places API  (needs GOOGLE_PLACES_API_KEY env var)
  - places_map       static HTML map via Google Maps Embed

Registration
------------
  from agent.tools import register_all_tools
  register_all_tools(tool_registry)          # in main.py after tool_registry is created

Environment variables (only places_search / places_map need them):
  GOOGLE_PLACES_API_KEY   — Google Cloud project with Places API enabled
  SANDBOX_DIR             — root for file tools  (default: ./agent_sandbox)
  BASH_TIMEOUT            — seconds before bash kill (default: 30)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import subprocess
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import httpx

from plugins.tool_registry import Tool, ToolRegistry
from plugins.tools_web import (
    WEB_FETCH_TOOL as CANONICAL_WEB_FETCH_TOOL,
    WEB_SEARCH_TOOL as CANONICAL_WEB_SEARCH_TOOL,
    _web_fetch as _canonical_web_fetch,
    _web_search as _canonical_web_search,
)
from config.logger import log
from memory.task_tracker import get_session_task_tracker

if TYPE_CHECKING:
    from orchestration.agent_manager import AgentManager


# ─── Config ───────────────────────────────────────────────────────────────────

GOOGLE_PLACES_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
SANDBOX_DIR       = Path(os.getenv("SANDBOX_DIR", "./agent_sandbox")).resolve()
BASH_TIMEOUT      = int(os.getenv("BASH_TIMEOUT", "30"))
HTTP_TIMEOUT      = 20.0

SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

_agent_manager: Optional[AgentManager] = None  # Injected during registration

WEB_SEARCH_TOOL = CANONICAL_WEB_SEARCH_TOOL
WEB_FETCH_TOOL = CANONICAL_WEB_FETCH_TOOL
_web_search = _canonical_web_search
_web_fetch = _canonical_web_fetch


def _normalize_search_results(results: list[dict], max_results: int = 8) -> list[dict]:
    """Compatibility helper kept for tests and fallback search formatting."""
    normalized: list[dict] = []
    seen_urls: set[str] = set()

    for item in results or []:
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        snippet = re.sub(r"\s+", " ", str(item.get("snippet", "")).strip()).strip()
        source = str(item.get("source", "")).strip()

        if not title and not snippet:
            continue
        if not snippet:
            snippet = title or "Untitled"
        if len(snippet) > 320:
            snippet = snippet[:317].rstrip() + "..."

        url_key = url.lower().rstrip("/")
        if url_key and url_key in seen_urls:
            continue
        if url_key:
            seen_urls.add(url_key)

        entry = {"title": title or "Untitled", "url": url, "snippet": snippet}
        if source:
            entry["source"] = source
        normalized.append(entry)
        if len(normalized) >= max(1, max_results):
            break

    return normalized


def _format_search_results(
    query: str,
    results: list[dict],
    engine: str = "unknown",
    max_results: int = 8,
) -> str:
    cleaned = _normalize_search_results(results, max_results=max_results)
    context_summary = "\n".join(f"- {item['snippet']}" for item in cleaned[:3])
    return json.dumps(
        {
            "type": "web_search_results",
            "query": query,
            "engine": engine,
            "context_summary": context_summary,
            "results": cleaned,
        }
    )


async def _search_duckduckgo(query: str, num_results: int) -> list[dict]:
    """Compatibility fallback used by tools_web when DuckDuckGo is requested directly."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    results: list[dict] = []

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
                headers=headers,
            )
            data = resp.json()

        if data.get("AbstractText"):
            results.append(
                {
                    "title": data.get("Heading", "Answer"),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data["AbstractText"],
                    "source": "duckduckgo",
                }
            )

        for topic in data.get("RelatedTopics", []):
            if len(results) >= num_results:
                break
            if isinstance(topic, dict) and "Text" in topic:
                results.append(
                    {
                        "title": topic.get("Text", "")[:80],
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", ""),
                        "source": "duckduckgo",
                    }
                )
        if results:
            return results[:num_results]
    except Exception:
        pass

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
            resp = await client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers={**headers, "Content-Type": "application/x-www-form-urlencoded"},
            )
            html = resp.text

        block_re = re.compile(
            r'class="result__title".*?href="([^"]+)"[^>]*>(.*?)</a>.*?'
            r'class="result__snippet"[^>]*>(.*?)</span>',
            re.DOTALL,
        )
        for match in block_re.finditer(html):
            if len(results) >= num_results:
                break
            url = match.group(1).strip()
            title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            snippet = re.sub(r"<[^>]+>", "", match.group(3)).strip()
            if url and title:
                results.append(
                    {
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "source": "duckduckgo",
                    }
                )
    except Exception:
        pass

    return results[:num_results]


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE SEARCH  —  DuckDuckGo images (no API key)
# ══════════════════════════════════════════════════════════════════════════════

async def _image_search(query: str, num_results: int = 5) -> str:
    """Search for images via DuckDuckGo and return URLs + titles."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "Chrome/120 Safari/537.36"
        ),
        "Referer": "https://duckduckgo.com/",
    }
    try:
        # Step 1: get vqd token (required by DDG image API)
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
            token_resp = await client.get(
                "https://duckduckgo.com/",
                params={"q": query},
                headers=headers,
            )
            vqd_match = re.search(r'vqd=(["\'])([^"\']+)\1', token_resp.text)
            if not vqd_match:
                return f"image_search: could not retrieve search token for '{query}'"
            vqd = vqd_match.group(2)

            # Step 2: image results
            img_resp = await client.get(
                "https://duckduckgo.com/i.js",
                params={"q": query, "vqd": vqd, "f": ",,,,,", "p": "1"},
                headers=headers,
            )
            data = img_resp.json()

        results = data.get("results", [])[:num_results]
        if not results:
            return f"No images found for: {query}"

        lines = [f"Image results for: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r.get('title', 'Untitled')}")
            lines.append(f"    Image URL:    {r.get('image', '')}")
            lines.append(f"    Source URL:   {r.get('url', '')}")
            lines.append(f"    Dimensions:   {r.get('width', '?')}x{r.get('height', '?')}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"image_search error: {e}"


IMAGE_SEARCH_TOOL = Tool(
    name="image_search",
    description=(
        "Search for images using DuckDuckGo. "
        "Returns image URLs, source pages, and dimensions. "
        "No API key required."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query":       {"type": "string",  "description": "Image search query"},
            "num_results": {"type": "integer", "description": "Number of images (default 5)", "default": 5},
        },
        "required": ["query"],
    },
    handler=_image_search,
    tags=["web", "images"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  BASH  —  Docker sandbox execution
# ══════════════════════════════════════════════════════════════════════════════

# Commands that are blocked outright regardless of context
_BLOCKED_COMMANDS = re.compile(
    r"\b(rm\s+-rf\s+/|mkfs|dd\s+if=|:\(\)\{\s*:\|:&\s*\};:|shutdown|reboot|"
    r"chmod\s+777\s+/|curl\s+.*\|\s*sh|wget\s+.*\|\s*sh)",
    re.IGNORECASE
)


async def _bash(command: str, timeout: int = BASH_TIMEOUT, workdir: Optional[str] = None) -> str:
    """
    Execute a bash command in a throwaway Docker container.
    STRICT SECURITY: If Docker is unavailable, no execution is possible.
    """
    # STRICT SAFETY: Block dangerous commands before they hit Docker
    if _BLOCKED_COMMANDS.search(command):
        return "[denied] Bash command blocked for safety (contains destructive patterns)."

    from plugins.docker_sandbox import run_in_docker
    # Propagate VIRTUAL_ENV hint if present
    venv_path = os.getenv("VIRTUAL_ENV")
    return await run_in_docker(command, timeout=timeout, workdir=workdir)



BASH_TOOL = Tool(
    name="bash",
    description=(
        "Execute a bash command in a sandboxed Linux environment. "
        "Use for running scripts, installing packages, processing files, "
        "or any shell task. Output is captured and returned. "
        f"Working directory: {SANDBOX_DIR}. Timeout: {BASH_TIMEOUT}s."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string",  "description": "Bash command to run"},
            "timeout": {"type": "integer", "description": f"Timeout seconds (default {BASH_TIMEOUT})", "default": BASH_TIMEOUT},
            "workdir": {"type": "string",  "description": "Working directory (must be inside sandbox)"},
        },
        "required": ["command"],
    },
    handler=_bash,
    tags=["code", "shell"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  FILE TOOLS  —  create / view / str_replace
# ══════════════════════════════════════════════════════════════════════════════

def _safe_path(path_str: str) -> Path:
    """Resolve path and ensure it stays inside SANDBOX_DIR."""
    normalized = str(path_str or "").strip()
    if normalized.startswith("/sandbox/"):
        normalized = normalized[len("/sandbox/"):]
    elif normalized == "/sandbox":
        normalized = "."
    p = (SANDBOX_DIR / normalized).resolve()
    if not str(p).startswith(str(SANDBOX_DIR)):
        raise ValueError(f"Path '{path_str}' escapes sandbox — refused")
    return p


async def _file_create(path: Optional[str] = None, content: str = "", filename: Optional[str] = None, encoding: str = "utf-8") -> str:
    """Create or overwrite a file inside the sandbox."""
    try:
        target_path = path or filename
        if not target_path:
            return "file_create error: either 'path' or 'filename' is required"
        target = _safe_path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding=encoding)
        
        rel_path = target.relative_to(SANDBOX_DIR)
        
        # If it's an HTML file, return a JSON preview object for the frontend
        if str(rel_path).lower().endswith(".html"):
            return json.dumps({
                "status": "success",
                "type": "app_view",
                "title": str(rel_path),
                "filename": str(rel_path),
                "path": f"/sandbox/{rel_path}"
            })
            
        return f"Created: {rel_path} ({len(content)} chars)"
    except ValueError as e:
        return f"file_create error: {e}"
    except Exception as e:
        return f"file_create error: {e}"


FILE_CREATE_TOOL = Tool(
    name="file_create",
    description=(
        "Create a new file (or overwrite) at the given path inside the sandbox. "
        "Parent directories are created automatically. "
        "CRITICAL: If creating .html dashboards, you MUST follow the 'V8 Platinum Standard': "
        "1. AESTHETICS: Use 'bg-black' (#000) with glassmorphism and glowing neon accents. "
        "2. INTERACTIVITY: Implement SPA-style views via vanilla JS (toggle .hidden or style.display). "
        "3. ASSETS: Use Lucide icons (cdn) and Unsplash imagery. No generic place-holders."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path":     {"type": "string", "description": "File path relative to sandbox root (Alias: filename)"},
            "filename": {"type": "string", "description": "Alias for path"},
            "content":  {"type": "string", "description": "File content to write (MUST escape double quotes!)"},
            "encoding": {"type": "string", "description": "Encoding (default utf-8)", "default": "utf-8"},
        },
        "required": ["content"],
    },
    handler=_file_create,
    tags=["files"],
)


async def _file_view(
    path: str,
    start_line: Optional[int] = None,
    end_line:   Optional[int] = None,
) -> str:
    """
    Read a file or list a directory.
    If path is a file: returns its contents (or a line range).
    If path is a directory: returns a recursive listing.
    """
    try:
        target = _safe_path(path)
    except ValueError as e:
        return f"file_view error: {e}"

    if not target.exists():
        return f"file_view: '{path}' not found"

    if target.is_dir():
        lines = []
        for item in sorted(target.rglob("*")):
            rel = item.relative_to(SANDBOX_DIR)
            prefix = "📁 " if item.is_dir() else "📄 "
            lines.append(prefix + str(rel))
        return f"Directory: {path}\n" + "\n".join(lines) if lines else f"Directory '{path}' is empty"

    # File
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"file_view error reading file: {e}"

    all_lines = text.splitlines(keepends=True)
    total     = len(all_lines)

    if start_line is not None or end_line is not None:
        s = max(0, (start_line or 1) - 1)
        e = min(total, end_line or total)
        selected = all_lines[s:e]
        header   = f"File: {path} (lines {s+1}–{e} of {total})\n"
    else:
        selected = all_lines
        header   = f"File: {path} ({total} lines)\n"

    content = "".join(selected)
    if len(content) > 12000:
        content = content[:12000] + f"\n\n[truncated — {len(content)-12000} more chars]"

    numbered = []
    for i, line in enumerate(selected, s + 1 if (start_line is not None) else 1):
        numbered.append(f"{i:>4} | {line.rstrip()}")

    return header + "\n".join(numbered)


FILE_VIEW_TOOL = Tool(
    name="file_view",
    description=(
        "Read the contents of a file or list a directory. "
        "Supports line ranges with start_line / end_line. "
        "All paths are relative to the sandbox root."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path":       {"type": "string",  "description": "File or directory path (relative to sandbox)"},
            "start_line": {"type": "integer", "description": "First line to read (1-indexed, optional)"},
            "end_line":   {"type": "integer", "description": "Last line to read inclusive (optional)"},
        },
        "required": ["path"],
    },
    handler=_file_view,
    tags=["files"],
)


async def _file_str_replace(path: str, old_str: str, new_str: str) -> str:
    """
    Replace an exact string in a file.
    The old_str must appear exactly once — fails if 0 or 2+ matches.
    """
    try:
        target = _safe_path(path)
    except ValueError as e:
        return f"file_str_replace error: {e}"

    if not target.exists():
        return f"file_str_replace: '{path}' not found"

    try:
        content = target.read_text(encoding="utf-8")
    except Exception as e:
        return f"file_str_replace: cannot read file: {e}"

    count = content.count(old_str)
    if count == 0:
        return "file_str_replace: old_str not found in file — no changes made"
    if count > 1:
        return (
            f"file_str_replace: old_str appears {count} times — must be unique. "
            "Add more surrounding context to old_str to make it unambiguous."
        )

    updated = content.replace(old_str, new_str, 1)
    try:
        target.write_text(updated, encoding="utf-8")
    except Exception as e:
        return f"file_str_replace: write failed: {e}"

    lines_changed = len(new_str.splitlines()) - len(old_str.splitlines())
    return (
        f"Replaced in {target.relative_to(SANDBOX_DIR)}: "
        f"{len(old_str)} chars → {len(new_str)} chars "
        f"({'+'if lines_changed>=0 else ''}{lines_changed} lines)"
    )


FILE_STR_REPLACE_TOOL = Tool(
    name="file_str_replace",
    description=(
        "Replace an exact string in a file with new content. "
        "old_str must appear exactly once. Add surrounding context "
        "to disambiguate if needed. All paths relative to sandbox root."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path":    {"type": "string", "description": "File path (relative to sandbox)"},
            "old_str": {"type": "string", "description": "Exact string to find and replace (must be unique in file)"},
            "new_str": {"type": "string", "description": "Replacement string (can be empty to delete)"},
        },
        "required": ["path", "old_str", "new_str"],
    },
    handler=_file_str_replace,
    tags=["files"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  WEATHER FETCH  —  open-meteo.com (free, no API key)
# ══════════════════════════════════════════════════════════════════════════════

async def _weather_fetch(location: str, units: str = "metric") -> str:
    """
    Fetch current weather + 3-day forecast.
    Geocodes the location string first, then calls open-meteo.
    Entirely free — no API key required.
    """
    try:
        # Step 1: geocode location string → lat/lon
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            geo = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1, "language": "en", "format": "json"},
            )
            geo.raise_for_status()
            geo_data = geo.json()

        results = geo_data.get("results")
        if not results:
            return f"weather_fetch: could not geocode '{location}'"

        r         = results[0]
        lat       = r["latitude"]
        lon       = r["longitude"]
        place     = f"{r.get('name', location)}, {r.get('country', '')}"

        # Step 2: fetch weather
        temp_unit = "celsius" if units == "metric" else "fahrenheit"
        wind_unit = "kmh"     if units == "metric" else "mph"

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            wx = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude":             lat,
                    "longitude":            lon,
                    "current":              "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_direction_10m",
                    "daily":                "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                    "temperature_unit":     temp_unit,
                    "wind_speed_unit":      wind_unit,
                    "forecast_days":        4,
                    "timezone":             "auto",
                },
            )
            wx.raise_for_status()
            data = wx.json()

        cur   = data.get("current", {})
        daily = data.get("daily", {})
        deg   = "°C" if units == "metric" else "°F"
        spd   = "km/h" if units == "metric" else "mph"

        wmo_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
            55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Slight showers", 81: "Moderate showers", 82: "Heavy showers",
            95: "Thunderstorm", 99: "Thunderstorm with hail",
        }

        def wmo(code) -> str:
            return wmo_codes.get(int(code), f"Code {code}")

        lines = [
            f"Weather for {place}  ({lat:.2f}°N, {lon:.2f}°E)",
            "",
            "── Current ─────────────────────────────",
            f"  Condition:    {wmo(cur.get('weather_code', 0))}",
            f"  Temperature:  {cur.get('temperature_2m', '?')}{deg}  (feels like {cur.get('apparent_temperature', '?')}{deg})",
            f"  Humidity:     {cur.get('relative_humidity_2m', '?')}%",
            f"  Wind:         {cur.get('wind_speed_10m', '?')} {spd}",
            f"  Precip:       {cur.get('precipitation', 0)} mm",
            "",
            "── 3-Day Forecast ───────────────────────",
        ]

        dates  = daily.get("time", [])
        hi     = daily.get("temperature_2m_max", [])
        lo     = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])
        codes  = daily.get("weather_code", [])

        for i in range(1, min(4, len(dates))):
            lines.append(
                f"  {dates[i]}  {wmo(codes[i]):<20} "
                f"↑{hi[i]}{deg} ↓{lo[i]}{deg}  "
                f"precip {precip[i]}mm"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"weather_fetch error: {e}"


WEATHER_TOOL = Tool(
    name="weather_fetch",
    description=(
        "Get current weather and 3-day forecast for any location. "
        "Uses open-meteo.com — no API key required. "
        "Accepts city names, addresses, or coordinates."
    ),
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name or address e.g. 'Toronto' or 'Paris, France'"},
            "units":    {"type": "string", "description": "Units: 'metric' (°C, km/h) or 'imperial' (°F, mph)", "default": "metric"},
        },
        "required": ["location"],
    },
    handler=_weather_fetch,
    tags=["weather", "geo"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  PLACES SEARCH  —  Google Places API
#  Requires: GOOGLE_PLACES_API_KEY env var
# ══════════════════════════════════════════════════════════════════════════════

async def _places_search(
    query:      str,
    location:   Optional[str] = None,
    radius_m:   int = 5000,
    max_results: int = 5,
) -> str:
    """
    Search Google Places API.
    query    — e.g. "ramen restaurants in Tokyo"
    location — optional lat,lon bias e.g. "43.65,-79.38"
    """
    if not GOOGLE_PLACES_KEY:
        return (
            "places_search: GOOGLE_PLACES_API_KEY environment variable not set. "
            "Get a key at https://console.cloud.google.com and enable the Places API."
        )

    try:
        params: dict = {
            "query":  query,
            "key":    GOOGLE_PLACES_KEY,
        }
        if location:
            params["location"] = location
            params["radius"]   = radius_m

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.get(
                "https://maps.googleapis.com/maps/api/place/textsearch/json",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

        if data.get("status") not in ("OK", "ZERO_RESULTS"):
            return f"places_search API error: {data.get('status')} — {data.get('error_message', '')}"

        places = data.get("results", [])[:max_results]
        if not places:
            return f"No places found for: {query}"

        lines = [f"Places results for: {query}\n"]
        for i, p in enumerate(places, 1):
            loc = p.get("geometry", {}).get("location", {})
            lines.append(f"[{i}] {p.get('name', 'Unknown')}")
            lines.append(f"    Address:  {p.get('formatted_address', 'N/A')}")
            lines.append(f"    Rating:   {p.get('rating', 'N/A')} ({p.get('user_ratings_total', 0)} reviews)")
            lines.append(f"    Types:    {', '.join(p.get('types', [])[:4])}")
            lines.append(f"    Lat/Lon:  {loc.get('lat', '')}, {loc.get('lng', '')}")
            lines.append(f"    Place ID: {p.get('place_id', 'N/A')}")
            if p.get("opening_hours"):
                open_now = p["opening_hours"].get("open_now")
                lines.append(f"    Open now: {'Yes' if open_now else 'No'}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"places_search error: {e}"


PLACES_SEARCH_TOOL = Tool(
    name="places_search",
    description=(
        "Search for real places, businesses, and attractions using Google Places. "
        "Returns names, addresses, ratings, types, coordinates, and Place IDs. "
        "Requires GOOGLE_PLACES_API_KEY environment variable."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query":       {"type": "string",  "description": "Natural language query e.g. 'coffee shops near CN Tower Toronto'"},
            "location":    {"type": "string",  "description": "Optional lat,lon bias e.g. '43.65,-79.38'"},
            "radius_m":    {"type": "integer", "description": "Search radius in metres when location set (default 5000)", "default": 5000},
            "max_results": {"type": "integer", "description": "Max results to return (default 5, max 20)", "default": 5},
        },
        "required": ["query"],
    },
    handler=_places_search,
    tags=["places", "geo", "maps"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  PLACES MAP  —  static HTML embed using Google Maps
# ══════════════════════════════════════════════════════════════════════════════

async def _places_map(
    places:    list[dict],
    title:     str = "Map",
    save_path: str = "map.html",
) -> str:
    """
    Generate an interactive HTML map from a list of places.
    Each place: {name, lat, lon, description?}
    Saves to sandbox/save_path. Returns the file path and a summary.
    Renders with Leaflet.js (open source, no API key required).
    """
    if not places:
        return "places_map: no places provided"

    markers_js = []
    bounds_lats = []
    bounds_lons = []

    for p in places:
        lat  = float(p.get("lat") or p.get("latitude") or 0)
        lon  = float(p.get("lon") or p.get("longitude") or p.get("lng") or 0)
        name = p.get("name", "Location").replace("'", "\\'")
        desc = p.get("description", "").replace("'", "\\'")
        bounds_lats.append(lat)
        bounds_lons.append(lon)
        popup = f"<b>{name}</b><br>{desc}" if desc else f"<b>{name}</b>"
        markers_js.append(
            f"L.marker([{lat}, {lon}])"
            f".bindPopup('{popup}')"
            f".addTo(map);"
        )

    center_lat = sum(bounds_lats) / len(bounds_lats)
    center_lon = sum(bounds_lons) / len(bounds_lons)
    markers_str = "\n    ".join(markers_js)

    html = textwrap.dedent(f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>{title}</title>
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
      <style>
        body {{ margin: 0; font-family: sans-serif; }}
        #map {{ height: 100vh; width: 100%; }}
        #title {{ position:absolute; top:10px; left:50%; transform:translateX(-50%);
                  z-index:999; background:white; padding:8px 16px;
                  border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,.2);
                  font-weight:bold; font-size:15px; }}
      </style>
    </head>
    <body>
      <div id="title">{title}</div>
      <div id="map"></div>
      <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
      <script>
        const map = L.map('map').setView([{center_lat}, {center_lon}], 14);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
          attribution: '© OpenStreetMap contributors'
        }}).addTo(map);
        {markers_str}
      </script>
    </body>
    </html>
    """).strip()

    try:
        target = _safe_path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
        return (
            f"Map saved: {target.relative_to(SANDBOX_DIR)}\n"
            f"Contains {len(places)} markers centred at {center_lat:.4f}, {center_lon:.4f}\n"
            f"Open the HTML file in a browser to view the interactive map.\n"
            f"Places mapped:\n" +
            "\n".join(f"  • {p.get('name','?')} ({p.get('lat') or p.get('latitude')}, {p.get('lon') or p.get('longitude') or p.get('lng')})" for p in places)
        )
    except Exception as e:
        return f"places_map error saving file: {e}"


PLACES_MAP_TOOL = Tool(
    name="places_map",
    description=(
        "Generate a local interactive HTML map from a list of places. "
        "Uses Leaflet.js + OpenStreetMap — no API key required. "
        "Each place needs name + lat + lon. Saves map HTML to sandbox. "
        "Best used after places_search to visualise results."
    ),
    parameters={
        "type": "object",
        "properties": {
            "places": {
                "type": "array",
                "description": "List of places to map",
                "items": {
                    "type": "object",
                    "properties": {
                        "name":        {"type": "string"},
                        "lat":         {"type": "number"},
                        "lon":         {"type": "number"},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "lat", "lon"],
                },
            },
            "title":     {"type": "string", "description": "Map title shown on page (default 'Map')", "default": "Map"},
            "save_path": {"type": "string", "description": "Output HTML path inside sandbox (default 'map.html')", "default": "map.html"},
        },
        "required": ["places"],
    },
    handler=_places_map,
    tags=["places", "maps", "geo"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  DEEP MEMORY (Semantic Graph)
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_runtime_embed_model(kwargs) -> str:
    embed_model = str(kwargs.get("_embed_model") or "").strip()
    return embed_model or "nomic-embed-text"


def _build_runtime_graph(kwargs):
    from memory.semantic_graph import SemanticGraph

    embed_model = _resolve_runtime_embed_model(kwargs)
    try:
        return SemanticGraph(embedding_model=embed_model)
    except TypeError:
        return SemanticGraph()


async def _store_memory(subject: str, predicate: str, object_: str, **kwargs) -> str:
    """Explicitly store a factual triplet into the semantic graph memory."""
    try:
        graph = _build_runtime_graph(kwargs)
        owner_id = kwargs.get("_owner_id")
        run_id = kwargs.get("_run_id")
        if owner_id is None and run_id is None:
            await graph.add_triplet(subject, predicate, object_)
        else:
            await graph.add_triplet(
                subject,
                predicate,
                object_,
                owner_id=owner_id,
                run_id=run_id,
            )
        return f"Successfully stored memory: [{subject}] --[{predicate}]--> [{object_}]"
    except Exception as e:
        return f"Failed to store memory: {e}"

STORE_MEMORY_TOOL = Tool(
    name="store_memory",
    description=(
        "Store a single declarative fact or preference about the user or the world into long-term semantic memory. "
        "Use this PROACTIVELY when the user tells you something important to remember for future conversations. "
        "Break the fact into a subject, a predicate (the verb/relationship), and an object."
    ),
    parameters={
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "The entity the fact is about (e.g., 'User', 'System', 'John')"},
            "predicate": {"type": "string", "description": "The relationship (e.g., 'likes', 'is allergic to', 'works at')"},
            "object_": {"type": "string", "description": "The target of the relationship (e.g., 'spicy food', 'peanuts', 'Google')"}
        },
        "required": ["subject", "predicate", "object_"]
    },
    handler=_store_memory,
    tags=["memory", "core"]
)

async def _query_memory(
    topic: str,
    session_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Traverse the semantic graph memory for facts related to a topic."""
    try:
        owner_id = kwargs.get("_owner_id")
        active_session_id = session_id or kwargs.get("_session_id")
        graph = _build_runtime_graph(kwargs)
        from memory.retrieval import unified_memory_search

        payload = await unified_memory_search(
            topic,
            owner_id=owner_id,
            session_id=active_session_id,
            top_k=5,
            graph=graph,
        )
        results = payload.get("results") or []
        stats = payload.get("stats") or {}

        if not results:
            return f"No memories found related to '{topic}'."

        lines = [
            f"Unified memory results for '{topic}' ({len(results)}):",
            f"  - source counts: {stats.get('source_counts', {})}",
        ]
        for item in results:
            kind = str(item.get("kind") or "")
            score = item.get("score")
            source = ", ".join(item.get("sources") or [])
            if kind in {"fact", "triplet"}:
                lines.append(
                    "  - "
                    f"[{item.get('subject', '')}] --[{item.get('predicate', '')}]--> [{item.get('object', '')}] "
                    f"(score: {score}, source: {source})"
                )
            else:
                anchor = str(item.get("anchor") or "").replace("\n", " ").strip()
                if len(anchor) > 180:
                    anchor = anchor[:177].rstrip() + "..."
                lines.append(
                    "  - "
                    f"[{item.get('key', 'memory_anchor')}] {anchor} "
                    f"(score: {score}, source: {source})"
                )
        return "\n".join(lines)
    except Exception as e:
        return f"Failed to query memory: {e}"

QUERY_MEMORY_TOOL = Tool(
    name="query_memory",
    description=(
        "Search the long-term semantic memory graph for stored facts and preferences. "
        "Use this when you need historical context about a specific topic, like 'dietary restrictions' or 'favorite movies'. "
        "Use this for durable remembered facts, corrections, and preferences rather than session evidence."
    ),
    parameters={
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "The broad topic or specific entity to search for (e.g., 'food preferences', 'John')"}
        },
        "required": ["topic"]
    },
    handler=_query_memory,
    tags=["memory", "core"]
)


async def _todo_write(
    tasks: list[dict],
    topic: Optional[str] = None,
    _session_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Create or replace the in-session task list."""
    session_id = _session_id or kwargs.get("session_id")
    if not session_id:
        return "todo_write failed: missing session context"

    tracker = get_session_task_tracker()
    
    # Case 1: tasks is a string (poor LLM formatting)
    if isinstance(tasks, str):
        try:
            tasks = json.loads(tasks)
        except:
            # Maybe it's a raw string list? 
            # We'll try one more fallback if it looks like JSON but has bad delimiters
            pass

    # Case 2: tasks is a list of strings (should be list of dicts)
    if isinstance(tasks, list):
        normalized = []
        for i, t in enumerate(tasks):
            if isinstance(t, str):
                normalized.append({"id": f"task_{i}", "content": t, "status": "pending"})
            else:
                normalized.append(t)
        tasks = normalized

    return tracker.write(session_id, tasks, topic=topic)


TODO_WRITE_TOOL = Tool(
    name="todo_write",
    description=(
        "Create or replace the current session task list for multi-step work. "
        "Use this at the start of complex workflows before calling other tools. "
        "Provide the full task list and an optional workflow topic. "
        "Output is the canonical task state string."
    ),
    parameters={
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "description": "Complete task list for this session.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Stable task id"},
                        "content": {"type": "string", "description": "Action-oriented task text"},
                        "status": {
                            "type": "string",
                            "description": "Task status",
                            "enum": ["pending", "in_progress", "completed"],
                        },
                        "priority": {
                            "type": "string",
                            "description": "Task priority",
                            "enum": ["low", "medium", "high"],
                        },
                    },
                    "required": ["id", "content"],
                },
            },
            "topic": {
                "type": "string",
                "description": "Short workflow/topic label for the active task set.",
            },
        },
        "required": ["tasks"],
    },
    handler=_todo_write,
    tags=["tasking", "planning"],
)


async def _todo_update(task_id: str, status: str, _session_id: Optional[str] = None, **kwargs) -> str:
    """Update one task status in the current session."""
    session_id = _session_id or kwargs.get("session_id")
    if not session_id:
        return "todo_update failed: missing session context"

    tracker = get_session_task_tracker()
    return tracker.update(session_id, task_id, status)


TODO_UPDATE_TOOL = Tool(
    name="todo_update",
    description=(
        "Update the status of a single task in the current session task list. "
        "Use immediately when a task starts or completes so the workflow state stays accurate. "
        "Output is the updated canonical task state string."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "Task id to update"},
            "status": {
                "type": "string",
                "description": "New status value",
                "enum": ["pending", "in_progress", "completed"],
            },
        },
        "required": ["task_id", "status"],
    },
    handler=_todo_update,
    tags=["tasking", "planning"],
)

# ══════════════════════════════════════════════════════════════════════════════
#  HTML RENDERING  —  Isolated Sandboxed Viewer
# ══════════════════════════════════════════════════════════════════════════════

async def _generate_app(html_content: str, title: str = "Standalone App") -> str:
    """
    Saves HTML content to a unique file and injects a premium 'V8 Platinum' theme.
    Returns metadata for the frontend to render an isolated iframe.
    """
    import hashlib
    import time
    
    # Generate a clean filename based on title
    safe_title = re.sub(r'[^a-zA-Z0-9]', '_', title).lower().strip("_")[:30]
    filename = f"{safe_title}.html" if safe_title else "app.html"
    file_path = SANDBOX_DIR / filename
    
    # Inject Premium V8 Platinum Theme if not already present
    if "<!DOCTYPE html>" not in html_content:
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{ --bg: #000; --text: #fff; --primary: #00ff85; }}
        body {{ 
            background: #000; color: var(--text); font-family: 'Inter', sans-serif; margin: 0; min-height: 100vh;
            -webkit-font-smoothing: antialiased;
        }}
        .glass {{ background: rgba(255,255,255,0.03); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); }}
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-thumb {{ background: #222; border-radius: 10px; }}
        .v8-card {{ background: #050505; border: 1px solid #111; border-radius: 12px; transition: all 0.3s ease; }}
        .v8-card:hover {{ border-color: var(--primary); box-shadow: 0 0 20px rgba(0,255,133,0.1); }}
    </style>
</head>
<body class="p-6">
    {html_content}
    <script>lucide.createIcons();</script>
</body>
</html>"""
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    return json.dumps({
        "status": "success",
        "type": "app_view",
        "title": title,
        "filename": filename,
        "path": f"/sandbox/{filename}"
    })

GENERATE_APP_TOOL = Tool(
    name="generate_app",
    description=(
        "Generate a standalone, interactive HTML application or dashboard. "
        "MUST follow the 'V8 Platinum Standard' (Production-grade, True Black #000). "
        "Requires modern typography (e.g. Outfit, Syne), glassmorphism, and SPA-style internal routing. "
        "Apps MUST feel alive with micro-animations and staggered reveals."
    ),
    parameters={
        "type": "object",
        "properties": {
            "html_content": {"type": "string", "description": "The HTML body. CRITICAL: Escape all double quotes (\") as \\\" and use 'bg-black' for backgrounds."},
            "title": {"type": "string", "description": "The application title."}
        },
        "required": ["html_content"]
    },
    handler=_generate_app,
    tags=["visual", "app"]
)

# ══════════════════════════════════════════════════════════════════════════════
#  Registration helper
# ══════════════════════════════════════════════════════════════════════════════

async def _pdf_processor(
    action: str,
    path: Optional[str] = None,
    output_path: Optional[str] = None,
    content: Optional[str] = None,
    paths: Optional[list[str]] = None,
    pages: Optional[list[int]] = None,
    rotation: Optional[int] = None,
) -> str:
    """
    Unified PDF processor tool handler.
    Actions: read, create, merge, split, rotate, metadata.
    """
    try:
        def _pdf_preview_payload(pdf_path: str, action_name: str, **extra) -> str:
            rel = str(Path(pdf_path).as_posix()).lstrip("/")
            payload = {
                "status": "success",
                "action": action_name,
                "type": "pdf_preview",
                "file": rel,
                "path": f"/sandbox/{rel}",
                "url": f"/sandbox/{rel}",
                "title": Path(rel).name,
            }
            payload.update(extra)
            return json.dumps(payload)

        if action == "create":
            if not output_path or not content:
                return "Error: 'output_path' and 'content' required for create."
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.pdfgen import canvas
            except ImportError:
                return "Error: 'reportlab' library missing. Run 'pip install reportlab'."
            
            target = _safe_path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            c = canvas.Canvas(str(target), pagesize=letter)
            width, height = letter
            text_lines = content.split('\n')
            y = height - 100
            for line in text_lines:
                c.drawString(100, y, line)
                y -= 15
            c.save()
            return _pdf_preview_payload(output_path, "create")

        if action == "read":
            if not path: return "Error: 'path' required for read."
            target = _safe_path(path)
            if not target.exists(): return f"Error: {path} not found."
            
            try:
                import pdfplumber
                with pdfplumber.open(target) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                    return _pdf_preview_payload(
                        path,
                        "read",
                        content=text,
                        pages=len(pdf.pages),
                    )
            except ImportError:
                # Fallback to pypdf
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(target)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                    return _pdf_preview_payload(
                        path,
                        "read",
                        content=text,
                        pages=len(reader.pages),
                        note="using pypdf fallback",
                    )
                except ImportError:
                    return "Error: 'pypdf' or 'pdfplumber' missing."

        if action == "merge":
            if not paths or not output_path: return "Error: 'paths' and 'output_path' required for merge."
            try:
                from pypdf import PdfWriter, PdfReader
                writer = PdfWriter()
                for p in paths:
                    reader = PdfReader(_safe_path(p))
                    for page in reader.pages:
                        writer.add_page(page)
                target = _safe_path(output_path)
                with open(target, "wb") as f:
                    writer.write(f)
                return _pdf_preview_payload(output_path, "merge")
            except ImportError:
                return "Error: 'pypdf' library missing."

        if action == "split":
            if not path: return "Error: 'path' required for split."
            try:
                from pypdf import PdfWriter, PdfReader
                reader = PdfReader(_safe_path(path))
                for i, page in enumerate(reader.pages):
                    writer = PdfWriter()
                    writer.add_page(page)
                    out_name = f"{Path(path).stem}_page_{i+1}.pdf"
                    target = _safe_path(out_name)
                    with open(target, "wb") as f:
                        writer.write(f)
                return json.dumps({"status": "success", "action": "split", "pages": len(reader.pages)})
            except ImportError:
                return "Error: 'pypdf' library missing."

        return f"Error: Unknown action '{action}'."

    except Exception as e:
        return f"Error during PDF operation: {e}"

PDF_PROCESSOR_TOOL = Tool(
    name="pdf_processor",
    description=(
        "Advanced PDF toolkit. Actions: 'read' (extract text), 'create' (basic PDF), "
        "'merge' (multiple files), 'split' (one file per page), 'rotate', 'metadata'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["read", "create", "merge", "split", "rotate", "metadata"]},
            "path": {"type": "string", "description": "Target PDF path in sandbox."},
            "output_path": {"type": "string", "description": "Output path for create/merge/split."},
            "content": {"type": "string", "description": "Text content for 'create'."},
            "paths": {"type": "array", "items": {"type": "string"}, "description": "List of PDF paths to merge."},
            "pages": {"type": "array", "items": {"type": "integer"}, "description": "Specific pages to extract/split."},
            "rotation": {"type": "integer", "description": "Degrees to rotate (90, 180, 270)."}
        },
        "required": ["action"]
    },
    handler=_pdf_processor,
    tags=["file", "pdf"]
)

# ══════════════════════════════════════════════════════════════════════════════
#  DELEGATION  —  Agentic Tool for orchestrating other agents
# ══════════════════════════════════════════════════════════════════════════════

async def _delegate_to_agent(target_agent_id: Optional[str] = None, task: Optional[str] = None, **kwargs) -> str:
    """Handler for the delegate_to_agent tool."""
    if not _agent_manager:
        return "Error: Agent Manager not initialized for delegation."
    
    try:
        # Compatibility layer:
        # Some models emit legacy shapes like {"agent": "...", "function": "...", "args": [...]}
        # even when instructed to use {"target_agent_id", "task"}.
        target_agent_id = (
            target_agent_id
            or kwargs.get("agent")
            or kwargs.get("agent_id")
            or kwargs.get("target")
            or "default"
        )

        if not task:
            task = kwargs.get("task")

        legacy_function = kwargs.get("function")
        legacy_args = kwargs.get("args")
        if not task and isinstance(legacy_function, str):
            if legacy_function in {"write_report", "create_report", "create_file", "file_create"} and isinstance(legacy_args, (list, tuple)) and legacy_args:
                out_path = str(legacy_args[0])
                report_body = str(legacy_args[1]) if len(legacy_args) > 1 else ""
                task = (
                    f"Create a file called '{out_path}' in the sandbox. "
                    "Use the file_create tool immediately. "
                    f"Write this content:\n{report_body}"
                )
            elif legacy_args is not None:
                task = f"Execute function '{legacy_function}' with args: {json.dumps(legacy_args)}"
            else:
                task = f"Execute function '{legacy_function}'."

        if not task or not str(task).strip():
            return (
                "Delegation error: missing task. "
                "Provide either {'target_agent_id','task'} or legacy {'agent','function','args'}."
            )

        parent_id = kwargs.get("_session_id")
        log("agent", "system", f"Delegating task to '{target_agent_id}': {str(task)[:50]}...", level="info")
        parent_run_id = kwargs.get("_run_id")
        owner_id = kwargs.get("_owner_id")
        runtime_kind_override = str(kwargs.get("_delegate_runtime_kind") or "").strip().lower() or "managed"
        delegate_kwargs = {
            "parent_id": parent_id,
            "parent_run_id": parent_run_id,
            "owner_id": owner_id,
            "runtime_kind_override": runtime_kind_override,
        }
        result = await _agent_manager.run_agent_task(target_agent_id, task, **delegate_kwargs)
        return result
    except Exception as e:
        return f"Delegation error: {e}"

DELEGATE_TO_AGENT_TOOL = Tool(
    name="delegate_to_agent",
    description=(
        "Delegate a specific task to another specialized agent. "
        "Use this for research, coding, or complex logic that requires a different personality or mindset."
    ),
    parameters={
        "type": "object",
        "properties": {
            "target_agent_id": {
                "type": "string", 
                "description": "ID of the agent to delegate to (e.g., 'coder', 'researcher', 'default')."
            },
            "task": {
                "type": "string",
                "description": "Detailed description of what the sub-agent should do."
            },
            "agent": {
                "type": "string",
                "description": "Legacy alias for target_agent_id."
            },
            "function": {
                "type": "string",
                "description": "Legacy function-style delegation request."
            },
            "args": {
                "type": "array",
                "description": "Legacy function arguments for compatibility."
            },
        },
        "required": []
    },
    handler=_delegate_to_agent,
    tags=["agentic", "system"]
)

async def _rag_search(query: str, top_k: int = 5, **kwargs) -> str:
    """Search everything retrieved in this conversation session."""
    session_id = kwargs.get("_session_id")
    if not session_id:
        return "rag_search requires a session context."
    from memory.session_rag import get_session_rag
    rag = get_session_rag(session_id, owner_id=kwargs.get("_owner_id"))
    results = await rag.query(query, top_k=top_k)
    return rag.format_results_for_llm(results, query)

RAG_SEARCH_TOOL = Tool(
    name="rag_search",
    description=(
        "Search everything retrieved earlier in this conversation — web pages fetched, "
        "files created, tool results, uploaded documents. Use this for session evidence recall "
        "before web_search to avoid redundant fetches. Do not use it for durable user preference memory."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in session memory",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default 5)",
            },
        },
        "required": ["query"],
    },
    handler=_rag_search,
    tags=["memory", "rag"],
)

# ══════════════════════════════════════════════════════════════════════════════
#  Registration helper
# ══════════════════════════════════════════════════════════════════════════════

ALL_TOOLS = [
    WEB_SEARCH_TOOL,
    WEB_FETCH_TOOL,
    IMAGE_SEARCH_TOOL,
    BASH_TOOL,
    FILE_CREATE_TOOL,
    FILE_VIEW_TOOL,
    FILE_STR_REPLACE_TOOL,
    WEATHER_TOOL,
    PLACES_SEARCH_TOOL,   # BUG FIX: was missing from ALL_TOOLS — caused 'not found in global registry' warning
    PLACES_MAP_TOOL,
    STORE_MEMORY_TOOL,
    QUERY_MEMORY_TOOL,
    TODO_WRITE_TOOL,
    TODO_UPDATE_TOOL,
    GENERATE_APP_TOOL,
    PDF_PROCESSOR_TOOL,
    RAG_SEARCH_TOOL,
    DELEGATE_TO_AGENT_TOOL,
]

def register_all_tools(registry: ToolRegistry, agent_manager: Optional[AgentManager] = None) -> None:
    """Register every built-in tool. Call this in main.py after creating tool_registry."""
    global _agent_manager
    if agent_manager:
        _agent_manager = agent_manager

    for tool in ALL_TOOLS:
        registry.register(tool)
    print(f"[tools] Registered {len(ALL_TOOLS)} built-in tools")


def register_tools(registry: ToolRegistry, *names: str) -> None:
    """Register a specific subset of tools by name."""
    tool_map = {t.name: t for t in ALL_TOOLS}
    for name in names:
        if name in tool_map:
            registry.register(tool_map[name])
        else:
            print(f"[tools] WARNING: unknown tool '{name}' — skipped")
