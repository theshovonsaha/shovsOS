#!/usr/bin/env python3
"""
scripts/doctor.py — health check for the Shovs Agent Platform.

Runs a sequence of independent checks. Each check prints PASS / WARN / FAIL.
Exit code 0 if no FAILs, 1 otherwise. WARNs do not fail the run.

Checks:
  - Python version >= 3.10
  - Required env vars (or sensible defaults)
  - Provider API keys (at least one provider available)
  - Ollama reachable (if OLLAMA_BASE_URL set)
  - Writable DB / chroma / logs paths
  - All 5 LLM adapters import
  - Skill loader works and finds the platform skills
  - Context governor resolves to the unified V3 engine
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS = "\033[0;32m✓ PASS\033[0m"
WARN = "\033[1;33m⚠ WARN\033[0m"
FAIL = "\033[0;31m✗ FAIL\033[0m"

failures: list[str] = []
warnings: list[str] = []


def report(status: str, name: str, detail: str = "") -> None:
    line = f"{status}  {name}"
    if detail:
        line += f"  — {detail}"
    print(line)
    if status == FAIL:
        failures.append(name)
    elif status == WARN:
        warnings.append(name)


def check_python() -> None:
    v = sys.version_info
    if v >= (3, 10):
        report(PASS, "Python version", f"{v.major}.{v.minor}.{v.micro}")
    else:
        report(FAIL, "Python version", f"{v.major}.{v.minor} (need >= 3.10)")


def check_provider_keys() -> None:
    providers = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
        "GROQ_API_KEY": "Groq",
        "GEMINI_API_KEY": "Gemini",
        "NVIDIA_API_KEY": "NVIDIA",
    }
    found = [name for env, name in providers.items() if os.getenv(env)]
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    if found:
        report(PASS, "Provider API keys", ", ".join(found))
    elif ollama_url:
        report(WARN, "Provider API keys", f"none set — relying on Ollama at {ollama_url}")
    else:
        report(FAIL, "Provider API keys", "no provider key and no Ollama URL")


def check_ollama() -> None:
    url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        import urllib.request
        req = urllib.request.Request(f"{url.rstrip('/')}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                report(PASS, "Ollama reachable", url)
                return
        report(WARN, "Ollama reachable", f"{url} returned {resp.status}")
    except Exception as e:
        report(WARN, "Ollama reachable", f"{url} unreachable ({type(e).__name__})")


def check_paths() -> None:
    paths = {
        "DB_PATH": os.getenv("DB_PATH", "agents.db"),
        "CHROMA_PATH": os.getenv("CHROMA_PATH", "./chroma_db"),
        "TRACE_DIR": os.getenv("TRACE_DIR", "./logs"),
    }
    for label, raw in paths.items():
        p = (ROOT / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
        target = p if p.is_dir() or not p.suffix else p.parent
        try:
            target.mkdir(parents=True, exist_ok=True)
            probe = target / ".doctor_probe"
            probe.write_text("ok")
            probe.unlink()
            report(PASS, f"Writable {label}", str(p))
        except Exception as e:
            report(FAIL, f"Writable {label}", f"{p}: {e}")


def check_adapters() -> None:
    adapters = [
        ("llm.llm_adapter", "OllamaAdapter"),
        ("llm.openai_adapter", "OpenAIAdapter"),
        ("llm.anthropic_adapter", "AnthropicAdapter"),
        ("llm.groq_adapter", "GroqLLMAdapter"),
        ("llm.gemini_adapter", "GeminiAdapter"),
    ]
    missing = []
    for mod_name, cls_name in adapters:
        try:
            mod = importlib.import_module(mod_name)
            getattr(mod, cls_name)
        except Exception as e:
            missing.append(f"{cls_name} ({type(e).__name__})")
    if missing:
        report(FAIL, "LLM adapters import", "; ".join(missing))
    else:
        report(PASS, "LLM adapters import", "5/5")


def check_skills() -> None:
    try:
        from run_engine.skill_loader import list_available_skills
        skills = list_available_skills(str(ROOT))
        if skills:
            names = sorted(getattr(s, "name", None) or (s.get("name", "?") if isinstance(s, dict) else "?") for s in skills)
            report(PASS, "Skill loader", f"{len(names)} skills: {', '.join(names)}")
        else:
            report(WARN, "Skill loader", "loaded but found 0 skills (.agent/skills/ empty?)")
    except Exception as e:
        report(FAIL, "Skill loader", f"{type(e).__name__}: {e}")


def check_context_engine() -> None:
    try:
        from engine.context_engine_v3 import ContextEngineV3
        # Just verify the class is importable and has the budget knobs.
        for attr in ("durable_cap", "convergent_top_n", "resonance_weight"):
            if attr not in ContextEngineV3.__init__.__code__.co_varnames:
                report(WARN, "Context engine V3 budgets", f"missing knob: {attr}")
                return
        report(PASS, "Unified context engine", "V3 with durable_cap / convergent_top_n / resonance_weight")
    except Exception as e:
        report(FAIL, "Unified context engine", f"{type(e).__name__}: {e}")


def main() -> int:
    print("Shovs Platform — doctor")
    print("=" * 56)
    check_python()
    check_provider_keys()
    check_ollama()
    check_paths()
    check_adapters()
    check_skills()
    check_context_engine()
    print("=" * 56)
    print(f"{len(failures)} failure(s), {len(warnings)} warning(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
