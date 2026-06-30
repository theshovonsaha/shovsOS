from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def main() -> int:
    checks = [
        _check_python(),
        _check_pytest(),
        _check_node(),
        _check_frontend_tooling(),
        _check_homebrew(),
        _check_llamacpp_binary(),
        _check_llamacpp_server(),
    ]
    print(json.dumps({"root": str(ROOT), "checks": checks}, indent=2))
    blocking = [item for item in checks if item["status"] == "fail" and item.get("required")]
    return 1 if blocking else 0


def _check_python() -> dict[str, Any]:
    ok = sys.version_info >= (3, 10)
    return {
        "name": "python",
        "status": "pass" if ok else "fail",
        "required": True,
        "detail": platform.python_version(),
    }


def _check_pytest() -> dict[str, Any]:
    try:
        import pytest  # noqa: F401
        return {"name": "pytest", "status": "pass", "required": True, "detail": "importable"}
    except Exception as exc:
        return {"name": "pytest", "status": "fail", "required": True, "detail": str(exc)}


def _check_node() -> dict[str, Any]:
    node = shutil.which("node")
    npm = shutil.which("npm")
    detail = {"node": node, "npm": npm}
    if not node or not npm:
        return {"name": "node/npm", "status": "warn", "required": False, "detail": detail}
    try:
        detail["node_version"] = subprocess.check_output([node, "--version"], text=True).strip()
        detail["npm_version"] = subprocess.check_output([npm, "--version"], text=True).strip()
    except Exception as exc:
        detail["version_error"] = str(exc)
    return {"name": "node/npm", "status": "pass", "required": False, "detail": detail}


def _check_frontend_tooling() -> dict[str, Any]:
    vite = REPO / "frontend_consumer" / "node_modules" / ".bin" / "vite"
    tsc = REPO / "frontend_consumer" / "node_modules" / ".bin" / "tsc"
    ok = vite.exists() and tsc.exists()
    return {
        "name": "harness frontend tooling",
        "status": "pass" if ok else "warn",
        "required": False,
        "detail": {
            "vite": str(vite),
            "vite_exists": vite.exists(),
            "tsc_exists": tsc.exists(),
            "fix": "from repo root run npm install, or install frontend dependencies used by the harness demo",
        },
    }


def _check_homebrew() -> dict[str, Any]:
    brew = shutil.which("brew")
    if not brew:
        return {
            "name": "homebrew",
            "status": "warn",
            "required": False,
            "detail": {"brew": None, "fix": "install llama.cpp from source, or install Homebrew first"},
        }
    detail: dict[str, Any] = {"brew": brew}
    try:
        prefix = subprocess.check_output([brew, "--prefix"], text=True, stderr=subprocess.STDOUT).strip()
        detail["prefix"] = prefix
        prefix_path = Path(prefix)
        detail["prefix_exists"] = prefix_path.exists()
        detail["prefix_writable"] = os.access(prefix, os.W_OK)
        locks = prefix_path / "var" / "homebrew" / "locks"
        detail["locks"] = str(locks)
        detail["locks_writable"] = os.access(locks, os.W_OK) if locks.exists() else None
        if detail["prefix_writable"] is False or detail["locks_writable"] is False:
            return {
                "name": "homebrew",
                "status": "warn",
                "required": False,
                "detail": {
                    **detail,
                    "fix": "Homebrew prefix is not writable by this user. Run brew doctor and repair ownership before brew install.",
                },
            }
    except Exception as exc:
        detail["error"] = str(exc)
        return {"name": "homebrew", "status": "warn", "required": False, "detail": detail}
    return {"name": "homebrew", "status": "pass", "required": False, "detail": detail}


def _check_llamacpp_binary() -> dict[str, Any]:
    bins = {
        "llama-server": shutil.which("llama-server"),
        "llama-cli": shutil.which("llama-cli"),
    }
    status = "pass" if any(bins.values()) else "warn"
    return {
        "name": "llama.cpp binaries",
        "status": status,
        "required": False,
        "detail": {
            **bins,
            "fix": "install llama.cpp, then ensure llama-server is on PATH",
        },
    }


def _check_llamacpp_server() -> dict[str, Any]:
    base_url = os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8081/v1").rstrip("/")
    parsed = _parse_host_port(base_url)
    detail: dict[str, Any] = {"base_url": base_url}
    if parsed:
        host, port = parsed
        detail["port_open"] = _port_open(host, port)
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = [
            str(item.get("id") or item.get("name") or "")
            for item in payload.get("data", [])
            if isinstance(item, dict)
        ]
        detail["models"] = [item for item in models if item][:10]
        return {"name": "llama.cpp server", "status": "pass", "required": False, "detail": detail}
    except urllib.error.HTTPError as exc:
        detail["error"] = f"HTTP {exc.code}"
    except Exception as exc:
        detail["error"] = f"{type(exc).__name__}: {exc}"
    detail["fix"] = "start llama-server on localhost, e.g. --host 127.0.0.1 --port 8081"
    return {"name": "llama.cpp server", "status": "warn", "required": False, "detail": detail}


def _parse_host_port(base_url: str) -> tuple[str, int] | None:
    try:
        without_scheme = base_url.split("://", 1)[1] if "://" in base_url else base_url
        host_port = without_scheme.split("/", 1)[0]
        host, port = host_port.rsplit(":", 1)
        return host, int(port)
    except Exception:
        return None


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
