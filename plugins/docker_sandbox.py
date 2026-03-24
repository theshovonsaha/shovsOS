"""
Docker Sandbox
--------------
Executes commands in an isolated Docker environment with two modes:

  - dedicated (default): a long-lived sandbox container reused across calls
  - ephemeral: create/run/remove a throwaway container per command

This gives safer command execution while keeping the sandbox responsive.

Requirements:
    pip install docker

Environment variables:
    DOCKER_SANDBOX_MODE            — dedicated|ephemeral (default: dedicated)
    DOCKER_SANDBOX_IMAGE           — base image (default: agent-sandbox:latest)
    DOCKER_SANDBOX_FALLBACK_IMAGE  — fallback image (default: python:3.11-slim)
    DOCKER_SANDBOX_CONTAINER_NAME  — dedicated container name (default: agent-sandbox-exec)
    DOCKER_SANDBOX_WORKDIR         — mounted workdir inside container (default: /workspace)
    DOCKER_SANDBOX_NETWORK         — none|bridge|host (default: none)
    DOCKER_SANDBOX_MEMORY          — memory limit (default: 512m)
    DOCKER_SANDBOX_PIDS_LIMIT      — process cap (default: 128)
    DOCKER_SANDBOX_CPU_COUNT       — cpu budget in cores (default: 1.0)
    DOCKER_SANDBOX_TIMEOUT         — max seconds per command (default: 30)
    DOCKER_SANDBOX_AUTO_PULL       — pull image if missing (default: false)
    DOCKER_DISABLED                — set to true to disable execution entirely

STRICT POLICY: If Docker is unavailable or disabled, no execution is performed.
"""

from __future__ import annotations

import asyncio
import os
import shlex
from pathlib import Path
from typing import Optional

def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: Optional[str], default: int) -> int:
    try:
        return int(str(value))
    except Exception:
        return default


def _as_float(value: Optional[str], default: float) -> float:
    try:
        return float(str(value))
    except Exception:
        return default


SANDBOX_MODE           = os.getenv("DOCKER_SANDBOX_MODE", "dedicated").strip().lower()
SANDBOX_IMAGE          = os.getenv("DOCKER_SANDBOX_IMAGE", "agent-sandbox:latest")
SANDBOX_FALLBACK_IMAGE = os.getenv("DOCKER_SANDBOX_FALLBACK_IMAGE", "python:3.11-slim")
SANDBOX_CONTAINER_NAME = os.getenv("DOCKER_SANDBOX_CONTAINER_NAME", "agent-sandbox-exec")
SANDBOX_WORKDIR        = os.getenv("DOCKER_SANDBOX_WORKDIR", "/workspace").strip() or "/workspace"
SANDBOX_NETWORK        = os.getenv("DOCKER_SANDBOX_NETWORK", "none").strip() or "none"
SANDBOX_MEMORY         = os.getenv("DOCKER_SANDBOX_MEMORY", "512m")
SANDBOX_PIDS_LIMIT     = _as_int(os.getenv("DOCKER_SANDBOX_PIDS_LIMIT"), 128)
SANDBOX_CPU_COUNT      = max(0.1, _as_float(os.getenv("DOCKER_SANDBOX_CPU_COUNT"), 1.0))
SANDBOX_TIMEOUT        = max(1, _as_int(os.getenv("DOCKER_SANDBOX_TIMEOUT"), 30))
SANDBOX_AUTO_PULL      = _as_bool(os.getenv("DOCKER_SANDBOX_AUTO_PULL"), False)

_VALID_MODES = {"dedicated", "ephemeral"}

# Files the agent creates live here on host and are mounted into the sandbox.
SANDBOX_DIR = Path(os.getenv("SANDBOX_DIR", "./agent_sandbox")).resolve()
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)


def _is_connection_error(message: str) -> bool:
    msg = (message or "").lower()
    return any(token in msg for token in ("connection", "socket", "daemon", "permission denied"))


def _resolve_container_workdir(workdir: Optional[str]) -> str:
    if not workdir:
        return SANDBOX_WORKDIR

    resolved = (SANDBOX_DIR / workdir).resolve()
    if str(resolved).startswith(str(SANDBOX_DIR)):
        relative = resolved.relative_to(SANDBOX_DIR).as_posix()
        if relative and relative != ".":
            return f"{SANDBOX_WORKDIR.rstrip('/')}/{relative}"
    return SANDBOX_WORKDIR


def _runtime_limits_kwargs() -> dict:
    kwargs = {
        "mem_limit": SANDBOX_MEMORY,
        "pids_limit": SANDBOX_PIDS_LIMIT,
        "network_mode": SANDBOX_NETWORK,
        "security_opt": ["no-new-privileges:true"],
    }
    nano_cpus = int(SANDBOX_CPU_COUNT * 1_000_000_000)
    if nano_cpus > 0:
        kwargs["nano_cpus"] = nano_cpus
    return kwargs


def _ensure_image(client, image: str) -> bool:
    import docker

    try:
        client.images.get(image)
        return True
    except docker.errors.ImageNotFound:
        if not SANDBOX_AUTO_PULL:
            return False
        try:
            client.images.pull(image)
            return True
        except Exception:
            return False
    except Exception:
        return False


def _select_image(client) -> Optional[str]:
    candidates = [SANDBOX_IMAGE]
    if SANDBOX_FALLBACK_IMAGE and SANDBOX_FALLBACK_IMAGE not in candidates:
        candidates.append(SANDBOX_FALLBACK_IMAGE)

    for image in candidates:
        if _ensure_image(client, image):
            return image
    return None


def _ensure_dedicated_container(client):
    import docker

    try:
        container = client.containers.get(SANDBOX_CONTAINER_NAME)
        if container.status != "running":
            container.start()
            container.reload()
        return container
    except docker.errors.NotFound:
        image = _select_image(client)
        if not image:
            raise RuntimeError(
                "Sandbox image unavailable. Build/start it with 'docker compose up -d agent-sandbox' "
                "or set DOCKER_SANDBOX_IMAGE to an existing local image."
            )

        container = client.containers.create(
            image=image,
            name=SANDBOX_CONTAINER_NAME,
            command=["sleep", "infinity"],
            working_dir=SANDBOX_WORKDIR,
            volumes={str(SANDBOX_DIR): {"bind": SANDBOX_WORKDIR, "mode": "rw"}},
            detach=True,
            tty=True,
            stdin_open=True,
            **_runtime_limits_kwargs(),
        )
        container.start()
        container.reload()
        return container


def _exec_in_dedicated_container(container, command: str, timeout: int, workdir: Optional[str]) -> str:
    container_workdir = _resolve_container_workdir(workdir)
    wrapped = f"timeout --signal=KILL {max(1, int(timeout))}s bash -lc {shlex.quote(command)}"

    result = container.exec_run(
        cmd=["bash", "-lc", wrapped],
        workdir=container_workdir,
        stdout=True,
        stderr=True,
    )
    output = (result.output or b"").decode("utf-8", errors="replace").strip()

    if result.exit_code == 124:
        return f"[timeout] Command exceeded {timeout}s limit."
    if result.exit_code != 0 and not output:
        return f"[error] Command failed with exit code {result.exit_code}"
    return output


def _run_ephemeral_container(client, command: str, timeout: int, workdir: Optional[str]) -> str:
    image = _select_image(client)
    if not image:
        raise RuntimeError(
            "Sandbox image unavailable. Build/start it with 'docker compose up -d agent-sandbox' "
            "or set DOCKER_SANDBOX_IMAGE to an existing local image."
        )

    wrapped = f"timeout --signal=KILL {max(1, int(timeout))}s bash -lc {shlex.quote(command)}"
    container_workdir = _resolve_container_workdir(workdir)

    container = client.containers.create(
        image=image,
        command=["bash", "-lc", wrapped],
        working_dir=container_workdir,
        volumes={str(SANDBOX_DIR): {"bind": SANDBOX_WORKDIR, "mode": "rw"}},
        **_runtime_limits_kwargs(),
    )

    container.start()
    try:
        result = container.wait(timeout=max(3, timeout + 5))
        status_code = int(result.get("StatusCode", 0))
        output = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace").strip()

        if status_code == 124:
            return f"[timeout] Command exceeded {timeout}s limit."
        if status_code != 0 and not output:
            return f"[error] Command failed with exit code {status_code}"
        return output
    finally:
        try:
            container.remove(force=True)
        except Exception:
            pass


async def run_in_docker(
    command: str,
    timeout: int = SANDBOX_TIMEOUT,
    workdir: Optional[str] = None,
) -> str:
    """
    Execute a bash command inside a throwaway Docker container.
    Returns stdout+stderr as a single string.
    
    STRICT SECURITY: 
    If DOCKER_DISABLED=true or Docker daemon is not available, 
    returns a denial message. No local fallback is allowed.
    """
    if _as_bool(os.getenv("DOCKER_DISABLED"), False):
        return "[denied] Docker execution is explicitly disabled via DOCKER_DISABLED."

    try:
        import docker
    except ImportError:
        return "[error] 'docker' python library not installed. Cannot execute command safely."

    try:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None,
            lambda: _run_docker_sync(command, timeout, workdir),
        )
        return await asyncio.wait_for(future, timeout=max(10, timeout + 10))
    except asyncio.TimeoutError:
        return f"[timeout] Command exceeded {timeout}s limit."
    except Exception as e:
        err_str = str(e)
        if _is_connection_error(err_str):
            return "[denied] Docker Desktop is not running. For safety, execution is strictly disallowed."

        return f"[error] Docker execution failed: {e}"


def _run_docker_sync(command: str, timeout: int, workdir: Optional[str]) -> str:
    """Synchronous docker execution worker used from a thread executor."""
    import docker

    try:
        client = docker.from_env()
    except Exception as e:
        return f"[denied] Cannot connect to Docker daemon: {e}. Execution disallowed."

    try:
        mode = SANDBOX_MODE if SANDBOX_MODE in _VALID_MODES else "dedicated"
        if mode == "ephemeral":
            return _run_ephemeral_container(client, command, timeout, workdir)

        container = _ensure_dedicated_container(client)
        return _exec_in_dedicated_container(container, command, timeout, workdir)

    except Exception as e:
        err = str(e)
        if _is_connection_error(err):
            return f"[denied] Cannot connect to Docker daemon: {err}. Is Docker Desktop running?"
        return f"[error] {err}"
    finally:
        try:
            client.close()
        except Exception:
            pass
