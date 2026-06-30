#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HOST="${HARNESS_BACKEND_HOST:-127.0.0.1}"
PORT="${HARNESS_BACKEND_PORT:-8091}"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "../../venv/bin/python" ]; then
    PY="../../venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
  else
    PY="python"
  fi
fi

"$PY" scripts/harness_backend.py --host "$HOST" --port "$PORT"
