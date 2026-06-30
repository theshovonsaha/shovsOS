#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HOST="${HARNESS_BACKEND_HOST:-127.0.0.1}"
PORT="${HARNESS_BACKEND_PORT:-8091}"
FRONTEND_HOST="${HARNESS_FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${HARNESS_FRONTEND_PORT:-5177}"
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

"$PY" scripts/harness_backend.py --host "$HOST" --port "$PORT" &
BACKEND_PID=$!

cleanup() {
  kill "$BACKEND_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "Backend:  http://$HOST:$PORT"
echo "Frontend: http://$FRONTEND_HOST:$FRONTEND_PORT"

cd frontend
../../../frontend_consumer/node_modules/.bin/vite --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"
