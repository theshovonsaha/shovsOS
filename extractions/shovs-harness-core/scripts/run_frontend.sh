#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../frontend"

../../../frontend_consumer/node_modules/.bin/vite --host "${HARNESS_FRONTEND_HOST:-127.0.0.1}" --port "${HARNESS_FRONTEND_PORT:-5177}"
