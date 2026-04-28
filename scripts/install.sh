#!/usr/bin/env bash
# Idempotent quick-install for the Shovs Agent Platform.
# Safe to re-run. For the full interactive setup use ./setup-linux-mac.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

green() { printf "\033[0;32m%s\033[0m\n" "$1"; }
yellow() { printf "\033[1;33m%s\033[0m\n" "$1"; }
red() { printf "\033[0;31m%s\033[0m\n" "$1"; }

green "▶ Shovs Platform install ($(pwd))"

if ! command -v python3 >/dev/null 2>&1; then
    red "✗ python3 not found — install Python 3.10+"; exit 1
fi
if ! command -v node >/dev/null 2>&1; then
    yellow "⚠ node not found — frontend install will be skipped"
fi

if [ ! -d venv ]; then
    green "▶ Creating venv"
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install --quiet --upgrade pip setuptools wheel
green "▶ Installing Python deps (requirements.txt)"
pip install --quiet -r requirements.txt

if command -v node >/dev/null 2>&1; then
    if [ -f package.json ]; then
        green "▶ Installing root npm deps"
        npm install --silent
    fi
    if [ -f frontend_shovs/package.json ]; then
        green "▶ Installing frontend_shovs deps"
        (cd frontend_shovs && npm install --silent)
    fi
fi

if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        yellow "▶ Created .env from .env.example — edit with your API keys"
    else
        cat > .env <<'EOF'
OLLAMA_BASE_URL=http://localhost:11434
PORT=8000
HOST=0.0.0.0
DB_PATH=agents.db
CHROMA_PATH=./chroma_db
TRACE_DIR=./logs
DEFAULT_MODEL=llama3.2
EMBED_MODEL=nomic-embed-text
EOF
        yellow "▶ Created default .env — add provider API keys before running"
    fi
fi

mkdir -p chroma_db logs data

green "✓ Install complete"
echo
echo "Next:"
echo "  python3 scripts/doctor.py     # verify the install"
echo "  npm run dev                   # start backend + frontend"
