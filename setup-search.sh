#!/bin/bash

# setup-search.sh — Proper SearXNG initialization

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SEARX_DIR="$DIR/searxng"

echo "--- Initializing SearXNG ---"

# Generate a random secret key
SECRET=$(openssl rand -hex 32)
sed -i '' "s/REPLACE_ME_WITH_RANDOM_HEX/$SECRET/" "$SEARX_DIR/data/settings.yml" || \
sed -i "s/REPLACE_ME_WITH_RANDOM_HEX/$SECRET/" "$SEARX_DIR/data/settings.yml"

echo "Starting SearXNG via Docker Compose..."
cd "$SEARX_DIR"
docker-compose up -d

echo "Done! SearXNG is running at http://localhost:8080"
echo "Export this to use it:"
echo "export SEARXNG_URL=http://localhost:8080"
