#!/usr/bin/env bash
# Launches the Flask backend that wraps the gesture agent.
# Usage:
#   ./web/run.sh                          # http://127.0.0.1:5050
#   WEB_PORT=8080 ./web/run.sh
#   AGENT_CONFIG_PATH=agent_config.json ./web/run.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export WEB_HOST="${WEB_HOST:-127.0.0.1}"
export WEB_PORT="${WEB_PORT:-5050}"

if command -v uv >/dev/null 2>&1; then
  exec uv run --with flask --with flask-cors python web/backend/app.py
else
  exec python web/backend/app.py
fi
