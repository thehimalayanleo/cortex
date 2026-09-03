#!/usr/bin/env bash
# Start the Cortex server on :8788 (serves web/dist when built). Vault defaults to ~/Cortex (CORTEX_VAULT to override).
set -euo pipefail
cd "$(dirname "$0")"
export CORTEX_VAULT="${CORTEX_VAULT:-$HOME/Cortex}"
exec uv run --python 3.11 --with fastapi --with 'uvicorn[standard]' --with openai --with pypdf --with pyyaml --with python-multipart \
  python -m uvicorn server.app:app --host 127.0.0.1 --port "${CORTEX_PORT:-8788}" "$@"
