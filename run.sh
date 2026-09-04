#!/usr/bin/env bash
# Start the Cortex server on :8788 (serves web/dist when built). Vault defaults to ~/Cortex (CORTEX_VAULT to override).
set -euo pipefail
cd "$(dirname "$0")"
export CORTEX_VAULT="${CORTEX_VAULT:-$HOME/Cortex}"
# GPU runs default to the home 5090 over Tailscale SSH; set CORTEX_SSH_HOST="" to disable, or point at another box.
export CORTEX_SSH_HOST="${CORTEX_SSH_HOST-ajinkya-5090}"
export CORTEX_SSH_PYTHON="${CORTEX_SSH_PYTHON:-\$HOME/lab-venv/bin/python}"
# Studio bricks on the box (Celwright): the Wan 2.2 image-to-video script and the conda env that has diffusers
export CINEMA_WAN_BRICK="${CINEMA_WAN_BRICK:-\$HOME/wan_i2v.py}"
export CINEMA_WAN_PYTHON="${CINEMA_WAN_PYTHON:-\$HOME/miniconda3/envs/pgr/bin/python}"
export CINEMA_PROTO="${CINEMA_PROTO:-\$HOME/celwright_v3b/heroset}"
exec uv run --python 3.11 --with fastapi --with 'uvicorn[standard]' --with openai --with pypdf --with pyyaml --with python-multipart \
  python -m uvicorn server.app:app --host 127.0.0.1 --port "${CORTEX_PORT:-8788}" "$@"
