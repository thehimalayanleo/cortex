#!/usr/bin/env bash
# Run Cortex as a login service on macOS: starts at login, restarts if it dies, logs to ~/Library/Logs/cortex.log.
# Usage: scripts/install_service.sh          (install or update)
#        scripts/install_service.sh remove   (stop and remove)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="app.aftersave.cortex"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
UV="$(command -v uv || true)"
[ -n "$UV" ] || { echo "uv not found on PATH; install it first (https://docs.astral.sh/uv/)"; exit 1; }

if [ "${1:-}" = "remove" ]; then
  launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
  rm -f "$PLIST"
  echo "removed $LABEL"
  exit 0
fi

mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>/bin/bash</string><string>$ROOT/run.sh</string>
  </array>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>$(dirname "$UV"):/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    <key>CORTEX_VAULT</key><string>${CORTEX_VAULT:-$HOME/Cortex}</string>
    <key>CORTEX_PORT</key><string>${CORTEX_PORT:-8788}</string>
    <key>CORTEX_SSH_HOST</key><string>${CORTEX_SSH_HOST-ajinkya-5090}</string>
    <key>CORTEX_SSH_PYTHON</key><string>${CORTEX_SSH_PYTHON:-\$HOME/lab-venv/bin/python}</string>
  </dict>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$HOME/Library/Logs/cortex.log</string>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/cortex.log</string>
</dict></plist>
EOF
launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL: Cortex runs at login on http://127.0.0.1:${CORTEX_PORT:-8788}; logs in ~/Library/Logs/cortex.log"
