#!/bin/bash
# Wrapper: start Teleport proxy, run sync, clean up.
# Used by the cron job so the proxy is always fresh each run.

PROXY_PORT=55756
TSH="/usr/local/bin/tsh"
VENV_PYTHON="/Users/heatherpainter/Desktop/rep_dashboard_project/venv/bin/python3"
SYNC_SCRIPT="/Users/heatherpainter/Desktop/rep_dashboard_project/scripts/rep_history_sync.py"

# Kill any stale proxy on our port
lsof -ti:$PROXY_PORT | xargs kill -9 2>/dev/null || true
sleep 1

# Start a fresh proxy in the background
"$TSH" proxy db \
  --db-user=analytics_read_only \
  --db-name=dev \
  --tunnel \
  --port $PROXY_PORT \
  bi-test > /dev/null 2>&1 &
PROXY_PID=$!

# Give it time to establish the tunnel
sleep 6

# Run the sync
"$VENV_PYTHON" "$SYNC_SCRIPT" --today

# Clean up proxy
kill $PROXY_PID 2>/dev/null || true
