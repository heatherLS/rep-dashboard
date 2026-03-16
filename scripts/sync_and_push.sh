#!/bin/bash
# Hourly sync: pull today's data from Redshift, commit CSV, push to GitHub
# Requires: Teleport proxy running on port 55756, VPN connected

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Check Teleport proxy is running
if ! lsof -ti:55756 > /dev/null 2>&1; then
    echo "$(date): Teleport proxy not running on port 55756 — skipping sync" >> "$SCRIPT_DIR/sync_log.txt"
    exit 0
fi

# Run sync
echo "$(date): Starting sync..." >> "$SCRIPT_DIR/sync_log.txt"
"$PROJECT_DIR/venv/bin/python3" "$SCRIPT_DIR/rep_history_sync.py" --today >> "$SCRIPT_DIR/sync_log.txt" 2>&1

# Commit + push if CSV changed
# Use GIT_DIR + GIT_WORK_TREE so git never needs to call getcwd()
# (macOS blocks getcwd() for cron on Desktop; file I/O still works)
export GIT_DIR="$PROJECT_DIR/.git"
export GIT_WORK_TREE="$PROJECT_DIR"

if git diff --quiet "$PROJECT_DIR/data/rep_history.csv"; then
    echo "$(date): No changes to push" >> "$SCRIPT_DIR/sync_log.txt"
else
    git add "$PROJECT_DIR/data/rep_history.csv"
    git commit -m "Auto-sync: rep history $(date '+%Y-%m-%d %H:%M')"
    git push origin main
    echo "$(date): Pushed updated CSV" >> "$SCRIPT_DIR/sync_log.txt"
fi
