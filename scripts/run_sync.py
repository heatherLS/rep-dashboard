#!/usr/bin/env python3
"""
Drop-in Python replacement for run_sync.sh.
Starts a fresh Teleport proxy, runs the sync, then tears the proxy down.
Called directly by cron via the venv Python so macOS doesn't block it.
"""
import os
import signal
import subprocess
import sys
import time

PROXY_PORT   = 55756
TSH          = "/usr/local/bin/tsh"
SYNC_SCRIPT  = os.path.join(os.path.dirname(__file__), "rep_history_sync.py")
VENV_PYTHON  = sys.executable  # already the venv Python when called from cron

def kill_port(port):
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    except Exception:
        pass

def main():
    # Kill any stale proxy on our port
    kill_port(PROXY_PORT)
    time.sleep(1)

    # Start a fresh Teleport proxy in the background
    proxy = subprocess.Popen(
        [
            TSH, "proxy", "db",
            "--db-user=analytics_read_only",
            "--db-name=dev",
            "--tunnel",
            f"--port={PROXY_PORT}",
            "bi-test",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Give the tunnel time to establish
    time.sleep(6)

    # Run the sync
    result = subprocess.run(
        [VENV_PYTHON, SYNC_SCRIPT, "--today"],
    )

    # Clean up proxy
    try:
        proxy.terminate()
        proxy.wait(timeout=5)
    except Exception:
        try:
            proxy.kill()
        except Exception:
            pass

    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
