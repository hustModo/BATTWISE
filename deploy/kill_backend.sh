#!/usr/bin/env bash
# Kill every BATTWISE backend process on a chosen port.
#
# Why we need this:
#   - `pkill -f "uvicorn app.backend.main"` only catches the uvicorn supervisor.
#   - `uvicorn --reload` forks a worker process whose argv may not contain
#     the original pattern (it's exec'd through multiprocessing).
#   - ComputeAgent now spawns multiprocessing.Process('spawn') children to
#     sandbox the LLM-generated MMD function. Those children are pure
#     `python -c "from multiprocessing.spawn import ..."` and DO NOT
#     contain the string "uvicorn" or "app.backend" anywhere.
#
# This script tries 3 strategies, in order from gentlest to nuclear:
#   1. fuser -k 8765/tcp   — kills whoever holds the port (most reliable)
#   2. pkill on every backend-related pattern we know about
#   3. waits up to 5s for the port to actually be free
#
# Usage:
#   deploy/kill_backend.sh              # defaults to port 8765
#   deploy/kill_backend.sh 9000         # custom port
#   BATRECYCLER_PORT=9000 deploy/kill_backend.sh
set -uo pipefail

PORT="${1:-${BATRECYCLER_PORT:-8765}}"

echo "[kill] freeing backend on :$PORT …"

# --- Strategy 1: kill by port (works without knowing the cmd) -----------------
if command -v fuser >/dev/null 2>&1; then
    fuser -k -9 "${PORT}/tcp" 2>/dev/null && echo "[kill]  fuser killed listener on :$PORT" || true
fi

# Fallback / extra: ss + lsof for older systems where fuser is missing.
if command -v ss >/dev/null 2>&1; then
    PIDS=$(ss -ltnp 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p {print $0}' | grep -oP 'pid=\K\d+' | sort -u)
    for pid in $PIDS; do
        kill -9 "$pid" 2>/dev/null && echo "[kill]  ss-found pid=$pid -> killed" || true
    done
fi
if command -v lsof >/dev/null 2>&1; then
    PIDS=$(lsof -ti :"$PORT" 2>/dev/null || true)
    for pid in $PIDS; do
        kill -9 "$pid" 2>/dev/null && echo "[kill]  lsof-found pid=$pid -> killed" || true
    done
fi

# --- Strategy 2: kill by command pattern --------------------------------------
# We pkill in two passes:
#   (a) the supervisor / reloader process
#   (b) any uvicorn worker AND any multiprocessing.spawn child belonging to us
PATTERNS=(
    'uvicorn app.backend.main'
    'uvicorn.*--reload'
    'app\.backend\.main'
    'app\.backend\.tools\.compute_worker' # standalone compute subprocess
    'compute_worker'                       # belt+braces
    'multiprocessing\.spawn'           # legacy fallback (older compute_agent versions)
    'multiprocessing\.forkserver'      # forkserver helper process
    'multiprocessing\.resource_tracker' # spawn/fork helper
)
for pat in "${PATTERNS[@]}"; do
    if pkill -9 -f "$pat" 2>/dev/null; then
        echo "[kill]  pkill -f '$pat' -> killed"
    fi
done

# --- Strategy 3: confirm the port is actually free ----------------------------
for i in 1 2 3 4 5; do
    sleep 0.4
    if ! ss -ltn 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p {found=1} END {exit !found}'; then
        echo "[kill] ✓ :$PORT is now free."
        exit 0
    fi
    echo "[kill]  …waiting ($i/5)…"
done

echo "[kill] ✗ :$PORT is STILL held by someone. Try:"
echo "         sudo lsof -i :$PORT"
echo "         sudo fuser -k -9 ${PORT}/tcp"
exit 1
