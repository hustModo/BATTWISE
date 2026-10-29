#!/usr/bin/env bash
# Dev mode: backend with --reload on :8765 OR frontend with vite on :5173.
# Usage:  deploy/run_dev.sh backend | frontend
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
TARGET="${1:-backend}"

find_python() {
    if [[ -n "${BATRECYCLER_PYTHON:-}" && -x "${BATRECYCLER_PYTHON}" ]]; then
        echo "${BATRECYCLER_PYTHON}"; return
    fi
    if [[ -x "$HOME/miniconda3/bin/python3" ]]; then
        echo "$HOME/miniconda3/bin/python3"; return
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3; return
    fi
    echo ""
}

# Pick a node we can find — explicit env wins, else $HOME/.local, else $PATH.
find_node() {
    if [[ -n "${NODE_BIN:-}" && -x "${NODE_BIN}/node" ]]; then
        echo "${NODE_BIN}"; return
    fi
    if [[ -x "$HOME/.local/node-v20.18.2-linux-x64/bin/node" ]]; then
        echo "$HOME/.local/node-v20.18.2-linux-x64/bin"; return
    fi
    if command -v node >/dev/null 2>&1; then
        dirname "$(command -v node)"; return
    fi
    echo ""
}

case "$TARGET" in
    backend)
        cd "$REPO_ROOT"
        PORT="${BATRECYCLER_PORT:-8765}"
        HOST="${BATRECYCLER_HOST:-127.0.0.1}"
        PYTHON="$(find_python)"
        if [[ -z "$PYTHON" ]]; then
            echo "[dev] ERROR: python3 not found. Set BATRECYCLER_PYTHON." >&2
            exit 1
        fi

        # Auto-clean any leftover process holding :$PORT. The usual culprit
        # is a previous `uvicorn --reload` worker OR a multiprocessing.spawn
        # child from ComputeAgent that out-lived its parent.
        # If the port is already free we skip the cleanup entirely (silent).
        if ss -ltn 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p {found=1} END {exit !found}'; then
            echo "[dev] :$PORT is busy, running kill_backend.sh first…"
            BATRECYCLER_PORT="$PORT" "$HERE/kill_backend.sh" || {
                echo "[dev] could not free :$PORT — falling through to uvicorn anyway."
            }
        fi

        # ─────────────────────────────────────────────────────────────
        # NOTE on why we DO NOT use `uvicorn --reload` here.
        #
        # The agents in this project legitimately write Python files to
        # disk while running:
        #   * DesignerAgent → Evo_Textgrad_*/problems/battery_mmd/gpt.py
        #   * ComputeAgent  → dianchi03/mmd_tools_new.py
        #   * EOH runner    → Evo_Textgrad_*/outputs/.../*.py
        #
        # uvicorn 0.32 + watchfiles 1.2 has a hard-to-disable behaviour
        # (see WatchFilesReload.__init__): any `--reload-dir` that is a
        # *subdirectory* of CWD is silently DROPPED, and the watcher is
        # forced back to CWD = repo root. That means our `--reload-dir
        # app/backend` flag was a no-op and every Designer/Compute run
        # killed the backend mid-flight ("点下一步又改回去了" symptom).
        #
        # Removing --reload entirely makes the dev backend identical to
        # production behaviour: it stays up across agent runs, and the
        # developer restarts manually after editing backend code:
        #
        #     deploy/kill_backend.sh && deploy/run_dev.sh backend
        #
        # If you want auto-reload back, run uvicorn manually with
        # `--reload-dir $PWD/app/backend` from a separate CWD outside the
        # repo, OR install entr/watchexec scoped to app/backend.
        # ─────────────────────────────────────────────────────────────
        if [[ "${BATRECYCLER_RELOAD:-0}" = "1" ]]; then
            echo "[dev] backend on ${HOST}:$PORT (--reload enabled via BATRECYCLER_RELOAD=1; WILL restart on every agent write — not recommended)"
            exec "$PYTHON" -m uvicorn app.backend.main:app \
                --reload \
                --reload-dir app/backend \
                --reload-exclude 'app/backend/workspace/*' \
                --host "$HOST" --port "$PORT"
        else
            echo "[dev] backend on ${HOST}:$PORT (reload OFF — restart manually after editing backend code)"
            exec "$PYTHON" -m uvicorn app.backend.main:app \
                --host "$HOST" --port "$PORT"
        fi
        ;;
    frontend)
        NODE_DIR="$(find_node)"
        if [[ -z "$NODE_DIR" ]]; then
            echo "[dev] ERROR: node not found. Set NODE_BIN=... or install node 20+." >&2
            exit 1
        fi
        export PATH="$NODE_DIR:$PATH"
        cd "$REPO_ROOT/app/frontend"
        FRONTEND_HOST="${BATRECYCLER_FRONTEND_HOST:-127.0.0.1}"
        FRONTEND_PORT="${BATRECYCLER_FRONTEND_PORT:-5173}"
        if [[ ! -d node_modules ]]; then
            echo "[dev] installing frontend deps…"
            npm install
        fi
        echo "[dev] frontend on ${FRONTEND_HOST}:$FRONTEND_PORT (proxies /api+/ws to :8765)"
        exec npx vite --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"
        ;;
    *)
        echo "usage: $0 {backend|frontend}" >&2
        exit 2
        ;;
esac
