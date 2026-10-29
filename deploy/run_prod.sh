#!/usr/bin/env bash
# Single-process production: backend serves the Vite build at /.
# Usage:  deploy/run_prod.sh {build|serve|all}
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ACTION="${1:-all}"

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

do_build() {
    NODE_DIR="$(find_node)"
    if [[ -z "$NODE_DIR" ]]; then
        echo "[prod] ERROR: node not found." >&2
        exit 1
    fi
    export PATH="$NODE_DIR:$PATH"
    cd "$REPO_ROOT/app/frontend"
    if [[ ! -d node_modules ]]; then
        echo "[prod] installing frontend deps…"
        npm ci || npm install
    fi
    echo "[prod] building frontend…"
    npx --no-install tsc --noEmit
    npx --no-install vite build
    echo "[prod] built  →  $REPO_ROOT/app/frontend/dist"
}

do_serve() {
    cd "$REPO_ROOT"
    PORT="${BATRECYCLER_PORT:-8765}"
    HOST="${BATRECYCLER_HOST:-0.0.0.0}"
    WORKERS="${UVICORN_WORKERS:-1}"
    PYTHON="$(find_python)"
    if [[ -z "$PYTHON" ]]; then
        echo "[prod] ERROR: python3 not found. Set BATRECYCLER_PYTHON." >&2
        exit 1
    fi
    echo "[prod] uvicorn on $HOST:$PORT (workers=$WORKERS)"
    exec "$PYTHON" -m uvicorn app.backend.main:app \
        --host "$HOST" --port "$PORT" \
        --workers "$WORKERS" \
        --no-access-log
}

case "$ACTION" in
    build) do_build ;;
    serve) do_serve ;;
    all)   do_build; do_serve ;;
    *)
        echo "usage: $0 {build|serve|all}" >&2
        exit 2
        ;;
esac
