"""FastAPI entrypoint for the BATTWISE backend."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.backend.api import rest, ws
from app.backend.services import llm_client
from app.backend.settings import REPO_ROOT, ensure_mmd_tools_healthy, settings


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
)
logger = logging.getLogger("battwise")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s (debug=%s)", settings.app_name, settings.debug)
    logger.info("Workspace: %s", settings.workspace_dir)
    # Restore dianchi03/mmd_tools_new.py from baseline if it's been polluted
    # by an aborted timeout test or a previous bad LLM injection. Without
    # this, the user would see Compute hang forever every time the backend
    # starts (because the worker subprocess loads whatever stub is on disk).
    try:
        ensure_mmd_tools_healthy()
        logger.info("mmd_tools_new.py health-check passed (or restored).")
    except Exception as exc:  # noqa: BLE001
        logger.warning("mmd_tools_new.py health-check raised: %s", exc)
    # One-shot migration: if the saved LLM settings still point at a model
    # alias Alibaba is sunsetting on 2026-05-30 (`*-latest`, `qwen-turbo*`,
    # `qwen-flash`, `qwen3-coder-plus`, `qwen-max-latest`), rewrite to
    # PINNED_MODEL so the user doesn't get blindsided by 404s post-cutover.
    try:
        status = llm_client.ensure_model_pinned()
        if status.get("migrated"):
            logger.info(
                "LLM model migrated from %r → %r (deprecated alias).",
                status.get("from"), status.get("to"),
            )
        else:
            logger.info("LLM model pinned check OK (current=%r).", status.get("current"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM model pin check raised: %s", exc)
    # Surface how many on-disk session files we currently have. Useful
    # when a user complains "my conversation disappeared" — if this
    # counter is 0 the session dir got wiped; if it's high, we can hint
    # at cleanup. Doesn't load them into memory (sessions stay lazily
    # loaded via store.get()).
    try:
        sessions_dir = settings.sessions_dir
        if sessions_dir.exists():
            n_sessions = sum(
                1 for p in sessions_dir.iterdir()
                if p.is_dir() and (p / "session.json").exists()
            )
            logger.info("Persisted sessions on disk: %d", n_sessions)
    except Exception:  # noqa: BLE001
        pass
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rest.router)
app.include_router(ws.router)


# ---------- production: serve built frontend ----------
#
# When BATRECYCLER_SERVE_FRONTEND is set (or app/frontend/dist exists), mount the
# Vite build under "/" and fall back to index.html for client-side routes.
# Development workflow remains unchanged: run `vite` separately and let it proxy
# /api and /ws to this backend.

_FRONTEND_DIST_ENV = os.environ.get("BATRECYCLER_FRONTEND_DIST", "")
_FRONTEND_DIST = (
    Path(_FRONTEND_DIST_ENV)
    if _FRONTEND_DIST_ENV
    else REPO_ROOT / "app" / "frontend" / "dist"
)


@app.get("/")
async def root_or_spa():
    index = _FRONTEND_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"ok": True, "name": settings.app_name}


if _FRONTEND_DIST.exists() and (_FRONTEND_DIST / "index.html").exists():
    logger.info("Mounting frontend SPA from %s", _FRONTEND_DIST)
    app.mount(
        "/assets",
        StaticFiles(directory=_FRONTEND_DIST / "assets", check_dir=False),
        name="assets",
    )

    @app.get("/{spa_path:path}")
    async def spa_fallback(spa_path: str):
        # Avoid hijacking API/WS prefixes — FastAPI matches more specific routes first.
        candidate = _FRONTEND_DIST / spa_path
        if spa_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_FRONTEND_DIST / "index.html")
