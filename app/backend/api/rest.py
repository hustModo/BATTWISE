"""REST API: sessions, messages, actions, datasets, artifacts, settings."""
from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.backend.orchestrator import pipeline
from app.backend.orchestrator.events import bus
from app.backend.orchestrator.session_store import store
from app.backend.services import data_resolver, llm_client
from app.backend.services import settings_store as cfg_store
from app.backend.services.settings_store import LLMConfig
from app.backend.settings import BUILTIN_DATASETS, settings


router = APIRouter(prefix="/api", tags=["battwise"])


# ---------- request bodies ----------

class MessageBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)


class ActionBody(BaseModel):
    type: str
    cluster: Optional[int] = None
    clusters: Optional[List[int]] = None
    mode: Optional[str] = None
    dataset_id: Optional[str] = None
    action: Optional[str] = None
    step_id: Optional[int] = None
    params: Optional[Dict[str, Any]] = None


class LLMSettingsBody(BaseModel):
    model: str
    base_url: str = ""
    api_key: Optional[str] = None  # None = leave unchanged; "" = clear
    temperature: float = 1.0


class LLMTestBody(BaseModel):
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None


# ---------- session ----------

@router.post("/session")
async def create_session() -> Dict[str, Any]:
    session = await store.create()
    return {"session_id": session.session_id, "state": session.to_dict()}


@router.get("/session/{session_id}/state")
async def session_state(session_id: str) -> Dict[str, Any]:
    session = await store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return session.to_dict()


@router.get("/session/{session_id}/events")
async def session_events(session_id: str) -> Dict[str, Any]:
    session = await store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return {"events": bus.history(session_id)}


@router.post("/session/{session_id}/message")
async def post_message(
    session_id: str, body: MessageBody, background: BackgroundTasks
) -> Dict[str, Any]:
    session = await store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    background.add_task(_safe_handle_message, session_id, body.text)
    return {"accepted": True}


async def _safe_handle_message(session_id: str, text: str) -> None:
    session = await store.get(session_id)
    if session is None:
        return
    try:
        await pipeline.handle_message(session, text)
    except Exception as exc:  # surface as event log
        from app.backend.orchestrator.events import AgentEvent

        await bus.publish(
            AgentEvent(
                kind="log",
                session_id=session_id,
                payload={"level": "error", "text": f"pipeline error: {exc}"},
            )
        )


@router.post("/session/{session_id}/action")
async def post_action(session_id: str, body: ActionBody) -> Dict[str, Any]:
    session = await store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    payload = body.model_dump(exclude_none=True)
    return await pipeline.handle_action(session, payload)


# ---------- datasets ----------

@router.get("/datasets")
async def list_datasets() -> Dict[str, Any]:
    return {"datasets": [d.to_dict() for d in data_resolver.list_datasets()]}


@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str) -> Dict[str, Any]:
    try:
        return data_resolver.resolve(dataset_id).to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ---------- dataset upload (chunked) ----------

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9-]")


class DatasetCreateBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    type: str = Field("custom", min_length=1, max_length=40)
    dataset_id: Optional[str] = Field(None, max_length=64)


class DatasetFinalizeBody(BaseModel):
    model_file: str = Field("", max_length=200)
    char_file: str = Field("", max_length=200)
    indicator_file: Optional[str] = Field(None, max_length=200)


def _slugify(text: str, max_len: int = 32) -> str:
    text = text.strip().lower().replace(" ", "-")
    text = _SAFE_ID_RE.sub("", text)
    return (text or "ds")[:max_len].strip("-") or "ds"


def _safe_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="filename required")
    name = name.replace("\\", "/").split("/")[-1]
    name = _SAFE_NAME_RE.sub("_", name)
    if name in {".", ".."} or name.startswith("."):
        raise HTTPException(status_code=400, detail=f"invalid filename: {name!r}")
    return name


def _dataset_root(dataset_id: str, *, must_exist: bool = True) -> Path:
    if dataset_id in BUILTIN_DATASETS:
        raise HTTPException(status_code=400, detail="cannot modify a built-in dataset")
    dataset_id = _slugify(dataset_id, max_len=64)
    if not dataset_id:
        raise HTTPException(status_code=400, detail="invalid dataset id")
    root = settings.datasets_dir / dataset_id
    if must_exist and not root.exists():
        raise HTTPException(status_code=404, detail=f"dataset draft not found: {dataset_id}")
    return root


@router.post("/datasets")
async def create_dataset_draft(body: DatasetCreateBody) -> Dict[str, Any]:
    """Create a draft dataset directory and return its dataset_id.

    Subsequent ``/datasets/{id}/upload`` calls populate the directory; a final
    ``/datasets/{id}/finalize`` writes meta.json so it shows up in the
    listing.
    """
    ds_id = _slugify(body.dataset_id or body.name)
    base = settings.datasets_dir / ds_id
    if base.exists():
        # avoid clobbering an existing draft / finalized dataset
        suffix = 2
        while (settings.datasets_dir / f"{ds_id}-{suffix}").exists():
            suffix += 1
        ds_id = f"{ds_id}-{suffix}"
        base = settings.datasets_dir / ds_id
    (base / "cells").mkdir(parents=True, exist_ok=True)
    (base / "_chunks").mkdir(parents=True, exist_ok=True)
    draft = {
        "dataset_id": ds_id,
        "name": body.name,
        "type": body.type or "custom",
        "draft": True,
    }
    (base / "_draft.json").write_text(
        json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"dataset_id": ds_id, "name": body.name, "type": draft["type"]}


@router.post("/datasets/{dataset_id}/upload")
async def upload_dataset_chunk(
    dataset_id: str,
    file: UploadFile = File(...),
    filename: str = Form(...),
    chunk_index: int = Form(0),
    chunk_count: int = Form(1),
    slot: str = Form("cells"),  # "cells" | "model" | "char" | "indicator"
) -> Dict[str, Any]:
    """Append one chunk of a file to the dataset draft.

    The file is assembled in ``_chunks/<safe_name>.part`` and atomically moved
    into ``cells/<safe_name>`` (or the root, for model/char/indicator slots)
    once the last chunk is received.
    """
    if chunk_count < 1 or chunk_index < 0 or chunk_index >= chunk_count:
        raise HTTPException(status_code=400, detail="invalid chunk_index / chunk_count")
    if slot not in {"cells", "model", "char", "indicator"}:
        raise HTTPException(status_code=400, detail=f"invalid slot: {slot!r}")
    root = _dataset_root(dataset_id, must_exist=True)
    safe = _safe_name(filename)

    max_bytes = settings.max_upload_mb * 1024 * 1024
    chunks_dir = root / "_chunks"
    chunks_dir.mkdir(exist_ok=True)
    part = chunks_dir / f"{safe}.part"

    mode = "wb" if chunk_index == 0 else "ab"
    written = 0
    aborted = False
    with part.open(mode) as f:
        while True:
            buf = await file.read(1024 * 1024)
            if not buf:
                break
            f.write(buf)
            written += len(buf)
            if part.stat().st_size > max_bytes:
                aborted = True
                break
    await file.close()
    if aborted:
        part.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"upload exceeds max_upload_mb={settings.max_upload_mb}",
        )

    completed = chunk_index == chunk_count - 1
    final_path: Optional[Path] = None
    if completed:
        # Move into final slot
        if slot == "cells":
            target_dir = root / "cells"
        else:
            target_dir = root
        target_dir.mkdir(exist_ok=True)
        target = target_dir / safe
        if target.exists():
            target.unlink()
        part.rename(target)
        final_path = target

    return {
        "dataset_id": dataset_id,
        "filename": safe,
        "slot": slot,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "received": written,
        "completed": completed,
        "stored_path": str(final_path) if final_path else None,
    }


@router.post("/datasets/{dataset_id}/finalize")
async def finalize_dataset(dataset_id: str, body: DatasetFinalizeBody) -> Dict[str, Any]:
    root = _dataset_root(dataset_id, must_exist=True)
    draft_path = root / "_draft.json"
    if not draft_path.exists():
        raise HTTPException(status_code=409, detail="no draft to finalize (already finalized?)")

    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    cells_dir = root / "cells"
    cell_files = (
        [p.name for p in cells_dir.iterdir() if p.is_file()]
        if cells_dir.exists()
        else []
    )
    if not cell_files:
        raise HTTPException(status_code=400, detail="no Cell files uploaded")

    # Validate slots
    for slot_name, slot_path in [
        ("model_file", body.model_file),
        ("char_file", body.char_file),
        ("indicator_file", body.indicator_file),
    ]:
        if slot_path and not (root / slot_path).exists():
            raise HTTPException(
                status_code=400, detail=f"{slot_name} {slot_path!r} not found in upload"
            )

    meta: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "name": draft.get("name", dataset_id),
        "type": draft.get("type", "custom"),
        "model_file": body.model_file or "",
        "char_file": body.char_file or "",
        "cell_count": sum(1 for n in cell_files if "Cell" in n),
        "builtin": False,
    }
    if body.indicator_file:
        meta["indicator_file"] = body.indicator_file
        meta["indicator_path"] = str(root / body.indicator_file)

    (root / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    draft_path.unlink(missing_ok=True)
    # Clean up any leftover chunks
    chunks_dir = root / "_chunks"
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir, ignore_errors=True)

    info = data_resolver.resolve(dataset_id)
    return info.to_dict()


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str) -> Dict[str, Any]:
    if dataset_id in BUILTIN_DATASETS:
        raise HTTPException(status_code=400, detail="cannot delete a built-in dataset")
    root = settings.datasets_dir / _slugify(dataset_id, max_len=64)
    if not root.exists():
        raise HTTPException(status_code=404, detail="dataset not found")
    shutil.rmtree(root, ignore_errors=True)
    return {"deleted": dataset_id}


# ---------- artifacts ----------

@router.get("/artifact/{session_id}/{name}")
async def get_artifact(session_id: str, name: str) -> FileResponse:
    session = await store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    for art in session.artifacts:
        if art["name"] == name:
            path = Path(art["path"])
            if not path.exists():
                raise HTTPException(status_code=410, detail="artifact missing on disk")
            filename = "cddi_function.py" if name in {"cddi_function.py", "gpt.py"} else path.name
            return FileResponse(path, filename=filename)
    raise HTTPException(status_code=404, detail="artifact not found")


# ---------- settings ----------

@router.get("/settings/llm")
async def get_llm_settings() -> Dict[str, Any]:
    panel = cfg_store.load()
    if panel:
        data = panel.masked()
        data["source"] = "settings"
        return data
    if settings.llm_api_key:
        return {
            "model": settings.llm_model,
            "base_url": settings.llm_base_url,
            "temperature": settings.llm_temperature,
            "api_key_masked": "****",
            "has_key": True,
            "source": "env",
        }
    if llm_client.has_credentials():
        cfg = llm_client.resolve_config()
        masked = f"{cfg.api_key[:4]}****{cfg.api_key[-4:]}" if len(cfg.api_key) >= 8 else "****"
        return {
            "model": cfg.model,
            "base_url": cfg.base_url,
            "temperature": cfg.temperature,
            "api_key_masked": masked,
            "has_key": True,
            "source": cfg.source,
        }
    return {"model": "", "base_url": "", "temperature": 1.0, "has_key": False, "source": "none"}


@router.put("/settings/llm")
async def put_llm_settings(body: LLMSettingsBody) -> Dict[str, Any]:
    current = cfg_store.load() or LLMConfig()
    key = body.api_key
    if key is None:
        key = current.api_key
    elif key == "":
        key = ""
    cfg = LLMConfig(
        model=body.model or current.model,
        base_url=body.base_url or current.base_url,
        api_key=key,
        temperature=body.temperature,
    )
    if not cfg.api_key:
        cfg_store.clear()
        return {"saved": True, "cleared": True}
    cfg_store.save(cfg)
    return {"saved": True}


@router.post("/settings/llm/test")
async def post_llm_test(body: LLMTestBody) -> Dict[str, Any]:
    return llm_client.ping(model=body.model, base_url=body.base_url, api_key=body.api_key)


class LLMProbeBody(BaseModel):
    candidates: Optional[List[str]] = None
    timeout: float = 12.0
    save_fastest: bool = False  # if True, write the winner back to settings


@router.post("/settings/llm/probe")
async def post_llm_probe(body: LLMProbeBody) -> Dict[str, Any]:
    """Latency-probe a list of candidate models against the configured base_url.

    Returns ``{results: [...], fastest, candidates_tested}`` sorted ok-first
    then by latency. When ``save_fastest=true`` and at least one model
    responded successfully, the encrypted settings store is updated so the
    rest of the app immediately starts using the winner.
    """
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        None,
        lambda: llm_client.probe_models(
            candidates=body.candidates,
            timeout=max(2.0, min(body.timeout, 60.0)),
        ),
    )
    fastest = next((r for r in results if r.get("ok")), None)
    saved = False
    if body.save_fastest and fastest:
        # Reuse the resolved (base_url, api_key) — only the model name is changed.
        try:
            cfg = llm_client.resolve_config()
            new_cfg = LLMConfig(
                model=fastest["model"],
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                temperature=cfg.temperature,
            )
            cfg_store.save(new_cfg)
            saved = True
        except Exception:
            saved = False
    return {
        "results": results,
        "fastest": fastest,
        "candidates_tested": [r.get("model") for r in results],
        "saved_fastest": saved,
    }


# ---------- meta ----------

@router.get("/health")
async def health() -> Dict[str, Any]:
    datasets = [d.dataset_id for d in data_resolver.list_datasets()]
    return {
        "ok": True,
        "app": settings.app_name,
        "llm_configured": llm_client.has_credentials(),
        "datasets": datasets,
    }
