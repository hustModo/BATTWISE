"""Tests for the chunked dataset upload REST API.

Uses FastAPI's TestClient; no real server needed.

Covers:
    1. POST /api/datasets         - create draft
    2. POST /api/datasets/{id}/upload (chunked) - upload cells + model + indicator
    3. POST /api/datasets/{id}/finalize - emits meta.json
    4. GET  /api/datasets         - newly created dataset appears
    5. DELETE /api/datasets/{id}  - removes it
    6. Sanity: cannot delete a built-in dataset
    7. Sanity: filename sanitization rejects ../escape
"""
from __future__ import annotations

import io
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.backend.main import app  # noqa: E402
from app.backend.settings import settings  # noqa: E402


def _chunked(data: bytes, chunk_size: int) -> list[bytes]:
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def _upload(client: TestClient, ds_id: str, *, filename: str, slot: str, content: bytes,
            chunks: int = 1) -> None:
    pieces = _chunked(content, max(1, len(content) // chunks + 1)) if chunks > 1 else [content]
    chunk_count = len(pieces)
    for i, p in enumerate(pieces):
        files = {"file": (filename, io.BytesIO(p), "application/octet-stream")}
        data = {
            "filename": filename,
            "chunk_index": str(i),
            "chunk_count": str(chunk_count),
            "slot": slot,
        }
        r = client.post(f"/api/datasets/{ds_id}/upload", data=data, files=files)
        assert r.status_code == 200, f"chunk {i} failed: {r.status_code} {r.text}"
        body = r.json()
        assert body["filename"] == filename
        assert body["chunk_index"] == i
        if i == chunk_count - 1:
            assert body["completed"] is True
            assert body["stored_path"]
        else:
            assert body["completed"] is False


def run() -> int:
    client = TestClient(app)

    # 1. create draft
    r = client.post("/api/datasets", json={"name": "Test Upload Set", "type": "first_200s"})
    assert r.status_code == 200, r.text
    ds_id = r.json()["dataset_id"]
    assert ds_id, "no dataset_id returned"
    print(f"[1] draft created: {ds_id}")

    # 2. upload 2 cell files (one chunked into 3 chunks)
    _upload(client, ds_id, filename="Cell2.xlsx", slot="cells", content=b"\x00" * 4096, chunks=3)
    _upload(client, ds_id, filename="Cell3.xlsx", slot="cells", content=b"\x00" * 1024, chunks=1)
    print("[2] cell files uploaded (chunked + single)")

    # Upload model + indicator
    _upload(client, ds_id, filename="Model_first_200s.xls", slot="model", content=b"M" * 256, chunks=1)
    _upload(client, ds_id, filename="indicator.xlsx", slot="indicator", content=b"I" * 256, chunks=1)
    print("[3] model + indicator uploaded")

    # 3. finalize
    r = client.post(
        f"/api/datasets/{ds_id}/finalize",
        json={
            "model_file": "Model_first_200s.xls",
            "char_file": "",
            "indicator_file": "indicator.xlsx",
        },
    )
    assert r.status_code == 200, f"finalize failed: {r.status_code} {r.text}"
    info = r.json()
    assert info["dataset_id"] == ds_id
    assert info["cell_count"] == 2, f"unexpected cell_count: {info}"
    assert info["builtin"] is False
    assert info["indicator_file"] == "indicator.xlsx"
    print(f"[4] finalize → {info['name']} / cells={info['cell_count']}")

    # Verify meta.json on disk
    meta_path = settings.datasets_dir / ds_id / "meta.json"
    assert meta_path.exists(), f"meta.json missing: {meta_path}"
    draft_path = settings.datasets_dir / ds_id / "_draft.json"
    assert not draft_path.exists(), "draft.json should be removed after finalize"
    chunks_dir = settings.datasets_dir / ds_id / "_chunks"
    assert not chunks_dir.exists(), "_chunks should be cleaned"
    print("[5] on-disk layout verified")

    # 4. listing includes it
    r = client.get("/api/datasets")
    assert r.status_code == 200, r.text
    ids = [d["dataset_id"] for d in r.json()["datasets"]]
    assert ds_id in ids, f"new dataset not in listing: {ids}"
    print(f"[6] listing includes {ds_id} (total {len(ids)} datasets)")

    # 5. cannot finalize twice
    r = client.post(
        f"/api/datasets/{ds_id}/finalize",
        json={"model_file": "Model_first_200s.xls"},
    )
    assert r.status_code == 409, f"expected 409, got {r.status_code}: {r.text}"
    print("[7] re-finalize correctly rejected")

    # 6. cannot delete a built-in
    r = client.delete("/api/datasets/builtin-first-200s")
    assert r.status_code == 400, r.text
    print("[8] cannot delete built-in dataset")

    # 7. filename sanitization
    r = client.post("/api/datasets", json={"name": "Sanitize Test", "type": "custom"})
    assert r.status_code == 200, r.text
    bad_id = r.json()["dataset_id"]
    files = {"file": ("evil", io.BytesIO(b"x"), "application/octet-stream")}
    data = {
        "filename": "../../etc/passwd",
        "chunk_index": "0",
        "chunk_count": "1",
        "slot": "cells",
    }
    r = client.post(f"/api/datasets/{bad_id}/upload", data=data, files=files)
    assert r.status_code == 200, r.text  # accepted but sanitized
    landed = r.json()["stored_path"]
    assert "etc/passwd" not in landed and ".." not in landed, f"unsafe path landed: {landed}"
    assert (settings.datasets_dir / bad_id / "cells").exists()
    print(f"[9] sanitization OK: landed at {landed}")
    client.delete(f"/api/datasets/{bad_id}")

    # 8. delete the dataset
    r = client.delete(f"/api/datasets/{ds_id}")
    assert r.status_code == 200, r.text
    assert not (settings.datasets_dir / ds_id).exists(), "dataset dir still on disk"
    print(f"[10] deleted {ds_id}")

    # Cleanup any leftover from earlier failed runs
    for stale in settings.datasets_dir.glob("test-upload-set*"):
        shutil.rmtree(stale, ignore_errors=True)
    for stale in settings.datasets_dir.glob("sanitize-test*"):
        shutil.rmtree(stale, ignore_errors=True)

    print("\nAll dataset upload tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
