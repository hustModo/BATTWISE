"""Resolve dataset paths for sessions.

Each session has an ``active_dataset_id``. This module looks the ID up
in (a) user-uploaded dataset registry, (b) BUILTIN_DATASETS fallback.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from app.backend.settings import BUILTIN_DATASETS, settings


@dataclass
class DatasetInfo:
    dataset_id: str
    name: str
    type: str  # first_200s | all_200 | cv_100 | cc_200 | custom
    folder: Path
    model_file: str
    char_file: str
    cell_count: int = 0
    indicator_path: Optional[Path] = None
    indicator_file: Optional[str] = None
    builtin: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "type": self.type,
            "folder": str(self.folder),
            "model_file": self.model_file,
            "char_file": self.char_file,
            "cell_count": self.cell_count,
            "indicator_path": str(self.indicator_path) if self.indicator_path else None,
            "indicator_file": self.indicator_file,
            "builtin": self.builtin,
        }


def _scan_cell_count(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(
        1
        for p in folder.iterdir()
        if p.is_file() and "Cell" in p.name and p.suffix.lower() in {".xls", ".xlsx"}
    )


def _builtin(dataset_id: str) -> Optional[DatasetInfo]:
    entry = BUILTIN_DATASETS.get(dataset_id)
    if not entry:
        return None
    folder = Path(entry["folder"])
    return DatasetInfo(
        dataset_id=dataset_id,
        name=entry["name"],
        type=entry["type"],
        folder=folder,
        model_file=entry["model_file"],
        char_file=entry["char_file"],
        cell_count=_scan_cell_count(folder),
        indicator_path=Path(entry["indicator_path"]) if entry.get("indicator_path") else None,
        indicator_file=entry.get("indicator_file"),
        builtin=True,
    )


def _from_registry(dataset_id: str) -> Optional[DatasetInfo]:
    meta_path = settings.datasets_dir / dataset_id / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    folder = settings.datasets_dir / dataset_id / "cells"
    indicator_path = meta.get("indicator_path")
    indicator_path_obj: Optional[Path] = None
    if indicator_path:
        indicator_path_obj = Path(indicator_path)
    elif meta.get("indicator_file"):
        candidate = settings.datasets_dir / dataset_id / meta["indicator_file"]
        if candidate.exists():
            indicator_path_obj = candidate
    return DatasetInfo(
        dataset_id=dataset_id,
        name=meta.get("name", dataset_id),
        type=meta.get("type", "custom"),
        folder=folder,
        model_file=meta.get("model_file", ""),
        char_file=meta.get("char_file", ""),
        cell_count=meta.get("cell_count", _scan_cell_count(folder)),
        indicator_path=indicator_path_obj,
        indicator_file=meta.get("indicator_file"),
        builtin=False,
    )


def resolve(dataset_id: str) -> DatasetInfo:
    info = _from_registry(dataset_id) or _builtin(dataset_id)
    if info is None:
        raise FileNotFoundError(f"Unknown dataset: {dataset_id}")
    return info


def list_datasets() -> List[DatasetInfo]:
    items: List[DatasetInfo] = []
    for ds_id in BUILTIN_DATASETS.keys():
        info = _builtin(ds_id)
        if info:
            items.append(info)
    if settings.datasets_dir.exists():
        for entry in settings.datasets_dir.iterdir():
            if entry.is_dir() and (entry / "meta.json").exists():
                info = _from_registry(entry.name)
                if info:
                    items.append(info)
    return items
