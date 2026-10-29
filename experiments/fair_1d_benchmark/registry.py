from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


def _resolve_path_kwargs(
    kwargs: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    resolved = dict(kwargs)
    for key, value in list(resolved.items()):
        if not isinstance(value, str):
            continue
        if key.endswith(("_path", "_file", "_dir", "_json")):
            path = Path(value)
            resolved[key] = path if path.is_absolute() else project_root / path
    return resolved


def load_reducer(spec: dict[str, Any], *, project_root: Path):
    class_path = spec.get("class_path")
    if not isinstance(class_path, str) or ":" not in class_path:
        raise ValueError("Method class_path must use 'module:ClassName' syntax")
    module_name, class_name = class_path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    try:
        reducer_class = getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(f"{class_path} does not exist") from exc
    kwargs = _resolve_path_kwargs(
        dict(spec.get("kwargs") or {}),
        Path(project_root),
    )
    reducer = reducer_class(**kwargs)
    if not hasattr(reducer, "fit_transform") or not hasattr(reducer, "name"):
        raise TypeError(f"{class_path} does not implement Reducer1D")
    return reducer
