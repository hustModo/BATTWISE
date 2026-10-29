"""Function-ised version of dianchi03/main.py.

Two entry points:
    - inject_mmd_code(code) -> rewrites dianchi03/mmd_tools_new.py so that
      MMD_cal_new is the LLM-designed function (with backup .bak).
    - compute_mmd_for_folder(...) -> a generator yielding incremental progress
      events plus a final summary, ready to be wrapped by an agent.

Original `dianchi03/main.py` and `mmd_tools_new.py` are not modified at import
time; we only rewrite mmd_tools_new.py when the user requests an injection,
and we always back the file up first.
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from app.backend.services.mmd_designer import _load_data, find_function_name
from app.backend.settings import DIANCHI_DIR, DIANCHI_MMD_TOOLS_NEW

logger = logging.getLogger(__name__)


_MMD_CAL_NEW_NAME = "MMD_cal_new"


# ---------- code injection ----------

def _rename_function(code: str, new_name: str) -> str:
    """Rename `calculate_mmd_vN` to ``new_name`` inside ``code``."""
    fn_name = find_function_name(code)
    if not fn_name:
        raise RuntimeError("No calculate_mmd_vN function found in code; cannot inject.")
    pattern = re.compile(r"\b" + re.escape(fn_name) + r"\b")
    return pattern.sub(new_name, code)


def inject_mmd_code(code: str, *, target_path: Path = DIANCHI_MMD_TOOLS_NEW) -> Path:
    """Rewrite ``dianchi03/mmd_tools_new.py`` with the new MMD_cal_new function.

    A `.bak` backup of the existing file is always created before overwrite.
    Returns the path that was written.
    """
    renamed = _rename_function(code, _MMD_CAL_NEW_NAME)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        backup = target_path.with_suffix(target_path.suffix + ".bak")
        shutil.copy2(target_path, backup)

    header = (
        "# ============================================================================\n"
        "# MMD_cal_new generated/injected by the BATTWISE Metric Designer.\n"
        "# Source LLM call timestamp: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n"
        "# Original baseline kept under mmd_tools_new.py.bak (most recent backup only).\n"
        "# ============================================================================\n"
    )
    target_path.write_text(header + "\n" + renamed.rstrip() + "\n", encoding="utf-8")
    return target_path


def load_injected_function(*, target_path: Path = DIANCHI_MMD_TOOLS_NEW) -> Callable:
    """Force-reload mmd_tools_new from disk and return MMD_cal_new."""
    module_name = "_batrecycler_mmd_tools_new_" + str(int(time.time() * 1000))
    spec = importlib.util.spec_from_file_location(module_name, target_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {target_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    fn = getattr(module, _MMD_CAL_NEW_NAME, None)
    if not callable(fn):
        raise RuntimeError(f"{target_path} did not expose {_MMD_CAL_NEW_NAME}")
    return fn


# ---------- helpers ----------

def _extract_cell_number(file_path: Path) -> int:
    m = re.search(r"Cell(\d+)", file_path.name)
    return int(m.group(1)) if m else int(1e9)


def _slice_cv(df: pd.DataFrame, row_start: int = 0, row_end: Optional[int] = None) -> np.ndarray:
    if "Current" not in df.columns or "Voltage" not in df.columns:
        raise KeyError("Cell/Model file must contain 'Current' and 'Voltage' columns.")
    if row_end is not None:
        return df[["Current", "Voltage"]].iloc[row_start:row_end].values
    return df[["Current", "Voltage"]].iloc[row_start:].values


def _list_cell_files(folder: Path) -> List[Path]:
    cells: List[Path] = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if "Cell" not in p.name:
            continue
        if "Model" in p.name:
            continue
        if p.suffix.lower() not in {".xls", ".xlsx", ".csv"}:
            continue
        cells.append(p)
    cells.sort(key=_extract_cell_number)
    return cells


# ---------- progress events ----------

@dataclass
class MMDProgress:
    """One progress tick during compute_mmd_for_folder()."""

    phase: str  # "start" | "model_loaded" | "cell" | "cell_skipped" | "saved" | "correlation" | "done"
    index: int = 0
    total: int = 0
    cell_num: Optional[int] = None
    filename: Optional[str] = None
    mmd_loss: Optional[float] = None
    sample_count: Optional[int] = None
    error: Optional[str] = None
    elapsed_s: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------- core compute ----------

def compute_mmd_for_folder(
    *,
    data_folder: Path,
    model_filename: str,
    mmd_fn: Callable,
    row_start: int = 0,
    row_end: Optional[int] = None,
    scale: float = 1.0,
    fix_sigma: float = 0,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    cells_filter: Optional[List[int]] = None,
) -> Iterator[MMDProgress]:
    """Generator yielding `MMDProgress` events as MMD is computed for each cell.

    Args:
        cells_filter: when provided, only Cells whose number is in this list are
            computed (used by RefinementAgent for sub-cluster MMD).

    The very last event has phase="done" and ``extra={"results": [list of dicts]}``.
    """
    start = time.time()
    folder = Path(data_folder)
    model_path = folder / model_filename
    wanted: Optional[set] = set(int(c) for c in cells_filter) if cells_filter else None

    yield MMDProgress(phase="start", total=0, filename=str(folder), elapsed_s=0.0)

    df_model = _load_data(model_path)
    if df_model is None:
        yield MMDProgress(
            phase="done",
            error=f"Cannot read Model file: {model_path}",
            elapsed_s=time.time() - start,
            extra={"results": []},
        )
        return

    try:
        xt = _slice_cv(df_model, row_start, row_end)
    except KeyError as exc:
        yield MMDProgress(
            phase="done",
            error=f"Model file column error: {exc}",
            elapsed_s=time.time() - start,
            extra={"results": []},
        )
        return

    yield MMDProgress(
        phase="model_loaded",
        filename=model_filename,
        sample_count=int(xt.shape[0]),
        elapsed_s=time.time() - start,
    )

    cells = _list_cell_files(folder)
    if wanted is not None:
        cells = [p for p in cells if _extract_cell_number(p) in wanted]
    total = len(cells)
    results: List[Dict[str, Any]] = []

    for idx, cell_path in enumerate(cells, start=1):
        df_cell = _load_data(cell_path)
        cell_num = _extract_cell_number(cell_path)
        if df_cell is None:
            yield MMDProgress(
                phase="cell_skipped",
                index=idx,
                total=total,
                cell_num=cell_num,
                filename=cell_path.name,
                error="failed to load file",
                elapsed_s=time.time() - start,
            )
            continue
        try:
            xs = _slice_cv(df_cell, row_start, row_end)
        except KeyError as exc:
            yield MMDProgress(
                phase="cell_skipped",
                index=idx,
                total=total,
                cell_num=cell_num,
                filename=cell_path.name,
                error=str(exc),
                elapsed_s=time.time() - start,
            )
            continue

        try:
            raw_loss = float(mmd_fn(xs, xt, fix_sigma, kernel_mul, kernel_num))
        except Exception:
            tb = traceback.format_exc(limit=4)
            yield MMDProgress(
                phase="cell_skipped",
                index=idx,
                total=total,
                cell_num=cell_num,
                filename=cell_path.name,
                error=tb.splitlines()[-1] if tb else "exception",
                elapsed_s=time.time() - start,
            )
            continue

        if not np.isfinite(raw_loss):
            yield MMDProgress(
                phase="cell_skipped",
                index=idx,
                total=total,
                cell_num=cell_num,
                filename=cell_path.name,
                error=f"non-finite mmd ({raw_loss!r})",
                elapsed_s=time.time() - start,
            )
            continue

        loss_scaled = raw_loss * scale
        results.append(
            {
                "Filename": cell_path.name,
                "Cell_num": cell_num,
                "MMD_Loss": loss_scaled,
                "Sample_count": int(xs.shape[0]),
            }
        )
        yield MMDProgress(
            phase="cell",
            index=idx,
            total=total,
            cell_num=cell_num,
            filename=cell_path.name,
            mmd_loss=loss_scaled,
            sample_count=int(xs.shape[0]),
            elapsed_s=time.time() - start,
        )

    yield MMDProgress(
        phase="done",
        index=len(results),
        total=total,
        elapsed_s=time.time() - start,
        extra={"results": results},
    )


def save_mmd_csv(results: List[Dict[str, Any]], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results).rename(columns={"MMD_Loss": "CDDI"})
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


# ---------- correlation analysis (Pearson vs characteristics) ----------

def correlation_analysis(
    mmd_results: List[Dict[str, Any]],
    char_path: Path,
    p_threshold: float = 1.0,
) -> Dict[str, Any]:
    """Pearson correlation between MMD_Loss and each numeric column in char file.

    Note on ``p_threshold``:
        We default to 1.0 (NOT 0.1) to stay consistent with the upstream
        bundled search pipeline which BATTWISE wraps as Reflective Search
        mode (see ``problems/battery_mmd/eval.py``: every CASE_CONFIG sets
        ``p_threshold: 1``). With p<1.0 the "significant" filter is essentially
        a no-op and ``mean_sig`` reduces to ``mean |r|`` — that is the metric
        EOH minimises (objective = -mean_sig), so this is what the user sees
        in the live log and what's directly comparable with EOH's objective.

        Passing a stricter threshold (e.g. 0.1) only emits an additional
        ``num_significant`` count; ``mean_sig`` continues to follow whatever
        threshold you pass, but with the new default it is no longer hard-cut
        to 0.0 just because every Pearson p-value is above 0.1.
    """
    if not mmd_results:
        return {"features": [], "mean_sig": 0.0, "all_mean_abs": 0.0, "matched": 0}
    df_char = _load_data(char_path)
    if df_char is None:
        return {
            "features": [],
            "mean_sig": 0.0,
            "all_mean_abs": 0.0,
            "matched": 0,
            "error": f"cannot load characteristics file: {char_path}",
        }

    if "Cell_num" not in df_char.columns:
        if "Cell_num" in df_char.iloc[0].astype(str).values:
            df_char.columns = df_char.iloc[0]
            df_char = df_char.iloc[1:].reset_index(drop=True)
    if "Cell_num" not in df_char.columns:
        return {
            "features": [],
            "mean_sig": 0.0,
            "all_mean_abs": 0.0,
            "matched": 0,
            "error": "char file has no Cell_num column",
        }
    df_char["Cell_num"] = pd.to_numeric(df_char["Cell_num"], errors="coerce")

    df_mmd = pd.DataFrame(mmd_results)
    if "CDDI" in df_mmd.columns:
        df_mmd["MMD_Loss"] = df_mmd.pop("CDDI")
    df_merged = pd.merge(df_mmd, df_char, on="Cell_num", how="inner")

    feature_results: List[Dict[str, Any]] = []
    all_abs: List[float] = []
    sig_abs: List[float] = []

    for col in df_char.columns:
        if col == "Cell_num":
            continue
        y = pd.to_numeric(df_merged[col], errors="coerce")
        x = df_merged["MMD_Loss"]
        valid = ~np.isnan(x) & ~np.isnan(y)
        if int(valid.sum()) <= 2:
            continue
        r, p = pearsonr(x[valid], y[valid])
        if np.isnan(r) or np.isnan(p):
            continue
        is_sig = bool(p < p_threshold)
        feature_results.append(
            {"feature": str(col), "r": float(r), "p": float(p), "significant": is_sig}
        )
        all_abs.append(abs(r))
        if is_sig:
            sig_abs.append(abs(r))

    return {
        "features": feature_results,
        "matched": int(len(df_merged)),
        "all_mean_abs": float(np.mean(all_abs)) if all_abs else 0.0,
        "mean_sig": float(np.mean(sig_abs)) if sig_abs else 0.0,
        "num_significant": int(len(sig_abs)),
    }


# ---------- one-shot helper ----------

def run_full_pipeline(
    *,
    data_folder: Path,
    model_filename: str,
    char_filename: Optional[str],
    out_csv: Path,
    mmd_fn: Optional[Callable] = None,
    row_start: int = 0,
    row_end: Optional[int] = None,
    scale: float = 1.0,
    on_progress: Optional[Callable[[MMDProgress], None]] = None,
    p_threshold: float = 1.0,  # see correlation_analysis() docstring
    cells_filter: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Synchronous helper that drains the generator and saves the CSV.

    Returns a dict with ``results``, ``saved_to``, ``correlation`` and ``elapsed_s``.
    """
    if mmd_fn is None:
        mmd_fn = load_injected_function()

    folder = Path(data_folder)
    results: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    final_progress: Optional[MMDProgress] = None

    for prog in compute_mmd_for_folder(
        data_folder=folder,
        model_filename=model_filename,
        mmd_fn=mmd_fn,
        row_start=row_start,
        row_end=row_end,
        scale=scale,
        cells_filter=cells_filter,
    ):
        if on_progress:
            on_progress(prog)
        if prog.phase == "cell" and prog.mmd_loss is not None:
            results.append(
                {
                    "Filename": prog.filename,
                    "Cell_num": prog.cell_num,
                    "MMD_Loss": prog.mmd_loss,
                    "Sample_count": prog.sample_count,
                }
            )
        elif prog.phase == "cell_skipped":
            skipped.append(
                {"Filename": prog.filename, "Cell_num": prog.cell_num, "error": prog.error}
            )
        elif prog.phase == "done":
            final_progress = prog

    save_mmd_csv(results, out_csv)

    correlation: Optional[Dict[str, Any]] = None
    if char_filename:
        char_path = folder / char_filename
        if char_path.exists():
            correlation = correlation_analysis(results, char_path, p_threshold=p_threshold)

    return {
        "results": results,
        "skipped": skipped,
        "saved_to": str(out_csv),
        "elapsed_s": final_progress.elapsed_s if final_progress else 0.0,
        "total_cells": final_progress.total if final_progress else len(results),
        "correlation": correlation,
    }


# ---------- CLI ----------

def _cli(argv: Optional[List[str]] = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="MMD runner (CLI)")
    parser.add_argument("--data-folder", required=True, type=Path)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--char-file", default=None)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--inject-from",
        type=Path,
        default=None,
        help="Path to a .py file with `calculate_mmd_vN` to inject as MMD_cal_new before compute.",
    )
    parser.add_argument("--no-inject", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

    if args.inject_from is not None:
        code = args.inject_from.read_text(encoding="utf-8")
        path = inject_mmd_code(code)
        print(f"[*] injected -> {path}")
    elif not args.no_inject and not DIANCHI_MMD_TOOLS_NEW.exists():
        print(f"[!] No injection requested and {DIANCHI_MMD_TOOLS_NEW} missing.")
        return 1

    mmd_fn = load_injected_function()
    print(f"[*] using mmd_fn from {DIANCHI_MMD_TOOLS_NEW}")

    def progress(p: MMDProgress) -> None:
        if p.phase == "cell":
            print(f"  [{p.index}/{p.total}] Cell{p.cell_num:>3} loss={p.mmd_loss:.6f}")
        elif p.phase == "cell_skipped":
            print(f"  [skip] Cell{p.cell_num} reason={p.error}")
        elif p.phase == "model_loaded":
            print(f"[*] model loaded: {p.filename} (samples={p.sample_count})")
        elif p.phase == "done":
            print(f"[*] done in {p.elapsed_s:.1f}s, {p.index}/{p.total} cells succeeded")

    summary = run_full_pipeline(
        data_folder=args.data_folder,
        model_filename=args.model_file,
        char_filename=args.char_file,
        out_csv=args.out_csv,
        mmd_fn=mmd_fn,
        scale=args.scale,
        on_progress=progress,
    )
    print("\n=== summary ===")
    print(json.dumps(
        {
            "saved_to": summary["saved_to"],
            "total_cells": summary["total_cells"],
            "elapsed_s": summary["elapsed_s"],
            "correlation": summary.get("correlation"),
            "skipped": len(summary["skipped"]),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
