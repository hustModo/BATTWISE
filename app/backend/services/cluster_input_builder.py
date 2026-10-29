"""Build minimal clustering input workbooks.

The system should not depend on the legacy First_200_select workbooks.
Both main First-200s clustering and All-200 refinement consume a derived
session-local table with exactly the fields needed by the algorithm:

    Cell_num | OCV | IR | CDDI

Capacity may be joined later for explanation, but it is deliberately not
included here so it cannot leak into clustering features or GA fitness.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def _read_excel(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path, engine="calamine")
    except Exception:
        return pd.read_excel(path, engine="openpyxl")


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _load_statis(statis_xlsx: Path) -> pd.DataFrame:
    df = _clean_columns(_read_excel(statis_xlsx))
    required = {"Cell_num", "OCV", "IR"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{statis_xlsx.name} missing required columns: {sorted(missing)}")
    out = df[["Cell_num", "OCV", "IR"]].copy()
    out["Cell_num"] = pd.to_numeric(out["Cell_num"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["Cell_num"]).copy()
    out["Cell_num"] = out["Cell_num"].astype(int)
    out["OCV"] = pd.to_numeric(out["OCV"], errors="coerce")
    out["IR"] = pd.to_numeric(out["IR"], errors="coerce")
    return out.dropna(subset=["OCV", "IR"])


def _load_ddia(ddia_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(ddia_csv)
    df.columns = [str(c).strip() for c in df.columns]
    if "Cell_num" not in df.columns:
        raise KeyError(f"{ddia_csv.name} missing Cell_num column")
    ddia_col = None
    for candidate in ("CDDI", "DDIA", "MMD_Loss", "CDDI_Loss", "DDIA_Loss", "DDI"):
        if candidate in df.columns:
            ddia_col = candidate
            break
    if ddia_col is None:
        raise KeyError(f"{ddia_csv.name} missing CDDI/MMD_Loss column")
    out = df[["Cell_num", ddia_col]].rename(columns={ddia_col: "CDDI"}).copy()
    out["Cell_num"] = pd.to_numeric(out["Cell_num"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["Cell_num"]).copy()
    out["Cell_num"] = out["Cell_num"].astype(int)
    out["CDDI"] = pd.to_numeric(out["CDDI"], errors="coerce")
    return out.dropna(subset=["CDDI"])


def build_cluster_input(
    *,
    statis_xlsx: Path,
    ddia_csv: Path,
    out_xlsx: Path,
    cell_nums: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Build a session-local clustering input table.

    Args:
        statis_xlsx: workbook containing Cell_num, OCV, IR (Capacity ignored).
        ddia_csv: CDDI compute result with Cell_num and CDDI/MMD_Loss.
        out_xlsx: destination workbook.
        cell_nums: optional selected-cell filter for All-200 refinement.
    """
    statis = _load_statis(Path(statis_xlsx))
    ddia = _load_ddia(Path(ddia_csv))
    merged = pd.merge(statis, ddia, on="Cell_num", how="inner")
    if cell_nums is not None:
        wanted = {int(c) for c in cell_nums}
        merged = merged[merged["Cell_num"].isin(wanted)].copy()
    merged = merged[["Cell_num", "OCV", "IR", "CDDI"]].sort_values("Cell_num")
    merged = merged.reset_index(drop=True)
    if len(merged) < 4:
        raise RuntimeError(f"Not enough rows for clustering input: {len(merged)}")

    out_xlsx = Path(out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    merged.to_excel(out_xlsx, index=False)
    return merged
