"""Merge All-200 sub-cluster labels back into the original parent clustering.

User flow: after the user runs the main First-200 clustering AND then
refines one (or several) parent clusters with All-200, this module
produces a single Excel workbook whose ``Merged_label`` column splits
those parent clusters into their All-200 sub-groups while leaving
non-refined cells untouched. Dispersion stats are then computed on the
new labelling so the dialog can show the user how much the sub-split
tightened the within-cluster variance.

The merged labels use a sequential integer scheme starting at
``max(original)+1`` so they're guaranteed to not collide with any
original cluster id, and the workbook carries a ``Mapping`` sheet
explaining which new id came from which (parent.sub) pair. A
human-readable ``Merged_display`` column ("3→0" / "3→1") is included
for the eyeball test.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd

from app.backend.services import clustering_runner


def _read_clustering(xlsx: Path) -> pd.DataFrame:
    """Read a cluster-label xlsx; raise a clear error if columns missing."""
    df = pd.read_excel(xlsx, engine="calamine")
    if "Cluster_label" not in df.columns or "Cell_num" not in df.columns:
        raise KeyError(
            f"{xlsx.name} must contain 'Cluster_label' and 'Cell_num' columns; "
            f"got {list(df.columns)[:8]}..."
        )
    df["Cell_num"] = df["Cell_num"].astype(int)
    df["Cluster_label"] = df["Cluster_label"].astype(int)
    return df


def merge_subclusters_back(
    *,
    parent_xlsx: Path,
    sub_xlsxs_by_parent: Mapping[int, Path],
    out_xlsx: Optional[Path] = None,
) -> Dict[str, Any]:
    """Replace parent labels of refined cells with sub-cluster labels.

    Args:
        parent_xlsx: original ``Cluster_label.xlsx`` from the main run.
        sub_xlsxs_by_parent: dict mapping parent cluster id -> the
            ``Sub_Cluster{P}_label.xlsx`` that was produced by the
            RefinementAgent for that parent.
        out_xlsx: where to write the merged workbook (multiple sheets).

    Returns dict with keys:
        merged_df:        the full dataframe with ``Merged_label`` column
        mapping:          list of {new_label, parent, sub_label, n_cells}
        dispersion:       dict from clustering_runner._compute_dispersion
        k_before/k_after: cluster counts before vs after merge
        xlsx_path:        str path of the written workbook (or None)
    """
    parent_df = _read_clustering(parent_xlsx)
    parent_df = parent_df.copy()
    parent_df = clustering_runner._normalise_indicator_columns(parent_df)
    parent_df = parent_df.drop(columns=["DDIA_source"], errors="ignore")
    parent_df["Merged_label"] = parent_df["Cluster_label"]
    parent_df["Merged_display"] = parent_df["Cluster_label"].astype(str)
    parent_df["Final_label"] = parent_df["Cluster_label"]
    parent_df["Final_display"] = parent_df["Cluster_label"].astype(str)
    parent_df["Original_group"] = parent_df["Cluster_label"]
    parent_df["Refined_group"] = pd.NA
    parent_df["Final_group"] = parent_df["Cluster_label"]
    parent_df["Label_source"] = "first_200_main"
    parent_df["CDDI_source"] = "first_200s"
    parent_df["Parent_cluster"] = parent_df["Cluster_label"]
    parent_df["Sub_cluster"] = pd.NA

    # Allocate a unique fresh integer id for every (parent, sub) cell, then
    # remember the mapping so we can write a Mapping sheet and let the user
    # trace where each new label came from.
    next_label = int(parent_df["Cluster_label"].max()) + 1
    mapping: List[Dict[str, Any]] = []

    for parent_id, sub_path in sub_xlsxs_by_parent.items():
        sub_df = _read_clustering(Path(sub_path))
        # Multi-parent refinement writes ONE sub-cluster workbook for the
        # union of selected cells. When the same workbook is passed for
        # several parents, filter it back to the cells that originally
        # belonged to this parent; otherwise a later parent overwrites the
        # earlier parent's merged labels.
        parent_cells = set(
            parent_df.loc[
                parent_df["Cluster_label"].astype(int) == int(parent_id),
                "Cell_num",
            ].astype(int)
        )
        sub_df = sub_df[sub_df["Cell_num"].astype(int).isin(parent_cells)].copy()
        if sub_df.empty:
            continue
        unique_subs = sorted(sub_df["Cluster_label"].astype(int).unique().tolist())
        # Map every sub-cluster id to a globally unique new id.
        sub_to_new: Dict[int, int] = {}
        for s in unique_subs:
            sub_to_new[s] = next_label
            next_label += 1

        # Sub_df has only the cells that belong to this parent cluster,
        # so we can replace by Cell_num.
        sub_lookup: Dict[int, int] = dict(zip(sub_df["Cell_num"], sub_df["Cluster_label"]))
        for cell_num, sub_lbl in sub_lookup.items():
            new_lbl = sub_to_new[int(sub_lbl)]
            mask = parent_df["Cell_num"] == int(cell_num)
            parent_df.loc[mask, "Merged_label"] = new_lbl
            parent_df.loc[mask, "Merged_display"] = f"{parent_id}\u2192{int(sub_lbl)}"
            parent_df.loc[mask, "Final_label"] = new_lbl
            parent_df.loc[mask, "Final_display"] = f"{parent_id}\u2192{int(sub_lbl)}"
            parent_df.loc[mask, "Original_group"] = int(parent_id)
            parent_df.loc[mask, "Refined_group"] = int(sub_lbl)
            parent_df.loc[mask, "Final_group"] = new_lbl
            parent_df.loc[mask, "Label_source"] = "all_200_refined"
            parent_df.loc[mask, "CDDI_source"] = "all_200"
            parent_df.loc[mask, "Parent_cluster"] = int(parent_id)
            parent_df.loc[mask, "Sub_cluster"] = int(sub_lbl)

        for sub_id, new_id in sub_to_new.items():
            n_cells = int((sub_df["Cluster_label"] == sub_id).sum())
            mapping.append({
                "new_label": int(new_id),
                "display": f"{parent_id}\u2192{int(sub_id)}",
                "parent_cluster": int(parent_id),
                "sub_label": int(sub_id),
                "n_cells": n_cells,
            })

    # Recompute dispersion on the merged labelling. The summarize_clusters
    # helper expects an integer Cluster_label column, so we point it at the
    # merged labels temporarily.
    merged_view = parent_df.drop(columns=["Cluster_label"]).rename(
        columns={"Merged_label": "Cluster_label"}
    )
    summary = clustering_runner.summarize_clusters(
        merged_view,
        merged_view["Cluster_label"].astype(int).tolist(),
    )
    dispersion = summary.get("dispersion") or {}
    # Restore Merged_label column on the returned df (summarize_clusters
    # mutated a copy internally, so parent_df is intact).
    merged_df = parent_df
    merged_df = merged_df.rename(columns={"MMD_Loss": "CDDI"})

    k_before = int(parent_df["Cluster_label"].nunique())
    k_after = int(parent_df["Merged_label"].nunique())

    xlsx_path: Optional[str] = None
    dispersion_xlsx_path: Optional[str] = None
    if out_xlsx is not None:
        out_xlsx = Path(out_xlsx)
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)
        dispersion_path = out_xlsx.with_name(out_xlsx.stem + "_dispersion.xlsx")
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
            merged_df.to_excel(xw, sheet_name="merged_labels", index=False)
            pd.DataFrame(mapping).to_excel(xw, sheet_name="mapping", index=False)
            # Also write the dispersion table next to the labels so the
            # analyst can audit without re-running anything.
            try:
                clustering_runner.write_dispersion_xlsx(dispersion, dispersion_path)
                dispersion_xlsx_path = str(dispersion_path)
            except Exception:  # noqa: BLE001
                pass
        xlsx_path = str(out_xlsx)

    return {
        "merged_df": merged_df,
        "mapping": mapping,
        "dispersion": dispersion,
        "k_before": k_before,
        "k_after": k_after,
        "xlsx_path": xlsx_path,
        "dispersion_xlsx_path": dispersion_xlsx_path,
        "summary": summary,
    }


def merge_all_available_subclusters(
    *,
    parent_xlsx: Path,
    session_ctx: Mapping[str, Any],
    out_xlsx: Optional[Path] = None,
) -> Dict[str, Any]:
    """Create the global sorting result from every refined parent found.

    Cells that were never refined keep their First-200 parent label; cells
    belonging to any refined parent use the latest All-200 sub-cluster label
    registered in the session context.
    """
    parent_df = _read_clustering(parent_xlsx)
    parent_ids = sorted(parent_df["Cluster_label"].astype(int).unique().tolist())
    sub_paths = _flatten_subxlsxs(parent_ids, session_ctx)
    return merge_subclusters_back(
        parent_xlsx=parent_xlsx,
        sub_xlsxs_by_parent=sub_paths,
        out_xlsx=out_xlsx,
    )


def _flatten_subxlsxs(
    parent_ids: Iterable[int],
    session_ctx: Mapping[str, Any],
) -> Dict[int, Path]:
    """Best-effort lookup of every sub-cluster xlsx the user has produced.

    The RefinementAgent stashes paths under several keys depending on
    how the refinement was triggered (single-cluster vs multi-cluster).
    We try them in priority order so callers don't have to.
    """
    out: Dict[int, Path] = {}
    for p in parent_ids:
        # legacy single-cluster key, then the new tagged keys.
        keys = [
            f"sub_cluster_xlsx_{p}",
            f"sub_cluster_xlsx_cluster{p}",
        ]
        # Also accept any multi-cluster tag that happened to include this parent.
        for k in keys:
            v = session_ctx.get(k)
            if v and Path(v).exists():
                out[int(p)] = Path(v)
                break
    return out
