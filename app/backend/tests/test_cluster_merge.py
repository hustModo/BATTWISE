"""Unit test: cluster_merge.merge_subclusters_back replaces parent labels
of refined cells with globally-unique sub-cluster ids, keeps non-refined
cells untouched, recomputes dispersion on the merged labelling, and
writes a workbook with 'merged_labels' + 'mapping' sheets plus a sister
'_dispersion.xlsx'.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.services import cluster_merge


def _synthetic_parent_and_subs(td: Path):
    rng = np.random.default_rng(42)
    parent = pd.DataFrame({
        "Cell_num": range(1, 31),
        "OCV": rng.normal(3.7, 0.05, 30),
        "IR": rng.normal(0.08, 0.01, 30),
        "Capacity": np.concatenate([
            rng.normal(2.0, 0.02, 8),
            rng.normal(1.8, 0.04, 7),
            rng.normal(1.5, 0.06, 8),
            rng.normal(1.2, 0.08, 7),
        ]),
        "MMD_Loss": rng.uniform(0.1, 0.9, 30),
        "F_curve_x": rng.normal(0.5, 0.1, 30),
        "Cluster_label": [0] * 8 + [1] * 7 + [2] * 8 + [3] * 7,
    })
    sub2 = parent[parent["Cluster_label"] == 2].copy()
    sub2["Cluster_label"] = [0] * 4 + [1] * 4
    sub3 = parent[parent["Cluster_label"] == 3].copy()
    sub3["Cluster_label"] = [0] * 3 + [1] * 4
    parent.to_excel(td / "parent.xlsx", index=False)
    sub2.to_excel(td / "sub2.xlsx", index=False)
    sub3.to_excel(td / "sub3.xlsx", index=False)
    return td / "parent.xlsx", {2: td / "sub2.xlsx", 3: td / "sub3.xlsx"}


def main() -> int:
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        parent_path, subs = _synthetic_parent_and_subs(td)
        out = td / "merged.xlsx"
        result = cluster_merge.merge_subclusters_back(
            parent_xlsx=parent_path,
            sub_xlsxs_by_parent=subs,
            out_xlsx=out,
        )

        if result["k_before"] != 4:
            print(f"  [FAIL] k_before={result['k_before']} (expected 4)")
            failed += 1
        else:
            print("  [OK] k_before=4")
        if result["k_after"] != 6:
            print(f"  [FAIL] k_after={result['k_after']} (expected 6 after splitting 2 parents into 2 each)")
            failed += 1
        else:
            print("  [OK] k_after=6")

        mapping_new_ids = [m["new_label"] for m in result["mapping"]]
        if len(set(mapping_new_ids)) != len(mapping_new_ids):
            print(f"  [FAIL] mapping has duplicate new_label ids: {mapping_new_ids}")
            failed += 1
        else:
            print(f"  [OK] mapping has 4 unique new ids: {mapping_new_ids}")

        if min(mapping_new_ids) <= 3:
            print(f"  [FAIL] new label {min(mapping_new_ids)} collides with original 0-3")
            failed += 1
        else:
            print("  [OK] new ids do not collide with originals (0..3)")

        merged = result["merged_df"]
        if not (merged.loc[merged["Cell_num"] <= 8, "Merged_label"] == 0).all():
            print("  [FAIL] non-refined Cluster 0 cells got mutated")
            failed += 1
        else:
            print("  [OK] non-refined Cluster 0 (Cell 1..8) kept label 0")
        if not (merged.loc[merged["Cell_num"] <= 15, "Label_source"] == "first_200_main").all():
            print("  [FAIL] non-refined cells should be labelled first_200_main")
            failed += 1
        else:
            print("  [OK] non-refined cells marked first_200_main")
        if not (merged.loc[(merged["Cell_num"] > 8) & (merged["Cell_num"] <= 15), "Merged_label"] == 1).all():
            print("  [FAIL] non-refined Cluster 1 cells got mutated")
            failed += 1
        else:
            print("  [OK] non-refined Cluster 1 (Cell 9..15) kept label 1")

        refined_cells = merged[merged["Cell_num"] > 15]
        if refined_cells["Merged_label"].isin([2, 3]).any():
            print(f"  [FAIL] some refined cells still have parent labels: "
                  f"{refined_cells.loc[refined_cells['Merged_label'].isin([2, 3]), ['Cell_num', 'Merged_label']].head().to_dict()}")
            failed += 1
        else:
            print("  [OK] all refined cells got new labels (no parent leakage)")
        if not (refined_cells["Label_source"] == "all_200_refined").all():
            print("  [FAIL] refined cells should be labelled all_200_refined")
            failed += 1
        else:
            print("  [OK] refined cells marked all_200_refined")
        if "Final_label" not in merged.columns or "Final_display" not in merged.columns:
            print("  [FAIL] final label/display columns missing")
            failed += 1
        else:
            print("  [OK] final label/display columns present")

        # Workbook sanity
        if not out.exists():
            print("  [FAIL] merged.xlsx not written")
            failed += 1
        else:
            sheets = pd.ExcelFile(out).sheet_names
            for s in ("merged_labels", "mapping"):
                if s not in sheets:
                    print(f"  [FAIL] missing sheet '{s}'; got {sheets}")
                    failed += 1
                else:
                    print(f"  [OK] sheet '{s}' present")
            sister = out.with_name(out.stem + "_dispersion.xlsx")
            if not sister.exists():
                print(f"  [FAIL] sister dispersion file missing at {sister}")
                failed += 1
            else:
                print(f"  [OK] sister dispersion xlsx written")
            if result.get("dispersion_xlsx_path") != str(sister):
                print(f"  [FAIL] dispersion_xlsx_path not returned correctly: {result.get('dispersion_xlsx_path')}")
                failed += 1
            else:
                print("  [OK] dispersion_xlsx_path returned")

        disp = result.get("dispersion") or {}
        if not (disp.get("per_cluster") or []):
            print("  [FAIL] dispersion.per_cluster is empty")
            failed += 1
        else:
            print(f"  [OK] dispersion computed on merged labels: "
                  f"{len(disp['per_cluster'])} cluster rows")

        # Regression: multi-parent refinement passes the SAME union
        # Sub_Clusters_x-y_label.xlsx for every selected parent. The merge
        # function must filter rows back to each original parent; otherwise
        # the later parent overwrites the earlier parent's cells.
        union = pd.concat([
            pd.read_excel(td / "sub2.xlsx"),
            pd.read_excel(td / "sub3.xlsx"),
        ], ignore_index=True)
        union_path = td / "sub_union.xlsx"
        union.to_excel(union_path, index=False)
        union_result = cluster_merge.merge_subclusters_back(
            parent_xlsx=parent_path,
            sub_xlsxs_by_parent={2: union_path, 3: union_path},
            out_xlsx=td / "merged_union.xlsx",
        )
        union_merged = union_result["merged_df"]
        c2 = union_merged[(union_merged["Cell_num"] >= 16) & (union_merged["Cell_num"] <= 23)]
        c3 = union_merged[union_merged["Cell_num"] >= 24]
        c2_display_ok = c2["Merged_display"].astype(str).str.startswith("2→").all()
        c3_display_ok = c3["Merged_display"].astype(str).str.startswith("3→").all()
        if not c2_display_ok or not c3_display_ok:
            print("  [FAIL] union sub-file merge leaked labels across parents")
            print(c2[["Cell_num", "Merged_display"]].to_string(index=False))
            print(c3[["Cell_num", "Merged_display"]].to_string(index=False))
            failed += 1
        else:
            print("  [OK] union sub-file merge filters rows per original parent")
        if union_result["k_after"] != 6:
            print(f"  [FAIL] union k_after={union_result['k_after']} (expected 6)")
            failed += 1
        else:
            print("  [OK] union k_after=6")

        global_result = cluster_merge.merge_all_available_subclusters(
            parent_xlsx=parent_path,
            session_ctx={
                "sub_cluster_xlsx_2": union_path,
                "sub_cluster_xlsx_3": union_path,
            },
            out_xlsx=td / "global.xlsx",
        )
        if global_result["k_after"] != 6:
            print(f"  [FAIL] global k_after={global_result['k_after']} (expected 6)")
            failed += 1
        else:
            print("  [OK] global merge includes all available refinements")
        global_df = global_result["merged_df"]
        sources = set(global_df["Label_source"].astype(str))
        if sources != {"first_200_main", "all_200_refined"}:
            print(f"  [FAIL] global Label_source set={sources}")
            failed += 1
        else:
            print("  [OK] global Label_source records main vs refined origin")

    print("PASSED" if failed == 0 else f"FAILED ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
