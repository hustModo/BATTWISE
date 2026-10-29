"""Unit test: summarize_clusters now produces a `dispersion` block with
per-cluster × per-feature mean/std/CV plus a capacity aggregate (F-ratio,
tightest/loosest cluster), AND write_dispersion_xlsx exports the same to
a 3-sheet Excel file.

Synthetic data layout mirrors the production schema: Cell_num + Capacity +
OCV + IR + MMD_Loss + a couple of curve-feature columns.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.services import clustering_runner as cr


def _synthetic_df(n_per_group: int = 10, k: int = 3) -> tuple[pd.DataFrame, list[int]]:
    rng = np.random.default_rng(7)
    n = n_per_group * k
    capacity = np.concatenate([
        rng.normal(2.0 - 0.3 * g, 0.02 + 0.02 * g, n_per_group) for g in range(k)
    ])
    df = pd.DataFrame({
        "Cell_num": range(1, n + 1),
        "OCV": rng.normal(3.7, 0.05, n),
        "IR": rng.normal(0.08, 0.01, n),
        "Capacity": capacity,
        "MMD_Loss": rng.uniform(0.1, 0.9, n),
        "F1": rng.normal(0.5, 0.1, n),
        "F2": rng.normal(1.0, 0.2, n),
    })
    labels = [g for g in range(k) for _ in range(n_per_group)]
    return df, labels


def main() -> int:
    failed = 0
    df, labels = _synthetic_df()
    summary = cr.summarize_clusters(df, labels)

    disp = summary.get("dispersion") or {}
    if not disp:
        print("  [FAIL] dispersion missing from summary")
        return 1

    if "Capacity" not in disp.get("features", []):
        print("  [FAIL] Capacity not in features list")
        failed += 1
    else:
        print("  [OK] features include Capacity + curve cols:", disp["features"])

    if len(disp.get("per_cluster", [])) != 3:
        print(f"  [FAIL] expected 3 cluster rows, got {len(disp.get('per_cluster', []))}")
        failed += 1

    for r in disp.get("per_cluster", []):
        cap = r.get("Capacity") or {}
        for key in ("mean", "std", "cv"):
            if cap.get(key) is None:
                print(f"  [FAIL] cluster {r.get('cluster')} Capacity.{key} missing")
                failed += 1

    aggr = disp.get("capacity_aggregate") or {}
    for key in ("tightest_cluster", "loosest_cluster", "mean_of_std", "f_ratio"):
        if aggr.get(key) is None:
            print(f"  [FAIL] capacity_aggregate.{key} missing")
            failed += 1
    if aggr.get("tightest_cluster") == aggr.get("loosest_cluster"):
        print("  [FAIL] tightest and loosest should differ on synthetic 3-group data")
        failed += 1
    if aggr.get("f_ratio") is not None and aggr["f_ratio"] < 1.0:
        print(f"  [FAIL] F-ratio={aggr['f_ratio']:.2f} is unexpectedly low (clusters are well-separated by construction)")
        failed += 1
    else:
        print(f"  [OK] capacity F-ratio={aggr.get('f_ratio'):.2f} ≫ 1 (groups well-separated)")

    # XLSX export
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "disp.xlsx"
        cr.write_dispersion_xlsx(disp, out)
        if not out.exists():
            print("  [FAIL] dispersion XLSX not written")
            failed += 1
        else:
            sheets = pd.ExcelFile(out).sheet_names
            for expected in ("by_cluster", "by_feature", "capacity_aggregate"):
                if expected not in sheets:
                    print(f"  [FAIL] sheet '{expected}' missing; got {sheets}")
                    failed += 1
                else:
                    print(f"  [OK] sheet '{expected}' present")
            by_cluster = pd.read_excel(out, sheet_name="by_cluster")
            if "Capacity_std" not in by_cluster.columns:
                print("  [FAIL] Capacity_std column missing from by_cluster sheet")
                failed += 1
            else:
                print(f"  [OK] by_cluster has {by_cluster.shape[0]}×{by_cluster.shape[1]} table")

    print("PASSED" if failed == 0 else f"FAILED ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
