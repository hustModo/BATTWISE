"""Offline test: end-to-end clustering on the built-in First-200s session output."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd

from app.backend.services import clustering_runner, data_resolver
from app.backend.services.clustering_runner import (
    cluster_dataset,
    cluster_subset,
    merge_mmd_into_indicator,
    run_adaptive_kmeans,
    summarize_clusters,
)


def t_merge_mmd_into_indicator() -> None:
    ds = data_resolver.resolve("builtin-first-200s")
    csv = ds.folder / "mmd_results.csv"
    assert csv.exists(), "no built-in mmd_results.csv to test against"

    merged = merge_mmd_into_indicator(csv, ds.indicator_path)
    assert "MMD_Loss" in merged.columns
    assert "OCV" in merged.columns and "IR" in merged.columns and "Capacity" in merged.columns
    assert len(merged) == 71, f"expected 71 cells, got {len(merged)}"
    expected_mmd = pd.read_csv(csv)[["Cell_num", "MMD_Loss"]].rename(
        columns={"MMD_Loss": "new"}
    )
    merged_check = merged.merge(expected_mmd, on="Cell_num", how="inner")
    diff = (merged_check["MMD_Loss"] - merged_check["new"]).abs().max()
    assert diff < 1e-6, f"MMD_Loss column not overwritten with new values, diff={diff}"
    print(f"  [OK] merged {len(merged)} rows; MMD_Loss column replaced (max diff {diff:.2e})")


def t_run_adaptive_kmeans() -> None:
    import numpy as np
    rng = np.random.default_rng(7)
    # 2 small well-separated blobs; keep this unit test fast.
    blob = lambda c: rng.normal(loc=c, scale=0.05, size=(6, 3))
    data = np.vstack([blob([0.1, 0.1, 0.1]), blob([0.9, 0.9, 0.9])])
    cap = rng.uniform(0, 1, size=data.shape[0])

    result = run_adaptive_kmeans(
        data, cap, pop_size=5, max_iter=2, min_k=2, max_k=2,
        use_capacity=False, seed=11,
    )
    assert result.k == 2, f"unexpected k={result.k}"
    assert len(result.labels) == data.shape[0]
    assert len(result.history) == 2
    print(f"  [OK] adaptive K found k={result.k} (dbi={result.dbi:.3f}) "
          f"in {result.elapsed_s:.2f}s, {len(result.history)} iters")


def t_cluster_dataset_full() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cells = list(range(1, 13))
        indicator = pd.DataFrame({
            "Cell_num": cells,
            "OCV": [3.1 + i * 0.01 for i in range(12)],
            "IR": [8.0 + (i % 4) * 0.5 for i in range(12)],
            "Capacity": [2.0 - i * 0.02 for i in range(12)],  # should not drive clustering
        })
        mmd = pd.DataFrame({
            "Cell_num": cells,
            "MMD_Loss": [0.1 + (i % 3) * 0.2 for i in range(12)],
        })
        indicator_xlsx = root / "statis.xlsx"
        csv = root / "ddia.csv"
        indicator.to_excel(indicator_xlsx, index=False)
        mmd.to_csv(csv, index=False)
        out_xlsx = root / "cluster_input_used.xlsx"
        result = cluster_dataset(
            mmd_csv=csv,
            indicator_xlsx=indicator_xlsx,
            out_xlsx=out_xlsx,
            pop_size=5,
            max_iter=2,
            min_k=2,
            max_k=3,
            use_capacity=False,
            seed=42,
        )
        assert 2 <= result["k"] <= 3, f"k out of expected band: {result['k']}"
        summary = result["summary"]
        assert len(summary["scatter"]) == result["n_samples"]
        assert len(summary["per_cluster"]) == result["k"]
        assert result["feature_cols"] == ["OCV", "IR", "MMD_Loss"]
        assert summary["axes"] == {"x": "OCV", "y": "IR"}
        for clust in summary["fingerprint"]["clusters"]:
            assert len(clust["values"]) == len(summary["fingerprint"]["features"])
        sizes = [c["size"] for c in summary["per_cluster"]]
        assert sum(sizes) == result["n_samples"]
        assert min(sizes) >= 3
        assert Path(result["merged_xlsx"]).exists()
        assert Path(result["cluster_xlsx"]).exists()
        saved_cols = list(pd.read_excel(result["cluster_xlsx"]).columns)
        assert "CDDI" in saved_cols and "MMD_Loss" not in saved_cols
    print(f"  [OK] cluster_dataset finished: k={result['k']}, fitness={result['fitness']:.3f}, "
          f"dbi={result['dbi']:.3f}, sizes={sizes}, {result['elapsed_s']:.1f}s")
    print(f"       merged_xlsx: {Path(result['merged_xlsx']).name}")
    print(f"       cluster_xlsx: {Path(result['cluster_xlsx']).name}")


def t_cluster_subset_ocv_ir_ddia_without_capacity() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cells = list(range(1, 13))
        cluster_input = pd.DataFrame({
            "Cell_num": cells,
            "OCV": [3.1 + i * 0.01 for i in range(12)],
            "IR": [8.0 + (i % 4) * 0.5 for i in range(12)],
            "DDIA": [0.1 + (i % 3) * 0.2 for i in range(12)],
        })
        posthoc = pd.DataFrame({
            "Cell_num": cells,
            "OCV": cluster_input["OCV"],
            "IR": cluster_input["IR"],
            "Capacity": [2.0 - i * 0.03 for i in range(12)],
        })
        input_xlsx = root / "all200_cluster_input.xlsx"
        posthoc_xlsx = root / "Statis data.xlsx"
        out_xlsx = root / "Sub_Cluster_label.xlsx"
        cluster_input.to_excel(input_xlsx, index=False)
        posthoc.to_excel(posthoc_xlsx, index=False)

        result = cluster_subset(
            mmd_csv=None,
            indicator_xlsx=input_xlsx,
            cell_nums=cells,
            out_xlsx=out_xlsx,
            pop_size=5,
            max_iter=2,
            min_k=2,
            max_k=2,
            seed=7,
            feature_weights={"ddia": 2.0},
            posthoc_xlsx=posthoc_xlsx,
        )
        assert result["n_samples"] == 12
        assert result["k"] == 2
        assert "OCV" in result["feature_cols"]
        assert "IR" in result["feature_cols"]
        assert "MMD_Loss" in result["feature_cols"]
        assert result["feature_weights"]["ddia"] == 2.0
        assert Path(result["cluster_xlsx"]).exists()
        assert result["summary"]["axes"] == {"x": "OCV", "y": "IR"}
        assert result["summary"]["boxplot"][0]["values"] != []
        assert result["summary"]["per_cluster"][0]["capacity"]["mean"] is not None
        saved_cols = list(pd.read_excel(result["cluster_xlsx"]).columns)
        assert "Capacity" in saved_cols  # post-hoc report column, not a training feature
    print("  [OK] cluster_subset uses OCV/IR/DDIA without Capacity")


def main() -> int:
    print("== test_clustering_runner ==")
    tests = [
        t_merge_mmd_into_indicator,
        t_run_adaptive_kmeans,
        t_cluster_dataset_full,
        t_cluster_subset_ocv_ir_ddia_without_capacity,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            failures.append(f"{t.__name__}: {exc}")
            print(f"  [FAIL] {t.__name__}: {exc}")
    if failures:
        print(f"FAILED: {len(failures)}/{len(tests)}")
        return 1
    print(f"PASSED: {len(tests)}/{len(tests)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
