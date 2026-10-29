"""Integration test: merge_subclusters_back output is the right shape
for the refinement chat assembly to render BOTH a capacity dispersion
block AND a curve-feature dispersion table.

This pins the wire between `cluster_merge` and `clustering_runner.
format_curve_dispersion_markdown` so a future refactor that drops
curve features from the merged dispersion (or removes the markdown
hook in `refinement_agent`) gets caught.

We don't drive the full RefinementAgent — that's a heavy LLM+Compute
E2E. Instead we synthesize a parent + sub workbook, run the production
merge function, and assert the post-merge dispersion can be rendered
end-to-end (which is exactly what `refinement_agent._handle_subcluster`
does after a successful sub-cluster).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.services import cluster_merge, clustering_runner  # noqa: E402


def _synthetic_parent_and_subs(td: Path) -> Tuple[Path, Dict[int, Path]]:
    """Parent with 30 cells across 4 clusters, plus sub-files that
    further split clusters 2 and 3. Mirrors the synthetic setup
    `test_cluster_merge.py` uses, but extends it with realistic curve
    features so format_curve_dispersion_markdown has something to
    render."""
    rng = np.random.default_rng(42)
    parent = pd.DataFrame({
        "Cell_num": range(1, 31),
        # Electrochemical (should not appear on fingerprint but DO
        # appear in dispersion).
        "OCV": rng.normal(3.7, 0.05, 30),
        "IR": rng.normal(0.08, 0.01, 30),
        # Curve-derived features with distinct per-cluster spreads so
        # the dispersion table is meaningful.
        "Gradchar_10s": np.concatenate([
            rng.normal(0.012, 0.0005, 8),
            rng.normal(0.010, 0.0010, 7),
            rng.normal(0.014, 0.0020, 8),
            rng.normal(0.011, 0.0030, 7),
        ]),
        "V_drop": np.concatenate([
            rng.normal(0.30, 0.02, 8),
            rng.normal(0.35, 0.04, 7),
            rng.normal(0.28, 0.08, 8),
            rng.normal(0.32, 0.10, 7),
        ]),
        "ICAp_c": np.concatenate([
            rng.normal(0.50, 0.01, 8),
            rng.normal(0.55, 0.02, 7),
            rng.normal(0.48, 0.03, 8),
            rng.normal(0.52, 0.04, 7),
        ]),
        "Capacity": np.concatenate([
            rng.normal(2.0, 0.02, 8),
            rng.normal(1.8, 0.04, 7),
            rng.normal(1.5, 0.06, 8),
            rng.normal(1.2, 0.08, 7),
        ]),
        "MMD_Loss": rng.uniform(0.1, 0.9, 30),
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


def test_merge_produces_dispersion_with_curve_features() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        parent, subs = _synthetic_parent_and_subs(td)
        merged = cluster_merge.merge_subclusters_back(
            parent_xlsx=parent,
            sub_xlsxs_by_parent=subs,
            out_xlsx=td / "merged.xlsx",
        )
        disp = merged.get("dispersion") or {}
        feats = set(disp.get("features") or [])
        # Both curve features AND Capacity are present in the
        # post-merge dispersion. Electrochemical OCV/IR are present
        # too (they're allowed in the XLSX, just not on the radar).
        assert {"Capacity", "Gradchar_10s", "V_drop", "ICAp_c"}.issubset(feats), feats
        # Per-cluster rows exist for every merged label.
        per_cluster = disp.get("per_cluster") or []
        merged_labels = {m["new_label"] for m in (merged.get("mapping") or [])}
        # All sub-labels appear in dispersion (and so do the non-refined
        # parents 0 and 1).
        seen = {r.get("cluster") for r in per_cluster}
        assert merged_labels.issubset(seen), (merged_labels, seen)
    return 0


def test_merge_dispersion_renders_curve_markdown() -> int:
    """The end-to-end glue: refinement_agent calls
    format_curve_dispersion_markdown on merge_subclusters_back's
    dispersion. Verify the combination produces a non-empty table
    with the expected structure."""
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        parent, subs = _synthetic_parent_and_subs(td)
        merged = cluster_merge.merge_subclusters_back(
            parent_xlsx=parent,
            sub_xlsxs_by_parent=subs,
            out_xlsx=td / "merged.xlsx",
        )
        md = clustering_runner.format_curve_dispersion_markdown(
            merged["dispersion"], top_n=3,
        )
        assert md is not None, "expected curve-feature markdown after merge"
        assert "**Curve-feature dispersion**" in md
        # Header row → at least one curve feature appears in the table.
        assert "`Gradchar_10s`" in md or "`V_drop`" in md or "`ICAp_c`" in md, md
        # Capacity is excluded — it has its own block in refinement chat.
        assert "`Capacity`" not in md
        # Table has one σ column per merged cluster.
        n_clusters = len({r["cluster"] for r in merged["dispersion"]["per_cluster"]})
        header_line = next(line for line in md.splitlines() if line.startswith("| Feature"))
        # 1 Feature col + N cluster cols + 1 Mean σ col → header has N+2 column delimiters → N+3 pipes
        assert header_line.count("|") == n_clusters + 3, header_line
    return 0


def test_capacity_block_data_present_for_chat() -> int:
    """RefinementAgent renders a capacity table (μ / σ / CV) from
    dispersion[per_cluster][i]['Capacity']. Verify those fields
    survive the merge."""
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        parent, subs = _synthetic_parent_and_subs(td)
        merged = cluster_merge.merge_subclusters_back(
            parent_xlsx=parent,
            sub_xlsxs_by_parent=subs,
            out_xlsx=td / "merged.xlsx",
        )
        per = merged["dispersion"]["per_cluster"]
        for row in per:
            cap = row.get("Capacity") or {}
            assert isinstance(cap.get("mean"), float), row
            assert isinstance(cap.get("std"), float), row
            # CV may be None when mean ≈ 0, but here capacities are nonzero.
            assert isinstance(cap.get("cv"), float), row
    return 0


def main() -> int:
    tests = [
        test_merge_produces_dispersion_with_curve_features,
        test_merge_dispersion_renders_curve_markdown,
        test_capacity_block_data_present_for_chat,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  [OK] {fn.__name__}")
        except AssertionError as exc:
            print(f"  [FAIL] {fn.__name__}: {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"  [ERROR] {fn.__name__}: {exc!r}")
            traceback.print_exc()
            failed += 1
    print("PASSED" if failed == 0 else f"FAILED ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
