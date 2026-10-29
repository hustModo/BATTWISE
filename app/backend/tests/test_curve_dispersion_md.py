"""Unit tests for clustering_runner.format_curve_dispersion_markdown.

The post-cluster chat message now appends a markdown block with the
top-N noisiest curve features and a per-cluster σ table. These tests
pin the formatter so the chat output (and downstream MD rendering)
stays stable.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.services.clustering_runner import (  # noqa: E402
    format_curve_dispersion_markdown,
)


def _mk_dispersion(features, per_cluster, feature_mean_std):
    return {
        "features": features,
        "per_cluster": per_cluster,
        "feature_mean_std": feature_mean_std,
        "capacity_aggregate": {},
    }


def test_happy_path_emits_table_and_headline() -> int:
    disp = _mk_dispersion(
        features=["Capacity", "V_init_mean", "V_drop", "IR"],
        per_cluster=[
            {"cluster": 0, "size": 24,
             "Capacity": {"mean": 1.0, "std": 0.01, "cv": 0.01},
             "V_init_mean": {"mean": 3.7, "std": 0.020, "cv": 0.005},
             "V_drop": {"mean": 0.3, "std": 0.080, "cv": 0.26},
             "IR": {"mean": 25.0, "std": 0.50, "cv": 0.02}},
            {"cluster": 1, "size": 31,
             "Capacity": {"mean": 0.9, "std": 0.02, "cv": 0.02},
             "V_init_mean": {"mean": 3.6, "std": 0.040, "cv": 0.011},
             "V_drop": {"mean": 0.4, "std": 0.110, "cv": 0.27},
             "IR": {"mean": 26.0, "std": 0.60, "cv": 0.023}},
        ],
        feature_mean_std={
            "Capacity": 0.015, "V_init_mean": 0.030,
            "V_drop": 0.095, "IR": 0.55,
        },
    )
    md = format_curve_dispersion_markdown(disp, top_n=3)
    assert md is not None
    # Capacity must NOT be in the table — capacity has its own block.
    assert "`Capacity`" not in md, md
    # Headline lists the noisiest curve feature first.
    assert "**Curve-feature dispersion**" in md
    # Top-1 by mean_of_std is IR (0.55), then V_drop (0.095), V_init_mean (0.030).
    headline_line = md.splitlines()[0]
    assert headline_line.index("`IR`") < headline_line.index("`V_drop`") < headline_line.index("`V_init_mean`")
    # Two cluster columns + Mean σ + Feature column = 4 cols → 3 pipes in header.
    table_header = next(line for line in md.splitlines() if line.startswith("| Feature"))
    assert table_header.count("|") == 5  # 4 cols => 5 pipes
    assert "Cluster 0 σ" in table_header and "Cluster 1 σ" in table_header
    # σ values for IR row reflect per-cluster std.
    ir_row = next(line for line in md.splitlines() if line.startswith("| `IR`"))
    assert "0.5" in ir_row and "0.6" in ir_row
    return 0


def test_top_n_limits_table_rows() -> int:
    disp = _mk_dispersion(
        features=["Capacity"] + [f"feat_{i}" for i in range(10)],
        per_cluster=[{
            "cluster": 0, "size": 10,
            **{f"feat_{i}": {"mean": 1.0, "std": float(i + 1), "cv": 0.1} for i in range(10)},
            "Capacity": {"mean": 1.0, "std": 0.01, "cv": 0.01},
        }],
        feature_mean_std={**{f"feat_{i}": float(i + 1) for i in range(10)}, "Capacity": 0.01},
    )
    md = format_curve_dispersion_markdown(disp, top_n=3)
    assert md is not None
    # 2 header rows + 3 data rows = 5 lines under the headline.
    table_lines = [l for l in md.splitlines() if l.startswith("|")]
    assert len(table_lines) == 2 + 3, f"got {len(table_lines)}: {md}"
    # Top 3 by σ should be feat_9, feat_8, feat_7.
    assert "`feat_9`" in md
    assert "`feat_8`" in md
    assert "`feat_7`" in md
    assert "`feat_0`" not in md
    return 0


def test_returns_none_when_no_curve_features() -> int:
    disp = _mk_dispersion(
        features=["Capacity"],
        per_cluster=[{"cluster": 0, "size": 5,
                      "Capacity": {"mean": 1.0, "std": 0.01, "cv": 0.01}}],
        feature_mean_std={"Capacity": 0.01},
    )
    md = format_curve_dispersion_markdown(disp)
    assert md is None
    return 0


def test_returns_none_when_no_mean_std() -> int:
    # Curve features listed but mean-of-std missing → can't rank → bail.
    disp = _mk_dispersion(
        features=["Capacity", "feat_a"],
        per_cluster=[{"cluster": 0, "size": 5,
                      "Capacity": {"mean": 1.0, "std": 0.01, "cv": 0.01},
                      "feat_a": {"mean": 1.0, "std": None, "cv": None}}],
        feature_mean_std={"Capacity": 0.01},  # no feat_a here
    )
    md = format_curve_dispersion_markdown(disp)
    assert md is None
    return 0


def test_missing_per_cluster_std_renders_em_dash() -> int:
    disp = _mk_dispersion(
        features=["Capacity", "feat_x"],
        per_cluster=[
            {"cluster": 0, "size": 1,
             "Capacity": {"mean": 1.0, "std": 0.01, "cv": 0.01},
             "feat_x": {"mean": 1.0, "std": None, "cv": None}},
            {"cluster": 1, "size": 4,
             "Capacity": {"mean": 1.0, "std": 0.01, "cv": 0.01},
             "feat_x": {"mean": 1.0, "std": 0.5, "cv": 0.5}},
        ],
        feature_mean_std={"feat_x": 0.5, "Capacity": 0.01},
    )
    md = format_curve_dispersion_markdown(disp)
    assert md is not None
    feat_row = next(line for line in md.splitlines() if line.startswith("| `feat_x`"))
    # Cluster 0 has std=None → em dash; cluster 1 has 0.5.
    assert "—" in feat_row
    assert "0.5" in feat_row
    return 0


def main() -> int:
    tests = [
        test_happy_path_emits_table_and_headline,
        test_top_n_limits_table_rows,
        test_returns_none_when_no_curve_features,
        test_returns_none_when_no_mean_std,
        test_missing_per_cluster_std_renders_em_dash,
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
