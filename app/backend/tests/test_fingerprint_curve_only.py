"""Pin Feature A — the cluster-fingerprint radar uses curve features only.

`clustering_runner._ELECTROCHEMICAL_COLS` lists the columns that aren't
measurable in the deployed application (OCV / IR / EIS-fit params). The
fingerprint payload must exclude them AND stamp ``feature_kind:
"curve_only"`` so the frontend can show its "curve features only" badge.

We assert both invariants on synthetic data, plus the dispersion block
must still contain those electrochemical columns (the analyst XLSX is
allowed to keep them).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.services import clustering_runner  # noqa: E402
from app.backend.services.clustering_runner import (  # noqa: E402
    _ELECTROCHEMICAL_COLS,
    summarize_clusters,
)


def _synth() -> pd.DataFrame:
    """30-row synthetic dataset with both electrochemical + curve cols."""
    rng = np.random.default_rng(2026)
    return pd.DataFrame({
        "Cell_num": range(1, 31),
        # Electrochemical cols — should be EXCLUDED from fingerprint.
        "OCV": rng.normal(3.7, 0.05, 30),
        "IR": rng.normal(0.08, 0.01, 30),
        "Rs": rng.normal(0.5, 0.02, 30),
        "Rp": rng.normal(1.2, 0.05, 30),
        # Curve-derived cols — should be INCLUDED.
        "Gradchar_10s": rng.normal(0.01, 0.001, 30),
        "ICAp_c": rng.normal(0.5, 0.02, 30),
        "RCVp": rng.normal(0.2, 0.01, 30),
        # Target column — never on the radar.
        "Capacity": rng.normal(1.5, 0.1, 30),
        "MMD_Loss": rng.uniform(0.1, 0.9, 30),
    })


def test_fingerprint_excludes_electrochemical_columns() -> int:
    df = _synth()
    labels = ([0] * 10) + ([1] * 10) + ([2] * 10)
    summary = summarize_clusters(df, labels)
    fp = summary["fingerprint"]
    feats = set(fp.get("features") or [])
    # No electrochemical cols on the radar.
    overlap = feats & _ELECTROCHEMICAL_COLS
    assert not overlap, f"fingerprint must NOT include electrochemical cols, got overlap={overlap}"
    # Capacity is the target, never a fingerprint axis either.
    assert "Capacity" not in feats
    # At least the curve cols we seeded must be present.
    assert {"Gradchar_10s", "ICAp_c", "RCVp"}.issubset(feats), feats
    return 0


def test_fingerprint_stamps_feature_kind() -> int:
    df = _synth()
    labels = ([0] * 15) + ([1] * 15)
    summary = summarize_clusters(df, labels)
    fp = summary["fingerprint"]
    assert fp.get("feature_kind") == "curve_only", fp
    # One values-array per cluster, same length as features.
    for entry in fp.get("clusters") or []:
        vals = entry.get("values") or []
        assert len(vals) == len(fp["features"]), (entry, fp["features"])
    return 0


def test_dispersion_still_includes_electrochemical_columns() -> int:
    """Dispersion table is the offline-analyst surface — keeps the
    electrochemical cols even though the fingerprint hides them."""
    df = _synth()
    labels = ([0] * 15) + ([1] * 15)
    summary = summarize_clusters(df, labels)
    disp = summary.get("dispersion") or {}
    disp_features = set(disp.get("features") or [])
    assert "OCV" in disp_features, disp_features
    assert "IR" in disp_features, disp_features
    return 0


def test_electrochemical_cols_constant_has_expected_members() -> int:
    """Tripwire: if someone adds a column to the deployment-blacklist,
    they must update this list. The intent is to keep the user's
    "OCV/IR aren't measurable in production" carve-out enforced.
    """
    expected = {
        "OCV", "IR",
        "Rs", "CPE1-T", "Rp", "CPE3-T", "R4", "CPE2-T", "R3",
        "Wo-R", "Wo-T",
    }
    assert _ELECTROCHEMICAL_COLS == expected, (
        f"_ELECTROCHEMICAL_COLS changed unexpectedly: "
        f"want={expected}, got={_ELECTROCHEMICAL_COLS}. "
        "If this is intentional update the test."
    )
    return 0


def test_fingerprint_empty_when_no_curve_cols() -> int:
    """Edge: dataset with only electrochemical cols → empty fingerprint,
    still stamped with feature_kind so frontend shows the badge."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "Cell_num": range(1, 21),
        "OCV": rng.normal(3.7, 0.05, 20),
        "IR": rng.normal(0.08, 0.01, 20),
        "Capacity": rng.normal(1.5, 0.1, 20),
        "MMD_Loss": rng.uniform(0.1, 0.9, 20),
    })
    labels = ([0] * 10) + ([1] * 10)
    summary = summarize_clusters(df, labels)
    fp = summary["fingerprint"]
    assert fp.get("feature_kind") == "curve_only"
    assert fp.get("features") == []
    # No clusters added when there are no curve features.
    assert fp.get("clusters") == []
    return 0


def main() -> int:
    tests = [
        test_fingerprint_excludes_electrochemical_columns,
        test_fingerprint_stamps_feature_kind,
        test_dispersion_still_includes_electrochemical_columns,
        test_electrochemical_cols_constant_has_expected_members,
        test_fingerprint_empty_when_no_curve_cols,
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
