"""Unit test: RouterAgent extracts cluster-stage knobs (min_k/max_k,
pop_size, max_iter, feedback_metric) from natural-language utterances,
in both English and Chinese.

Importantly, params must only attach when the message mentions
cluster-context words — a plain "cluster into 6 groups" should NOT
get spurious pop_size or feedback_metric values.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.agents.router_agent import _extract_cluster_params, _heuristic


# (text, want_subset_in_decision_cluster_params)
CASES: list[tuple[str, dict]] = [
    (
        "Cluster with k=3..8, pop 100, iter 60, feedback dbi.",
        {"min_k": 3, "max_k": 8, "pop_size": 100, "max_iter": 60, "feedback_metric": "dbi"},
    ),
    (
        "聚类 k=4 到 8 种群=120 迭代=50 反馈=方差",
        {"min_k": 4, "max_k": 8, "pop_size": 120, "max_iter": 50, "feedback_metric": "variance"},
    ),
    (
        "Run clustering, use variance as feedback",
        {"feedback_metric": "variance"},
    ),
    (
        "Run clustering with OCV=1 IR=1 DDIA=2",
        {"feature_weights": {"ocv": 1.0, "ir": 1.0, "ddia": 2.0}},
    ),
    (
        "聚类 权重 OCV 1 IR 1 DDI 2",
        {"feature_weights": {"ocv": 1.0, "ir": 1.0, "ddia": 2.0}},
    ),
    (
        "Run clustering",
        {},  # plain — no params
    ),
    (
        "Cluster into 6 groups",
        {},  # target_k is handled separately; no cluster_params
    ),
]


def main() -> int:
    failed = 0
    for text, want in CASES:
        h = _heuristic(text)
        got = h.get("cluster_params") or {}
        if not want:
            ok = not got
        else:
            ok = all(got.get(k) == v for k, v in want.items())
        flag = "OK" if ok else "FAIL"
        print(f"  [{flag}] {text!r}\n        cluster_params={got} (want subset {want})")
        if not ok:
            failed += 1

    # direct extractor: k range with hyphen
    if _extract_cluster_params("k=5-9 pop=50 iter=40").get("min_k") != 5:
        print("  [FAIL] hyphen k-range not parsed")
        failed += 1
    else:
        print("  [OK] hyphen k-range parsed")

    # rejection: out-of-range pop_size
    got = _extract_cluster_params("pop=1 iter=99999")
    if got.get("pop_size") is not None:
        print(f"  [FAIL] pop=1 should have been rejected; got {got}")
        failed += 1
    else:
        print("  [OK] rejected out-of-range pop_size")

    # CN context: 反馈=综合 → combined
    got = _extract_cluster_params("使用 反馈=综合")
    if got.get("feedback_metric") != "combined":
        print(f"  [FAIL] CN '反馈=综合' → {got}")
        failed += 1
    else:
        print("  [OK] CN '反馈=综合' → combined")

    print("PASSED" if failed == 0 else f"FAILED ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
