"""Unit test: the May-30 deprecation guardrail.

Verifies that:
  * The known-sunsetting Qwen aliases are flagged by `is_deprecated_model`.
  * `_promote_if_deprecated` rewrites them to PINNED_MODEL.
  * Models that are still alive (qwen3.5-plus, qwen3-max, deepseek-v3, ...)
    pass through unchanged.
  * `DEFAULT_PROBE_MODELS` contains no deprecated entries — important so
    a future `/api/settings/llm/probe` call can't reintroduce a stale
    alias by accident.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.services import llm_client


DEPRECATED = [
    "qwen-turbo-latest",
    "qwen-plus-latest",
    "qwen-flash",
    "qwen3-coder-plus",
    "qwen-max-latest",
    "qwen-turbo",
    "qwen-turbo-2024-09-19",
]

ALIVE = [
    "qwen3.6-plus",
    "qwen3.6-plus-2026-04-02",
    "qwen3.5-plus",
    "qwen3.5-flash",
    "qwen3-max",
    "qwen3-max-2026-01-23",
    "qwen3.7-max",
    "deepseek-v3",
    "deepseek-v3.2",
]


def main() -> int:
    failed = 0

    for m in DEPRECATED:
        if not llm_client.is_deprecated_model(m):
            print(f"  [FAIL] {m!r} should be flagged as deprecated")
            failed += 1
            continue
        promoted = llm_client._promote_if_deprecated(m)
        if promoted != llm_client.PINNED_MODEL:
            print(f"  [FAIL] {m!r} promoted to {promoted!r}, want {llm_client.PINNED_MODEL!r}")
            failed += 1
        else:
            print(f"  [OK] {m!r} → {promoted!r}")

    for m in ALIVE:
        if llm_client.is_deprecated_model(m):
            print(f"  [FAIL] {m!r} incorrectly flagged as deprecated")
            failed += 1
            continue
        promoted = llm_client._promote_if_deprecated(m)
        if promoted != m:
            print(f"  [FAIL] {m!r} promoted to {promoted!r}, want unchanged")
            failed += 1
        else:
            print(f"  [OK] {m!r} unchanged")

    # Probe candidate list must be clean.
    for m in llm_client.DEFAULT_PROBE_MODELS:
        if llm_client.is_deprecated_model(m):
            print(f"  [FAIL] DEFAULT_PROBE_MODELS contains deprecated {m!r}")
            failed += 1
    if not any(c == llm_client.PINNED_MODEL for c in llm_client.DEFAULT_PROBE_MODELS):
        print(f"  [FAIL] PINNED_MODEL {llm_client.PINNED_MODEL!r} missing from probe candidates")
        failed += 1
    else:
        print(f"  [OK] probe candidates: {llm_client.DEFAULT_PROBE_MODELS}")

    cfg = llm_client.ResolvedConfig(
        model="qwen3.6-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="test",
        temperature=1.0,
        source="builtin:qwen",
    )
    kwargs = {"model": "qwen3.6-plus", "messages": []}
    thinking = llm_client._apply_extra_body(
        kwargs,
        model_name="qwen3.6-plus",
        cfg=cfg,
        enable_thinking=False,
    )
    if thinking is not False or kwargs.get("extra_body") != {"enable_thinking": False}:
        print(f"  [FAIL] qwen thinking toggle not applied: thinking={thinking}, kwargs={kwargs}")
        failed += 1
    else:
        print("  [OK] qwen thinking toggle applies enable_thinking=False")

    deepseek_kwargs = {"model": "deepseek-v3", "messages": []}
    llm_client._apply_extra_body(
        deepseek_kwargs,
        model_name="deepseek-v3",
        cfg=llm_client.ResolvedConfig(
            model="deepseek-v3",
            base_url="https://api.deepseek.com/v1",
            api_key="test",
            temperature=1.0,
            source="builtin:deepseek",
        ),
        enable_thinking=False,
    )
    if "extra_body" in deepseek_kwargs:
        print(f"  [FAIL] deepseek should not receive qwen extra_body: {deepseek_kwargs}")
        failed += 1
    else:
        print("  [OK] deepseek does not receive qwen thinking toggle")

    print("PASSED" if failed == 0 else f"FAILED ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
