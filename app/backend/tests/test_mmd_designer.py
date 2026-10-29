"""Offline unit-style smoke test for the MMD Designer pipeline.

Uses a stubbed LLM (no network) to validate:
- Prompt assembly
- Code extraction
- Sandbox execution (NaN/Inf detection)
- Repair loop semantics
- gpt.py write-back (with .bak backup)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.services import mmd_designer
from app.backend.settings import EVO_GPT_PATH


# ---------- LLM stubs ----------

GOOD_CODE = """```python
import numpy as np
from scipy.spatial.distance import cdist

def calculate_mmd_v9(xs: np.ndarray, xt: np.ndarray, fix_sigma: float = 0,
                     kernel_mul: float = 2.0, kernel_num: int = 5) -> float:
    if xs.size == 0 or xt.size == 0:
        return 0.0
    n = xs.shape[0]
    m = xt.shape[0]
    x = np.vstack([xs, xt])
    L2 = cdist(x, x, metric='euclidean') ** 2
    if fix_sigma == 0:
        bw = max(np.sum(L2) / max((n + m) ** 2 - (n + m), 1), 1e-8)
    else:
        bw = max(float(fix_sigma), 1e-8)
    bw /= max(kernel_mul ** (kernel_num // 2), 1e-8)
    bandwidths = [max(bw * (kernel_mul ** i), 1e-8) for i in range(kernel_num)]
    K = sum(np.exp(-L2 / b) for b in bandwidths)
    ss = K[:n, :n].mean()
    tt = K[n:, n:].mean()
    st = K[:n, n:].mean()
    val = float(ss + tt - 2 * st)
    if not np.isfinite(val):
        return 0.0
    return val
```"""

NAN_CODE = """```python
import numpy as np

def calculate_mmd_v9(xs, xt, fix_sigma=0, kernel_mul=2.0, kernel_num=5):
    return float('nan')
```"""

EXC_CODE = """```python
import numpy as np

def calculate_mmd_v9(xs, xt, fix_sigma=0, kernel_mul=2.0, kernel_num=5):
    raise ValueError("simulated explosion")
```"""

NO_CODE = "Sorry, I don't think there is a good MMD function for this task."


def make_scripted_chat(responses: List[str]):
    """Return a chat() compatible callable that yields scripted responses in order."""
    state = {"i": 0}

    def _chat(messages: List[Dict[str, str]], temperature: float = 0.7, **_kw: Any) -> str:
        i = state["i"]
        if i >= len(responses):
            raise RuntimeError("scripted chat exhausted")
        state["i"] = i + 1
        return responses[i]

    return _chat


# ---------- Tests ----------

def t_extract_code() -> None:
    code = mmd_designer.extract_code(GOOD_CODE)
    assert code is not None and "calculate_mmd_v9" in code, "extraction failed"
    assert mmd_designer.find_function_name(code) == "calculate_mmd_v9"
    print("  [OK] extract_code + find_function_name")


def t_prompt_assembly() -> None:
    prompt = mmd_designer.build_initial_prompt(version=7)
    assert "calculate_mmd_v7" in prompt
    assert "Baseline (seed) implementation" in prompt
    assert "Hard requirements" in prompt
    print("  [OK] build_initial_prompt")


def t_full_pipeline_good_on_first_try() -> None:
    chat = make_scripted_chat([GOOD_CODE])
    result = mmd_designer.design_mmd(
        max_attempts=5,
        sandbox_cells=3,
        version=9,
        seed=7,
        write_back=False,
        llm_chat=chat,
    )
    assert result.passed, f"expected pass, got {result.last_error}"
    assert result.attempts == 1
    assert len(result.sample_losses) == 3
    assert all(v != 0.0 for v in result.sample_losses.values()), "losses should differ"
    print(f"  [OK] one-shot success in {result.attempts} attempt; "
          f"losses={ {k: f'{v:.3g}' for k, v in result.sample_losses.items()} }")


def t_nan_then_recovery() -> None:
    chat = make_scripted_chat([NAN_CODE, EXC_CODE, GOOD_CODE])
    result = mmd_designer.design_mmd(
        max_attempts=5,
        sandbox_cells=3,
        version=9,
        seed=11,
        write_back=False,
        llm_chat=chat,
    )
    assert result.passed, f"expected pass after recovery, last_error={result.last_error}"
    assert result.attempts == 3, f"expected 3 attempts, got {result.attempts}"
    assert not result.history[0].passed and "Non-finite" in (result.history[0].error or "")
    assert not result.history[1].passed and "simulated explosion" in (result.history[1].error or "")
    assert result.history[2].passed
    print("  [OK] NaN → exception → repaired recovery in 3 attempts")


def t_repair_loop_exhausts() -> None:
    chat = make_scripted_chat([NAN_CODE] * 3)
    result = mmd_designer.design_mmd(
        max_attempts=3,
        sandbox_cells=3,
        version=9,
        seed=13,
        write_back=False,
        llm_chat=chat,
    )
    assert not result.passed
    assert result.attempts == 3
    assert all(not a.passed for a in result.history)
    print("  [OK] repair loop exhaustion handled cleanly")


def t_no_code_response() -> None:
    chat = make_scripted_chat([NO_CODE, GOOD_CODE])
    result = mmd_designer.design_mmd(
        max_attempts=5,
        sandbox_cells=3,
        version=9,
        seed=17,
        write_back=False,
        llm_chat=chat,
    )
    assert result.passed
    assert result.attempts == 2
    assert "code block" in (result.history[0].error or "").lower()
    print("  [OK] graceful handling of missing code block + recovery")


def t_write_back_creates_bak() -> None:
    chat = make_scripted_chat([GOOD_CODE])
    original_existed = EVO_GPT_PATH.exists()
    original_text = EVO_GPT_PATH.read_text(encoding="utf-8") if original_existed else None

    result = mmd_designer.design_mmd(
        max_attempts=3,
        sandbox_cells=3,
        version=9,
        seed=19,
        write_back=True,
        llm_chat=chat,
    )
    try:
        assert result.passed
        assert EVO_GPT_PATH.exists()
        assert "calculate_mmd_v9" in EVO_GPT_PATH.read_text(encoding="utf-8")
        if original_existed:
            bak = EVO_GPT_PATH.with_suffix(EVO_GPT_PATH.suffix + ".bak")
            assert bak.exists(), "expected .bak backup"
        print("  [OK] write_back created gpt.py + .bak backup")
    finally:
        if original_existed and original_text is not None:
            EVO_GPT_PATH.write_text(original_text, encoding="utf-8")
        bak = EVO_GPT_PATH.with_suffix(EVO_GPT_PATH.suffix + ".bak")
        if bak.exists():
            bak.unlink()


def main() -> int:
    print("== test_mmd_designer ==")
    tests = [
        t_extract_code,
        t_prompt_assembly,
        t_full_pipeline_good_on_first_try,
        t_nan_then_recovery,
        t_repair_loop_exhausts,
        t_no_code_response,
        t_write_back_creates_bak,
    ]
    failures: List[str] = []
    for t in tests:
        try:
            t()
        except Exception as exc:
            failures.append(f"{t.__name__}: {exc}")
            print(f"  [FAIL] {t.__name__}: {exc}")
    print()
    if failures:
        print(f"FAILED: {len(failures)}/{len(tests)}")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"PASSED: {len(tests)}/{len(tests)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
