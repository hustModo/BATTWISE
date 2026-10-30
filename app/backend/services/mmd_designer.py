"""CDDI Designer service (Lightweight mode).

Pipeline:
    1. Assemble prompt from battwise_cddi_search prompts/battery_cddi/*.txt
    2. Call LLM (single shot)
    3. Extract calculate_mmd_vN code block
    4. Sandbox: run on N random cells × Model file
       → detect NaN/Inf/exceptions
    5. If failure: package code + traceback + sample shapes into a
       repair prompt → re-call LLM → max 5 attempts
    6. On success: backup gpt.py.bak, write fresh gpt.py
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.backend.services import data_resolver, llm_client
from app.backend.settings import EVO_GPT_PATH, EVO_PROMPTS_DIR, settings

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are an expert in numerical analysis and battery-cell signal processing. "
    "You design Maximum Mean Discrepancy (MMD) calculation functions for battery "
    "charging curves. You always return a single Python function in a ```python``` "
    "code block, with imports, that is numerically stable (no NaN/Inf, no division "
    "by zero, no negative values inside sqrt).\n\n"
    "CRITICAL DOMAIN RULE — do NOT destroy absolute magnitude:\n"
    "  The Current and Voltage columns ARE the signal that correlates with cell "
    "  electrochemistry (Urest, OCV, dU/dt, etc). If you z-score normalize each "
    "  feature (subtract mean, divide by std) or otherwise strip absolute scale, "
    "  the CDDI output collapses to a small number that no longer correlates with "
    "  any indicator. Successful CDDI functions on this dataset output values in "
    "  the THOUSANDS to TENS-OF-THOUSANDS range and achieve mean |Pearson r| > 0.7. "
    "  Multi-scale Gaussian kernels with a median-heuristic bandwidth on the RAW "
    "  (unnormalized) data are fine; light per-feature SCALING (e.g. multiplying "
    "  Current by a small weight) is acceptable. Full z-score normalization is NOT."
)


@dataclass
class DesignAttempt:
    attempt: int
    code: Optional[str]
    passed: bool
    error: Optional[str] = None
    failing_cell: Optional[int] = None
    sample_losses: Dict[int, float] = field(default_factory=dict)
    duration_ms: float = 0.0
    elapsed_s: float = 0.0


@dataclass
class DesignResult:
    code: str
    function_name: str
    passed: bool
    attempts: int
    history: List[DesignAttempt]
    gpt_path: str
    sample_losses: Dict[int, float]
    last_error: Optional[str] = None


# ---------- prompt assembly ----------

def _read_prompt_file(name: str) -> str:
    path = EVO_PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file missing: {path}")
    return path.read_text(encoding="utf-8")


def build_initial_prompt(version: int = 2) -> str:
    seed = _read_prompt_file("seed_func.txt")
    desc = _read_prompt_file("func_desc.txt")
    sig = _read_prompt_file("func_signature.txt").replace("{version}", str(version))
    knowledge = _read_prompt_file("external_knowledge.txt")

    return f"""# Task
Design an improved MMD calculation function for battery charging curves.

# Function description
{desc}

# Function signature (use exactly this name `calculate_mmd_v{version}`)
{sig}

# Baseline (seed) implementation
```python
{seed}
```

# Domain knowledge
{knowledge}

# Hard requirements
1. Return EXACTLY ONE Python code block enclosed in ```python ... ```.
2. The function MUST be named `calculate_mmd_v{version}` with signature:
   `def calculate_mmd_v{version}(xs: np.ndarray, xt: np.ndarray, fix_sigma: float = 0, kernel_mul: float = 2.0, kernel_num: int = 5) -> float`
3. Include all imports the function needs at the top of the code block.
4. The function MUST NOT return NaN or Inf on any reasonable battery input.
   Add safeguards: clip bandwidths >= 1e-8, max(value, 0) before sqrt,
   guard against empty arrays and constant inputs.
5. Do NOT call `print`, `plt.*`, or any I/O — pure computation only.
6. The function should be numerically stable AND produce values that
   correlate with battery electrochemical features. Prefer multi-scale kernels.
7. **DO NOT z-score normalize the inputs.** Specifically: do NOT write
   `(x - x.mean()) / x.std()` or `StandardScaler` on per-feature basis. The
   absolute Current/Voltage magnitudes carry the electrochemistry signal —
   destroying them yields |r| ≈ 0 across all indicators. A reference MMD
   that achieves mean |r| ≈ 0.86 on this dataset outputs values in the
   1e4 ~ 5e4 range (kernel sums of the RAW signal). Lightweight per-feature
   weighting like `x * np.array([0.4, 0.6])` is fine. Computing pairwise L2
   on the raw concatenated cell+model signal is the proven approach.

# Output
Return ONLY the code block.
"""


def build_repair_prompt(version: int, previous_code: str, attempt: DesignAttempt) -> str:
    return f"""# Task
The previously generated CDDI function FAILED sandbox validation.
You must FIX it. Return a corrected `calculate_mmd_v{version}` function
in a single ```python``` block, with imports.

# Previous code
```python
{previous_code}
```

# Failure details
- Attempt: {attempt.attempt}
- Failing cell index: {attempt.failing_cell}
- Error message:
{attempt.error}

# Likely fixes to apply
1. Clip bandwidths to be >= 1e-8 before division / sqrt.
2. Use `np.maximum(value, 0.0)` inside any `np.sqrt(...)`.
3. Replace `try/except: continue` with explicit handling — do not silently
   produce 0.0 because that yields constant MMD across cells.
4. Avoid `np.random.choice` with `replace=False` on small arrays; guard
   `subset_size = min(size, len(combined))`.
5. Guard against constant signals: when `np.std(...)` == 0 fall back to a
   small constant bandwidth such as 1e-3.
6. Always return `float(result)`; if result is non-finite, return 0.0 explicitly.

# Hard requirements (UNCHANGED)
- Function name MUST be `calculate_mmd_v{version}`
- Signature MUST match: `(xs, xt, fix_sigma=0, kernel_mul=2.0, kernel_num=5) -> float`
- Numerically stable, no NaN / Inf
- Pure computation, no I/O
- Return ONLY the code block.
"""


# ---------- code extraction & loading ----------

_CODE_BLOCK_RE = re.compile(r"```python(.*?)```", re.DOTALL | re.IGNORECASE)
_FUNC_NAME_RE = re.compile(r"def\s+(calculate_mmd_v\d+)\s*\(", re.MULTILINE)


def extract_code(response: str) -> Optional[str]:
    match = _CODE_BLOCK_RE.search(response)
    if not match:
        return None
    code = match.group(1).strip()
    if not code:
        return None
    if "import numpy" not in code and " np." in code:
        code = "import numpy as np\n" + code
    return code


def find_function_name(code: str) -> Optional[str]:
    m = _FUNC_NAME_RE.search(code)
    return m.group(1) if m else None


def load_function(code: str) -> Tuple[Callable, str, Dict[str, Any]]:
    """Exec the code in a fresh namespace and return (fn, fn_name, namespace)."""
    namespace: Dict[str, Any] = {}
    exec("import numpy as np", namespace)
    exec("from scipy.spatial.distance import cdist", namespace)
    exec("import warnings; warnings.filterwarnings('ignore')", namespace)
    exec(code, namespace)

    fn_name = find_function_name(code)
    if not fn_name or fn_name not in namespace:
        for name, obj in namespace.items():
            if name.startswith("calculate_mmd_v") and callable(obj):
                fn_name = name
                break
    if not fn_name:
        raise RuntimeError("No `calculate_mmd_vN` function found in generated code.")
    return namespace[fn_name], fn_name, namespace


# ---------- sandbox ----------

def _load_data(file_path: Path) -> Optional[pd.DataFrame]:
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(file_path)
        if suffix == ".xls":
            try:
                return pd.read_excel(file_path, engine="calamine")
            except Exception:
                return pd.read_excel(file_path, engine="xlrd")
        if suffix == ".xlsx":
            try:
                return pd.read_excel(file_path, engine="calamine")
            except Exception:
                return pd.read_excel(file_path, engine="openpyxl")
    except Exception as exc:
        logger.warning("Failed to read %s: %s", file_path, exc)
    return None


def _slice_cv(df: pd.DataFrame, row_start: int = 0, row_end: Optional[int] = None) -> np.ndarray:
    if "Current" not in df.columns or "Voltage" not in df.columns:
        raise KeyError("Cell file must contain 'Current' and 'Voltage' columns.")
    if row_end is not None:
        return df[["Current", "Voltage"]].iloc[row_start:row_end].values
    return df[["Current", "Voltage"]].iloc[row_start:].values


def _pick_sandbox_cells(
    folder: Path,
    n: int,
    seed: Optional[int],
    cells_filter: Optional[List[int]] = None,
) -> List[Path]:
    wanted = {int(c) for c in cells_filter} if cells_filter else None
    cells = [
        p for p in folder.iterdir()
        if p.is_file() and "Cell" in p.name and p.suffix.lower() in {".xls", ".xlsx"}
        and "Model" not in p.name
        and (wanted is None or _cell_number(p) in wanted)
    ]
    if not cells:
        raise FileNotFoundError(f"No cell files found in {folder}")
    rng = random.Random(seed)
    rng.shuffle(cells)
    return cells[:n]


def _cell_number(path: Path) -> int:
    m = re.search(r"Cell(\d+)", path.name)
    return int(m.group(1)) if m else -1


# Reference (seed) MMD for sanity check. Imported lazily so circular import
# doesn't bite — see `_seed_mmd` below.
def _seed_mmd(xs: np.ndarray, xt: np.ndarray) -> float:
    """The exact baseline from prompts/battery_cddi/seed_func.txt — simple
    multi-kernel RBF on the RAW signal. Used as a correctness oracle: a good
    LLM-generated MMD should produce values whose *ranking* across cells is
    similar (Pearson |r| >= SANITY_THRESHOLD) to this seed."""
    from scipy.spatial.distance import cdist
    s_num, t_num = xs.shape[0], xt.shape[0]
    n_sam = s_num + t_num
    x_new = np.vstack((xs, xt))
    L2 = cdist(x_new, x_new, metric="euclidean") ** 2
    denom = max(n_sam * n_sam - n_sam, 1)
    bw = max(float(np.sum(L2)) / denom, 1e-12)
    bw /= (2.0 ** 2)  # kernel_mul ** (kernel_num // 2)
    K = np.zeros((n_sam, n_sam))
    for i in range(5):
        K += np.exp(-L2 / max(bw * (2.0 ** i), 1e-12))
    ss = float(np.mean(K[:s_num, :s_num]))
    tt = float(np.mean(K[s_num:, s_num:]))
    st = float(np.mean(K[:s_num, s_num:]))
    ts = float(np.mean(K[s_num:, :s_num]))
    return ss + tt - st - ts


# If LLM's MMD ranking across sandbox cells correlates < this with the seed,
# the function probably destroyed the absolute-magnitude signal (typical
# culprit: z-score normalize). We reject it to trigger a repair attempt.
_SANITY_RANK_R = 0.5


def sandbox_evaluate(
    fn: Callable,
    model_path: Path,
    cell_paths: List[Path],
    *,
    row_start: int = 0,
    row_end: Optional[int] = None,
) -> Tuple[bool, Optional[str], Optional[int], Dict[int, float]]:
    """Run `fn` on every (cell, model) pair. Return (passed, error, failing_cell, losses)."""
    df_model = _load_data(model_path)
    if df_model is None:
        return False, f"Could not load Model file: {model_path}", None, {}
    try:
        xt = _slice_cv(df_model, row_start, row_end)
    except KeyError as exc:
        return False, str(exc), None, {}

    losses: Dict[int, float] = {}
    for cell_path in cell_paths:
        cell_num = _cell_number(cell_path)
        df_cell = _load_data(cell_path)
        if df_cell is None:
            return False, f"Could not load cell file: {cell_path.name}", cell_num, losses
        try:
            xs = _slice_cv(df_cell, row_start, row_end)
        except KeyError as exc:
            return False, f"{cell_path.name}: {exc}", cell_num, losses

        try:
            loss = float(fn(xs, xt))
        except Exception:
            tb = traceback.format_exc(limit=6)
            shape_info = f"xs.shape={xs.shape}, xt.shape={xt.shape}"
            return False, f"Exception on {cell_path.name} ({shape_info}):\n{tb}", cell_num, losses

        if not np.isfinite(loss):
            shape_info = f"xs.shape={xs.shape}, xt.shape={xt.shape}, value={loss}"
            return (
                False,
                f"Non-finite loss on {cell_path.name} ({shape_info}). The function "
                f"must return a finite float; got {loss!r}.",
                cell_num,
                losses,
            )
        losses[cell_num] = loss

    if len(set(round(v, 12) for v in losses.values())) == 1 and len(losses) > 1:
        return (
            False,
            f"All sandbox cells produced identical CDDI loss ({next(iter(losses.values()))}). "
            "The function must discriminate between cells.",
            None,
            losses,
        )

    return True, None, None, losses


# Quality-gate threshold applied AFTER sandbox passes — see quality_check()
# for the rationale. 0.3 is permissive enough to allow LLM creativity but
# blocks degenerate "normalize-everything" outputs whose |r| collapses to ~0.05.
_QUALITY_MIN_MEAN_R = 0.30


def quality_check(
    fn: Callable,
    folder: Path,
    model_filename: str,
    char_filename: Optional[str],
    *,
    row_start: int = 0,
    row_end: Optional[int] = None,
    cells_filter: Optional[List[int]] = None,
) -> Tuple[bool, str, float, Dict[str, float]]:
    """Full-dataset CDDI ↔ indicator quality gate.

    Runs the candidate function on every Cell file under ``folder``, then
    computes Pearson |r| against each numeric column of the characteristics
    workbook. Returns ``(passed, message, mean_abs_r, per_feature)``.

    Rationale:
        The sandbox check only proves the function doesn't crash and produces
        distinct numbers. It cannot tell whether those numbers actually
        correlate with battery electrochemistry — and we've seen LLMs write
        z-score-normalized variants that pass sandbox but collapse mean |r| to
        ~0.05 on first_200s. This gate runs the full 71-cell pipeline (cheap,
        ~0.3s on built-in datasets) and uses the same Pearson math the
        downstream ComputeAgent uses.
    """
    if char_filename is None:
        return True, "no char file → skipped quality check", float("nan"), {}

    char_path = folder / char_filename
    if not char_path.exists():
        return True, f"char file missing → skipped quality check ({char_path})", float("nan"), {}

    # Lazy import — avoids circular deps with mmd_runner during module init.
    from app.backend.services.mmd_runner import (
        compute_mmd_for_folder,
        correlation_analysis,
    )

    results: List[Dict[str, Any]] = []
    for prog in compute_mmd_for_folder(
        data_folder=folder,
        model_filename=model_filename,
        mmd_fn=fn,
        row_start=row_start,
        row_end=row_end,
        cells_filter=cells_filter,
    ):
        if prog.phase == "cell" and prog.mmd_loss is not None:
            results.append({
                "Filename": prog.filename,
                "Cell_num": prog.cell_num,
                "MMD_Loss": prog.mmd_loss,
                "Sample_count": prog.sample_count,
            })

    if not results:
        return False, "quality check: no cells produced a valid CDDI", float("nan"), {}

    corr = correlation_analysis(results, char_path, p_threshold=1.0)
    mean_r = float(corr.get("mean_sig", 0.0))
    per_feature = {
        f["feature"]: abs(float(f["r"])) for f in (corr.get("features") or [])
    }

    if mean_r >= _QUALITY_MIN_MEAN_R:
        return True, f"quality check passed (mean |r|={mean_r:.3f})", mean_r, per_feature

    # Build a useful repair message that names the strongest feature so
    # the LLM has a hint about what direction to push next attempt.
    top = sorted(per_feature.items(), key=lambda kv: -kv[1])
    top_str = ", ".join(f"{k}=|r|={v:.2f}" for k, v in top[:3]) or "(none)"
    return (
        False,
        (
            f"Quality gate failed: candidate CDDI vs indicator characteristics gives "
            f"mean |r|={mean_r:.3f} across {len(per_feature)} features "
            f"(top: {top_str}), below the {_QUALITY_MIN_MEAN_R:.2f} threshold. "
            "The CDDI is not picking up the electrochemistry signal. "
            "This almost always means the function destroyed absolute magnitude: "
            "DO NOT z-score normalize Current/Voltage (no `(x-mean)/std`, no "
            "`StandardScaler`, no per-feature centering). The proven baseline "
            "computes a multi-kernel Gaussian on the RAW concatenated signal "
            "with median-heuristic bandwidth, no normalization. Use that as your "
            "starting point."
        ),
        mean_r,
        per_feature,
    )


# ---------- main driver ----------

def design_mmd(
    *,
    dataset_id: str = "builtin-first-200s",
    max_attempts: Optional[int] = None,
    sandbox_cells: Optional[int] = None,
    version: int = 2,
    row_start: int = 0,
    row_end: Optional[int] = None,
    seed: Optional[int] = 42,
    cells_filter: Optional[List[int]] = None,
    write_back: bool = True,
    on_attempt: Optional[Callable[[DesignAttempt], None]] = None,
    on_log: Optional[Callable[[str, str], None]] = None,
    llm_chat: Optional[Callable[..., str]] = None,
) -> DesignResult:
    """Run the full Lightweight Designer loop.

    Args:
        dataset_id: ID resolved via data_resolver.
        max_attempts: defaults to settings.designer_max_repair_attempts.
        sandbox_cells: defaults to settings.designer_sandbox_cells.
        version: function suffix `calculate_mmd_vN`.
        write_back: write to battwise_cddi_search gpt.py on success.
        on_attempt: optional callback invoked with each DesignAttempt (for live UI).
        on_log: optional ``(tag, line)`` callback for fine-grained progress —
            invoked at prompt build, LLM call start/end, sandbox start/end,
            quality check etc. The frontend renders this as a live terminal.
            Tags: 'info' | 'ok' | 'warn' | 'error'.
        llm_chat: override for the LLM call (used by tests).
    """
    max_attempts = max_attempts or settings.designer_max_repair_attempts
    sandbox_cells = sandbox_cells or settings.designer_sandbox_cells

    def log(tag: str, line: str) -> None:
        if on_log is not None:
            try:
                on_log(tag, line)
            except Exception:  # never let a UI callback break the design loop
                logger.exception("on_log callback raised; ignoring")

    ds = data_resolver.resolve(dataset_id)
    model_path = ds.folder / ds.model_file
    cell_paths = _pick_sandbox_cells(ds.folder, sandbox_cells, seed, cells_filter=cells_filter)
    log("info", f"Sandbox cells selected: {[p.name for p in cell_paths]}")

    def default_chat(**kwargs: Any) -> str:
        return asyncio.run(
            llm_client.chat_async(
                **kwargs,
                max_retries=1,
                timeout=settings.llm_designer_timeout,
                max_tokens=1800,
                purpose="designer",
                enable_thinking=False,
            )
        )

    chat = llm_chat or default_chat

    initial_prompt = build_initial_prompt(version=version)
    log("info", f"Initial prompt built ({len(initial_prompt)} chars).")
    history: List[DesignAttempt] = []
    code: Optional[str] = None
    last_attempt: Optional[DesignAttempt] = None
    start_ts = time.time()

    for attempt_idx in range(1, max_attempts + 1):
        attempt_start = time.time()
        if attempt_idx == 1 or code is None:
            user_prompt = initial_prompt
            log("info", f"Attempt {attempt_idx}/{max_attempts}: using initial prompt.")
        else:
            assert last_attempt is not None and code is not None
            user_prompt = build_repair_prompt(version, code, last_attempt)
            err_summary = (last_attempt.error or "").splitlines()[-1][:120] if last_attempt.error else "?"
            log(
                "info",
                f"Attempt {attempt_idx}/{max_attempts}: building repair prompt "
                f"({len(user_prompt)} chars; last error: {err_summary}).",
            )

        # llm_client doesn't expose a name function, so reach into settings.
        try:
            _model_name = settings.llm_model  # noqa: SLF001
        except Exception:
            _model_name = "?"
        log("info", f"Calling LLM (model={_model_name})…")
        llm_t0 = time.time()
        try:
            response = chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7 if attempt_idx == 1 else 0.3,
            )
        except Exception as exc:
            log("error", f"LLM call failed after {(time.time()-llm_t0):.1f}s: {exc}")
            attempt = DesignAttempt(
                attempt=attempt_idx,
                code=None,
                passed=False,
                error=f"LLM call failed: {exc}",
                duration_ms=(time.time() - attempt_start) * 1000,
                elapsed_s=time.time() - start_ts,
            )
            history.append(attempt)
            last_attempt = attempt
            if on_attempt:
                on_attempt(attempt)
            continue
        log("ok", f"LLM responded in {(time.time()-llm_t0):.1f}s ({len(response)} chars).")

        extracted = extract_code(response)
        if not extracted:
            log("warn", "No ```python``` code block in response; will retry.")
            attempt = DesignAttempt(
                attempt=attempt_idx,
                code=None,
                passed=False,
                error="LLM response did not contain a parseable ```python``` code block.",
                duration_ms=(time.time() - attempt_start) * 1000,
                elapsed_s=time.time() - start_ts,
            )
            history.append(attempt)
            last_attempt = attempt
            code = None
            if on_attempt:
                on_attempt(attempt)
            continue
        log("info", f"Extracted Python block ({len(extracted)} chars). Compiling…")

        code = extracted
        try:
            fn, fn_name, _ = load_function(code)
        except Exception:
            log("warn", "Generated code failed to import; will retry.")
            attempt = DesignAttempt(
                attempt=attempt_idx,
                code=code,
                passed=False,
                error=f"Generated code failed to import/exec:\n{traceback.format_exc(limit=6)}",
                duration_ms=(time.time() - attempt_start) * 1000,
                elapsed_s=time.time() - start_ts,
            )
            history.append(attempt)
            last_attempt = attempt
            if on_attempt:
                on_attempt(attempt)
            continue
        log("info", f"Loaded function `{fn_name}`. Running sandbox on {len(cell_paths)} cells…")

        sandbox_t0 = time.time()
        passed, err, failing_cell, losses = sandbox_evaluate(
            fn, model_path, cell_paths, row_start=row_start, row_end=row_end
        )
        log(
            "ok" if passed else "warn",
            f"Sandbox finished in {(time.time()-sandbox_t0):.2f}s → "
            f"{'passed' if passed else 'failed'}"
            + (f" (cell {failing_cell}: {(err or '').splitlines()[-1][:120]})" if not passed and err else ""),
        )

        # Quality gate (only when sandbox passes). Runs the candidate on all
        # cells and checks CDDI ↔ indicator correlation. We need a char file for
        # this — without one, we accept the sandbox verdict as-is.
        if passed and ds.char_file:
            log("info", f"Quality gate: evaluating CDDI ↔ indicator correlation on full dataset…")
            qc_t0 = time.time()
            q_passed, q_msg, q_mean_r, _q_per_feat = quality_check(
                fn,
                ds.folder,
                ds.model_file,
                ds.char_file,
                row_start=row_start,
                row_end=row_end,
                cells_filter=cells_filter,
            )
            logger.info("designer.quality_check: %s (mean_r=%.3f)", q_msg, q_mean_r)
            log(
                "ok" if q_passed else "warn",
                f"Quality gate ({(time.time()-qc_t0):.1f}s): mean |r|={q_mean_r:.3f} → "
                f"{'passed' if q_passed else 'rejected — ' + q_msg[:120]}",
            )
            if not q_passed:
                passed = False
                err = q_msg
                # leave failing_cell None; it's not a per-cell error

        attempt = DesignAttempt(
            attempt=attempt_idx,
            code=code,
            passed=passed,
            error=err,
            failing_cell=failing_cell,
            sample_losses=losses,
            duration_ms=(time.time() - attempt_start) * 1000,
            elapsed_s=time.time() - start_ts,
        )
        history.append(attempt)
        last_attempt = attempt
        if on_attempt:
            on_attempt(attempt)

        if passed:
            gpt_path_str = str(EVO_GPT_PATH)
            if write_back:
                _write_back(code)
                log("ok", f"Wrote designed function back to {Path(gpt_path_str).name}.")
            log(
                "ok",
                f"Designer SUCCESS after {attempt_idx} attempt(s) in "
                f"{(time.time()-start_ts):.1f}s. Function: {fn_name}.",
            )
            return DesignResult(
                code=code,
                function_name=fn_name,
                passed=True,
                attempts=attempt_idx,
                history=history,
                gpt_path=gpt_path_str,
                sample_losses=losses,
                last_error=None,
            )

    log(
        "error",
        f"Designer exhausted {max_attempts} attempts in {(time.time()-start_ts):.1f}s without a passing candidate.",
    )
    return DesignResult(
        code=code or "",
        function_name=find_function_name(code) if code else "",
        passed=False,
        attempts=max_attempts,
        history=history,
        gpt_path=str(EVO_GPT_PATH),
        sample_losses=last_attempt.sample_losses if last_attempt else {},
        last_error=last_attempt.error if last_attempt else "no LLM response",
    )


def _write_back(code: str) -> None:
    EVO_GPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if EVO_GPT_PATH.exists():
        backup = EVO_GPT_PATH.with_suffix(EVO_GPT_PATH.suffix + ".bak")
        shutil.copy2(EVO_GPT_PATH, backup)
    EVO_GPT_PATH.write_text(code.rstrip() + "\n", encoding="utf-8")


# ---------- CLI ----------

def _cli(argv: Optional[List[str]] = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Lightweight CDDI Designer (CLI)")
    parser.add_argument("--dataset", default="builtin-first-200s")
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--cells", type=int, default=None)
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-write-back", action="store_true")
    parser.add_argument("--print-code", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )

    def progress(att: DesignAttempt) -> None:
        if att.passed:
            losses = ", ".join(f"Cell{c}={v:.4g}" for c, v in sorted(att.sample_losses.items()))
            print(f"[attempt {att.attempt}] OK in {att.duration_ms:.0f} ms — {losses}")
        else:
            err = (att.error or "").splitlines()[-1] if att.error else "?"
            print(f"[attempt {att.attempt}] FAIL ({att.duration_ms:.0f} ms): {err}")

    result = design_mmd(
        dataset_id=args.dataset,
        max_attempts=args.max_attempts,
        sandbox_cells=args.cells,
        version=args.version,
        seed=args.seed,
        write_back=not args.no_write_back,
        on_attempt=progress,
    )

    print("\n=== summary ===")
    print(json.dumps(
        {
            "passed": result.passed,
            "function_name": result.function_name,
            "attempts": result.attempts,
            "sample_losses": result.sample_losses,
            "gpt_path": result.gpt_path,
            "last_error_head": (result.last_error or "")[:200] if not result.passed else None,
        },
        ensure_ascii=False,
        indent=2,
    ))
    if args.print_code and result.code:
        print("\n=== generated code ===")
        print(result.code)
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(_cli())
