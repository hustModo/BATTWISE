"""Build an EOH seed population from prior `outputs/output/<timestamp>/` runs.

EOH supports `--use_seed=True --seed_path=<json>` where the JSON is a list of
`{algorithm, code}` items. We pre-populate that file from the curated set of
historical winners so that a fresh reflective-search run starts at a strong
point, instead of burning 5+ generations re-discovering the basics.

This is what lets the user say "默认 3 代" (`max_fe=3`) and still see
meaningful evolution: gen 0 is a high-quality seed pool, gens 1..max_fe
just need to refine it.

Public:
    build_seeds(...) -> Path
        Writes seeds.json under EVO_DIR (or a custom path) and returns it.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from app.backend.settings import EVO_DIR

logger = logging.getLogger(__name__)


# Curated set of historical runs (matches the dirs the user pointed at).
# Each is a relative path under EVO_DIR / "outputs" / "output".
# Skipped: "2026-02-05_22-39-09" and "2026-02-05_22-39-07" — empty (main.log only).
DEFAULT_HISTORY_DIRS = [
    "2026-02-06_06-32-14",
    "2026-02-06_08-18-59",
    "2026-02-06_04-54-40",
    "2026-02-06_00-33-56",
    "2026-02-05_22-39-19",
]

# Case-specific historical winners. Keep the default mixed pool for generic
# fallbacks, but DesignerAgent should prefer these when it knows the active
# dataset. This prevents an All-200 (AII_200, case 4) run from being seeded
# only by First-200 / CV / CC winners.
CASE_HISTORY_DIRS = {
    # eval.py mapping: 2 = first_200s
    2: [
        "2026-02-06_06-32-14",
        "2026-01-31_23-32-03",
        "2026-01-22_02-42-48",
        "2025-12-23_13-23-53",
        "2025-12-17_13-36-43",
        "2025-12-14_01-16-36",
    ],
    # eval.py mapping: 4 = AII_200 / All-200 refinement data
    4: [
        "2026-02-04_22-57-43",
        "2026-01-31_18-20-05",
        "2026-01-22_04-18-11",
        "2026-01-19_22-54-49",
        "2026-01-16_10-41-22",
        "2026-01-15_11-09-44",
    ],
}

# Minimum number of seed individuals we want to expose. Too few makes the
# initial population homogeneous and EOH's crossover operators (e1/e2) have
# nothing useful to recombine. We pad with the seed_func baseline if the
# curated dirs don't yield enough.
MIN_SEEDS = 4

# Cap so we don't dump 50+ identical-looking candidates into gen 0.
MAX_SEEDS = 8


@dataclass
class _Candidate:
    algorithm: str
    code: str
    objective: float
    source: str


def _read_best_json(path: Path) -> Optional[_Candidate]:
    """Read a best_population_generation_N.json.

    EOH writes those as a single dict (the best individual of that gen).
    Some older runs use Infinity literally → sanitize before json.loads.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"\bInfinity\b", "1e308", text)
        text = re.sub(r"\b-Infinity\b", "-1e308", text)
        text = re.sub(r"\bNaN\b", "null", text)
        data = json.loads(text)
        if isinstance(data, list):
            data = data[0] if data else None
        if not isinstance(data, dict):
            return None
        code = data.get("code")
        algorithm = data.get("algorithm") or "Historical EOH winner"
        objective = data.get("objective")
        if not isinstance(code, str) or not code.strip():
            return None
        try:
            obj_f = float(objective)
        except (TypeError, ValueError):
            obj_f = float("inf")
        return _Candidate(
            algorithm=algorithm.strip(),
            code=code.strip(),
            objective=obj_f,
            source=path.name,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Skipping unreadable seed file %s: %s", path, exc)
        return None


def _read_best_code_txt(path: Path) -> Optional[_Candidate]:
    """Read a best_code_battery_cddi.txt (final winner of a run)."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
    code = (m.group(1) if m else raw).strip()
    if not code or "def " not in code:
        return None
    return _Candidate(
        algorithm=f"Historical final winner ({path.name})",
        code=code,
        objective=-1.0,  # unknown — treat as good (best_code is the winner)
        source=path.name,
    )


def _seed_baseline() -> _Candidate:
    """Tiny known-good fallback so the seed pool is never empty."""
    code = (
        "import numpy as np\n"
        "from scipy.spatial.distance import cdist\n\n"
        "def calculate_mmd_v2(xs, xt, fix_sigma=0, kernel_mul=2.0, kernel_num=5):\n"
        "    s_num, t_num = xs.shape[0], xt.shape[0]\n"
        "    n_sam = s_num + t_num\n"
        "    x_new = np.vstack([xs, xt])\n"
        "    L2 = cdist(x_new, x_new) ** 2\n"
        "    bw = max(float(L2.sum()) / max(n_sam * n_sam - n_sam, 1), 1e-12)\n"
        "    bw /= 4.0\n"
        "    K = np.zeros((n_sam, n_sam))\n"
        "    for i in range(kernel_num):\n"
        "        K += np.exp(-L2 / max(bw * (kernel_mul ** i), 1e-12))\n"
        "    ss = float(K[:s_num, :s_num].mean())\n"
        "    tt = float(K[s_num:, s_num:].mean())\n"
        "    st = float(K[:s_num, s_num:].mean())\n"
        "    return ss + tt - 2 * st\n"
    )
    return _Candidate(
        algorithm="Baseline multi-kernel MMD on raw signal",
        code=code,
        objective=0.0,
        source="<baseline-fallback>",
    )


def _gather_candidates(history_dirs: Iterable[str]) -> List[_Candidate]:
    root = EVO_DIR / "outputs" / "output"
    out: List[_Candidate] = []
    seen_codes: set[str] = set()

    def _push(c: _Candidate) -> None:
        # de-dupe on code body — runs often emit identical winners.
        key = re.sub(r"\s+", " ", c.code).strip()
        if key in seen_codes:
            return
        seen_codes.add(key)
        out.append(c)

    for sub in history_dirs:
        d = root / sub
        if not d.exists():
            continue

        # final winner first (best signal)
        final_txt = d / "best_code_battery_cddi.txt"
        if not final_txt.exists():
            final_txt = d / "best_code_battery_mmd.txt"  # Historical saved runs.
        if final_txt.exists():
            c = _read_best_code_txt(final_txt)
            if c:
                c.source = f"{sub}/{c.source}"
                _push(c)

        # then any per-generation best dicts (we pick the latest gens first,
        # since those carry the most-evolved code).
        gens = sorted(
            d.glob("best_population_generation_*.json"),
            key=lambda p: int(re.search(r"(\d+)\.json$", p.name).group(1))
            if re.search(r"(\d+)\.json$", p.name)
            else -1,
            reverse=True,
        )
        # take at most the top 2 gens per dir to keep diversity across runs.
        for g in gens[:2]:
            c = _read_best_json(g)
            if c:
                c.source = f"{sub}/{c.source}"
                _push(c)

    if not out:
        out.append(_seed_baseline())
    # pad to MIN_SEEDS with the baseline if needed
    while len(out) < MIN_SEEDS:
        out.append(_seed_baseline())
    # rank: lower objective first (EOH minimises); unknown=-1.0 ranks first.
    out.sort(key=lambda c: c.objective)
    return out[:MAX_SEEDS]


def build_seeds(
    *,
    history_dirs: Optional[Iterable[str]] = None,
    case_num: Optional[int] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """Materialise a seeds.json EOH can consume via `exp_seed_path`.

    Args:
        history_dirs: relative names under EVO_DIR/outputs/output. If omitted
            and case_num is known, use the curated case-specific directories;
            otherwise defaults to DEFAULT_HISTORY_DIRS.
        case_num: eval.py case id (2=first_200s, 4=AII_200/All-200).
        out_path: where to write the JSON. Defaults to
            ``EVO_DIR / "seeds_from_history.json"`` so all sessions can
            share the file.

    Returns:
        Path to the written JSON.
    """
    if history_dirs is not None:
        history_dirs = list(history_dirs)
    elif case_num in CASE_HISTORY_DIRS:
        history_dirs = CASE_HISTORY_DIRS[int(case_num)]
    else:
        history_dirs = DEFAULT_HISTORY_DIRS
    cands = _gather_candidates(history_dirs)
    payload = [{"algorithm": c.algorithm, "code": c.code} for c in cands]
    out_path = out_path or (
        EVO_DIR / (f"seeds_from_history_case{case_num}.json" if case_num is not None else "seeds_from_history.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "EOH seeds.json built with %d candidates from %d historical dirs → %s",
        len(payload), len(history_dirs), out_path,
    )
    return out_path
