"""BATTWISE reflective-search subprocess wrapper.

Responsibilities:
    * Spawn ``Evo_Textgrad_battery_eoh_deepseek/main.py`` as a child process
      with user-supplied Hydra overrides (case, max_fe, pop_size, model, ...).
    * Locate the output directory the child creates under
      ``outputs/output/<timestamp>/`` and tail its ``main.log``.
    * Parse Generation / Objective Value / Saved-to checkpoints into a stream
      of :class:`ReflectiveSearchEvent` records.
    * Surface the latest best code by reading
      ``best_population_generation_N.json`` whenever a new one appears.
    * Expose a cancel hook (``terminate``) so the agent can abort gracefully.

The parser is split out as :func:`parse_log_line` for unit testing without
needing to run a full reflective-search experiment.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Dict, Iterable, List, Mapping, Optional, Tuple

from app.backend.services import llm_client
from app.backend.settings import EVO_DIR, EVO_GPT_PATH

logger = logging.getLogger(__name__)


# ==========================================
# Event model
# ==========================================

@dataclass
class ReflectiveSearchEvent:
    phase: str  # "startup" | "case" | "iter_eval" | "generation" | "objective" | "best_path" | "best_code" | "finished" | "error" | "stdout"
    generation: Optional[int] = None
    total_generations: Optional[int] = None
    objective: Optional[float] = None
    case_num: Optional[List[int]] = None
    iter_idx: Optional[int] = None
    response_id: Optional[int] = None
    code_path: Optional[str] = None
    code_preview: Optional[str] = None
    text: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)


# ==========================================
# Log line parser
# ==========================================

_GEN_RE = re.compile(
    r"Generation\s+(?P<gen>\d+)(?:\s*\(Initial\)|/(?P<total>\d+))\s*-\s*BEST SOLUTION",
    re.I,
)
_OBJ_LINE_RE = re.compile(r"Objective\s+Value:\s*(?P<val>[-+]?\w[\w\.+-]*)", re.I)
_OBJ_ITER_RE = re.compile(
    r"Iteration\s+(?P<it>\d+),\s+response_id\s+(?P<rid>\d+):\s*Objective\s+value:\s*(?P<val>[-+]?\w[\w\.+-]*)",
    re.I,
)
_SAVED_BEST_RE = re.compile(
    r"Saved\s+to:\s*(?P<path>.+best_population_generation_(?P<gen>\d+)\.json)\s*$",
    re.I,
)
_BEST_CODE_OVERALL_RE = re.compile(r"Best Code Overall:\s*(?P<preview>.+)$")
_BEST_CODE_PATH_RE = re.compile(r"Best Code Path Overall:\s*(?P<path>.+)$")
_BEST_OBJ_OVERALL_RE = re.compile(r"Best Objective Overall:\s*(?P<val>.+)$")
_CASE_RE = re.compile(r"Case num:\s*\[?(?P<cases>[\d,\s]+)\]?")
_ITER_EVAL_RE = re.compile(
    r"Iteration\s+(?P<it>\d+):\s+Code Run\s+(?P<rid>\d+)\s+successful",
    re.I,
)
_STARTUP_RE = re.compile(
    r"Starting (?:BATTWISE Reflective Metric Search|EOH-Textgrad evolution)",
    re.I,
)
_INIT_OK_RE = re.compile(
    r"(?:BATTWISE Reflective Metric Search|EOH-Textgrad) initialized successfully",
    re.I,
)
_FINISHED_RE = re.compile(
    r"(?:BATTWISE Reflective Metric Search|EOH-Textgrad evolution) completed",
    re.I,
)


def _to_float(s: str) -> Optional[float]:
    s = s.strip().rstrip(",")
    if not s:
        return None
    if s.lower() in {"inf", "+inf"}:
        return float("inf")
    if s.lower() == "-inf":
        return float("-inf")
    if s.lower() == "nan":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return None


def parse_log_line(line: str) -> Optional[ReflectiveSearchEvent]:
    """Map one ``main.log`` line to a structured event. Returns None when the
    line is uninteresting (HTTP requests, progress noise, etc.)."""
    if not line:
        return None
    body = line
    # strip "[ts][root][LEVEL] - " prefix if present
    if "] - " in body:
        body = body.split("] - ", 1)[-1]
    body = body.strip()
    if not body:
        return None

    if _STARTUP_RE.search(body) or _INIT_OK_RE.search(body):
        return ReflectiveSearchEvent(phase="startup", text=body)

    m = _CASE_RE.search(body)
    if m and body.lower().startswith("case num"):
        cases = [int(x) for x in re.findall(r"\d+", m.group("cases"))]
        return ReflectiveSearchEvent(phase="case", case_num=cases)

    m = _GEN_RE.search(body)
    if m:
        gen = int(m.group("gen"))
        total = int(m.group("total")) if m.group("total") else None
        return ReflectiveSearchEvent(phase="generation", generation=gen, total_generations=total)

    m = _OBJ_LINE_RE.search(body)
    if m and "response_id" not in body:
        return ReflectiveSearchEvent(phase="objective", objective=_to_float(m.group("val")))

    m = _OBJ_ITER_RE.search(body)
    if m:
        return ReflectiveSearchEvent(
            phase="iter_eval",
            iter_idx=int(m.group("it")),
            response_id=int(m.group("rid")),
            objective=_to_float(m.group("val")),
        )

    m = _ITER_EVAL_RE.search(body)
    if m:
        return ReflectiveSearchEvent(
            phase="iter_eval",
            iter_idx=int(m.group("it")),
            response_id=int(m.group("rid")),
            text="run_ok",
        )

    m = _SAVED_BEST_RE.search(body)
    if m:
        return ReflectiveSearchEvent(
            phase="best_path",
            code_path=m.group("path").strip(),
            generation=int(m.group("gen")),
        )

    m = _BEST_CODE_OVERALL_RE.search(body)
    if m:
        return ReflectiveSearchEvent(phase="best_code", code_preview=m.group("preview").strip())

    m = _BEST_OBJ_OVERALL_RE.search(body)
    if m:
        return ReflectiveSearchEvent(phase="finished", objective=_to_float(m.group("val")))

    if _FINISHED_RE.search(body):
        return ReflectiveSearchEvent(phase="finished", text=body)

    return None


def parse_lines(lines: Iterable[str]) -> List[ReflectiveSearchEvent]:
    """Convenience: parse a sequence of lines (for tests)."""
    out: List[ReflectiveSearchEvent] = []
    for line in lines:
        ev = parse_log_line(line)
        if ev is not None:
            out.append(ev)
    return out


# ==========================================
# Code preview loader
# ==========================================

def load_best_code_from_json(path: Path) -> Optional[str]:
    """Read a ``best_population_generation_N.json`` and return the inner code."""
    try:
        # The file uses `Infinity` (not a valid JSON token) for inf objectives,
        # so we sanitize the text before parsing.
        text = path.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"\bInfinity\b", "1e308", text)
        text = re.sub(r"\b-Infinity\b", "-1e308", text)
        text = re.sub(r"\bNaN\b", "null", text)
        data = json.loads(text)
        if isinstance(data, dict):
            code = data.get("code")
            if isinstance(code, str):
                return code
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0].get("code") if isinstance(data[0].get("code"), str) else None
    except Exception as exc:
        logger.warning("Failed to read EOH best-code JSON %s: %s", path, exc)
    return None


# ==========================================
# Subprocess job
# ==========================================

def prepare_reflective_search_runtime(
    overrides: Optional[Iterable[str]],
    base_env: Mapping[str, str],
    config: llm_client.ResolvedConfig,
) -> Tuple[List[str], Dict[str, str]]:
    """Build child overrides and environment without mutating the parent.

    Explicit EOH-only variables and an explicit Hydra model override take
    precedence. Otherwise the child inherits the application's resolved LLM
    configuration.
    """
    runtime_overrides = list(overrides or [])
    if not any(item.startswith("model=") for item in runtime_overrides):
        runtime_overrides.append(f"model={config.model}")

    env = dict(base_env)
    if not env.get("BATRECYCLER_EOH_API_KEY") and not env.get(
        "BATRECYCLER_EOH_API_KEYS"
    ):
        env["BATRECYCLER_EOH_API_KEY"] = config.api_key
    if config.base_url and not env.get("BATRECYCLER_EOH_BASE_URL"):
        env["BATRECYCLER_EOH_BASE_URL"] = config.base_url
    return runtime_overrides, env


@dataclass
class ReflectiveSearchJob:
    process: asyncio.subprocess.Process
    started_at: float
    output_dir: Optional[Path] = None
    log_path: Optional[Path] = None
    overrides: List[str] = field(default_factory=list)
    case_num: Optional[List[int]] = None
    pid: Optional[int] = None

    def is_running(self) -> bool:
        return self.process.returncode is None

    async def terminate(self, kill_after: float = 5.0) -> int:
        if not self.is_running():
            return self.process.returncode or 0
        try:
            self.process.terminate()
        except ProcessLookupError:
            return self.process.returncode or 0
        try:
            return await asyncio.wait_for(self.process.wait(), timeout=kill_after)
        except asyncio.TimeoutError:
            try:
                self.process.kill()
            except ProcessLookupError:
                pass
            return await self.process.wait()


async def start_reflective_search(
    *,
    overrides: Optional[List[str]] = None,
    cwd: Path = EVO_DIR,
    case_num: Optional[List[int]] = None,
    log_to: Optional[Path] = None,
) -> ReflectiveSearchJob:
    """Spawn ``python main.py`` in EVO_DIR with the provided Hydra overrides.

    The child creates its own ``outputs/output/<timestamp>/`` folder; we
    discover that folder afterwards by polling for the newest entry.
    """
    if not cwd.exists():
        raise FileNotFoundError(f"Reflective-search workspace not found: {cwd}")
    config = llm_client.resolve_config()
    overrides, env = prepare_reflective_search_runtime(overrides, os.environ, config)
    if case_num and not any(o.startswith("case_num=") for o in overrides):
        overrides.append(f"case_num=[{','.join(str(c) for c in case_num)}]")

    cmd = [sys.executable, "main.py", *overrides]
    env.setdefault("PYTHONUNBUFFERED", "1")

    stdout_file = open(log_to, "wb") if log_to else asyncio.subprocess.DEVNULL
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        env=env,
        stdout=stdout_file if log_to else asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.STDOUT,
    )

    job = ReflectiveSearchJob(
        process=proc,
        started_at=time.time(),
        overrides=overrides,
        case_num=case_num,
        pid=proc.pid,
    )
    return job


# ==========================================
# Output-dir discovery + log tailing
# ==========================================

async def find_new_output_dir(
    *,
    root: Path = EVO_DIR / "outputs" / "output",
    before: Optional[set] = None,
    started_after: float = 0.0,
    timeout: float = 30.0,
    poll: float = 0.5,
) -> Optional[Path]:
    """Wait until the child has created a new timestamped output sub-directory.

    The directory is identified as one whose mtime exceeds ``started_after``
    and that contains (or will contain) ``main.log``.
    """
    before = set(before or [])
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if root.exists():
            for entry in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if not entry.is_dir():
                    continue
                if entry.name in before:
                    continue
                try:
                    if entry.stat().st_mtime < started_after - 1.0:
                        continue
                except FileNotFoundError:
                    continue
                # Accept even before main.log appears so we can start polling
                return entry
        await asyncio.sleep(poll)
    return None


async def tail_log(
    log_path: Path,
    *,
    job: ReflectiveSearchJob,
    poll: float = 0.5,
    quiet_grace: float = 2.0,
) -> AsyncIterator[str]:
    """Async generator yielding new lines from ``log_path`` as they appear.

    Stops once the underlying subprocess has exited *and* no new lines arrive
    for ``quiet_grace`` seconds.
    """
    last_size = 0
    buffer = ""
    last_data_ts = time.monotonic()
    while True:
        try:
            size = log_path.stat().st_size if log_path.exists() else 0
        except FileNotFoundError:
            size = 0
        if size > last_size:
            with log_path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(last_size)
                chunk = f.read()
            buffer += chunk
            last_size = size
            last_data_ts = time.monotonic()
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                yield line
        # exit conditions
        if not job.is_running():
            # one more sweep already done; allow grace period then break
            if time.monotonic() - last_data_ts > quiet_grace:
                if buffer:
                    yield buffer
                return
        await asyncio.sleep(poll)
