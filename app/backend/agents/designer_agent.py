"""CDDI Designer Agent.

Two modes (driven by ``session.mode``):

* **lightweight** — wraps ``services.mmd_designer.design_mmd``.
  One LLM call + sandbox repair loop. Done in seconds.
  Emits agent_started / thinking / tool_call / code_update / artifact / agent_finished.

* **reflective_search** — wraps ``services.reflective_search_runner``.
  Spawns ``Evo_Textgrad_battery_eoh_deepseek/main.py`` as a child process,
  tails its ``main.log`` and streams generation/objective progress plus the
  current best code (Monaco). Cancellable from the workbench.
"""
from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.backend.agents.base import AgentBase
from app.backend.naming import REFLECTIVE_SEARCH_MODE, is_reflective_search_mode
from app.backend.orchestrator.session_store import Artifact, store
from app.backend.services import eoh_seeds, mmd_designer, reflective_search_runner
from app.backend.services.mmd_designer import DesignAttempt, DesignResult
from app.backend.settings import EVO_DIR, EVO_GPT_PATH, settings


# Lightweight defaults for the conversational Reflective Search path. The
# upstream EoH configuration
# config_battery_mmd.yaml ships max_fe=50, pop_size=20 — far too heavy for
# an interactive demo. Combined with the historical seed pool (gen 0 is
# already strong), these values give a meaningful but short evolution.
REFLECTIVE_SEARCH_LIGHT_DEFAULTS: Dict[str, str] = {
    "max_fe": "3",
    "pop_size": "4",
    "batch_size": "4",
    # The next ones are NOT user-exposed but we still pin them for speed
    # and predictability (originals: 20 / 60 / 5 — too greedy).
    "llm_max_workers": "4",
    "llm_timeout": "60",
    "reflection_frequency": "3",
}


def _stash_designed_code(session, dataset_id: str, code: str, function_name: Optional[str] = None) -> None:
    by_dataset = dict(session.context.get("designed_code_by_dataset") or {})
    by_dataset[dataset_id] = {
        "code": code,
        "function_name": function_name,
        "t": time.time(),
    }
    session.context["designed_code_by_dataset"] = by_dataset
    session.context["last_designed_dataset_id"] = dataset_id
    session.context["last_designed_code"] = code
    if function_name:
        session.context["last_designed_function"] = function_name


def _dataset_to_search_case(dataset_id: str) -> int:
    """Map backend dataset ids to eval.py case ids.

    eval.py mapping: 2=first_200s, 4=AII_200/All-200.
    """
    return {
        "builtin-first-200s": 2,
        "builtin-first-100s": 1,
        "builtin-all-200": 4,
    }.get(dataset_id, 2)

# Deprecated code-level aliases for integrations written before BATTWISE.
FULL_EOH_LIGHT_DEFAULTS = REFLECTIVE_SEARCH_LIGHT_DEFAULTS
_dataset_to_eoh_case = _dataset_to_search_case


class DesignerAgent(AgentBase):
    agent_id = "designer"
    node_id = "designer"

    async def _log_llm(self, tag: str, line: str) -> None:
        """Stream a single runtime-log line into the llm_log frontend channel.

        Shared format with ComputeAgent / ClusterAgent so the user has ONE
        live terminal showing every agent step (incl. the LLM call lifecycle).
        Tags: info | ok | warn | error.
        """
        await self.emit(
            "chart_data",
            chart="llm_log",
            data={"line": line[:500], "tag": tag, "agent": self.agent_id},
        )

    async def run(
        self,
        user_text: str = "",
        *,
        version: Optional[int] = None,
        max_attempts: Optional[int] = None,
        sandbox_cells: Optional[int] = None,
        cells_filter: Optional[List[int]] = None,
    ) -> Dict[str, Any]:  # type: ignore[override]
        await self.started()

        if is_reflective_search_mode(self.session.mode):
            return await self._run_reflective_search(cells_filter=cells_filter)

        dataset_id = self.session.active_dataset_id
        effective_max_attempts = max_attempts or settings.designer_max_repair_attempts
        effective_sandbox_cells = sandbox_cells or settings.designer_sandbox_cells
        await self.thinking(
            f"Designing a new CDDI scoring function. dataset={dataset_id}, "
            f"max_attempts={effective_max_attempts}, "
            f"sandbox_cells={effective_sandbox_cells}."
        )
        await self._log_llm(
            "info",
            "Designer started (lightweight). "
            f"dataset={dataset_id}, max_attempts={effective_max_attempts}, "
            f"sandbox_cells={effective_sandbox_cells}.",
        )

        next_version = self._next_version()

        call_id = await self.tool_call(
            "mmd_designer.design_mmd",
            {
                "dataset_id": dataset_id,
                "version": next_version,
                "max_attempts": effective_max_attempts,
                "sandbox_cells": effective_sandbox_cells,
            },
            status="running",
        )

        loop = asyncio.get_running_loop()
        progress_queue: asyncio.Queue[DesignAttempt] = asyncio.Queue()
        log_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

        def on_attempt(att: DesignAttempt) -> None:
            loop.call_soon_threadsafe(progress_queue.put_nowait, att)

        def on_log(tag: str, line: str) -> None:
            # `mmd_designer.design_mmd` calls this from its worker thread on
            # every meaningful step: prompt build, LLM call start/end, code
            # extraction, sandbox start/end, quality check, ...
            loop.call_soon_threadsafe(log_queue.put_nowait, (tag, line))

        async def consume_attempts() -> None:
            while True:
                att = await progress_queue.get()
                if att is None:
                    break
                await self._emit_attempt(att)

        async def consume_logs() -> None:
            while True:
                item = await log_queue.get()
                if item is None:
                    break
                tag, line = item
                await self._log_llm(tag, line)

        start_ts = time.time()
        attempt_consumer = asyncio.create_task(consume_attempts())
        log_consumer = asyncio.create_task(consume_logs())

        try:
            result: DesignResult = await loop.run_in_executor(
                None,
                lambda: mmd_designer.design_mmd(
                    dataset_id=dataset_id,
                    version=next_version,
                    max_attempts=effective_max_attempts,
                    sandbox_cells=effective_sandbox_cells,
                    on_attempt=on_attempt,
                    on_log=on_log,
                    write_back=True,
                ),
            )
        finally:
            await progress_queue.put(None)  # type: ignore[arg-type]
            await log_queue.put(None)  # type: ignore[arg-type]
            await attempt_consumer
            await log_consumer

        elapsed = time.time() - start_ts
        await self.tool_done(
            call_id,
            "mmd_designer.design_mmd",
            duration_ms=elapsed * 1000,
            result={
                "passed": result.passed,
                "attempts": result.attempts,
                "function_name": result.function_name,
                "sample_losses": result.sample_losses,
            },
            error=None if result.passed else (result.last_error or "design failed"),
        )

        if result.passed:
            await self._emit_code_update(
                code=result.code,
                attempts=result.attempts,
                passed=True,
                last_error=None,
            )
            await self._emit_artifact(result)
            self.session.context["last_designed_version"] = next_version
            _stash_designed_code(
                self.session,
                dataset_id,
                result.code,
                result.function_name,
            )
            await store.update(self.session)
            await self.finished(
                result={
                    "passed": True,
                    "attempts": result.attempts,
                    "function_name": result.function_name,
                    "sample_losses": result.sample_losses,
                    "gpt_path": result.gpt_path,
                }
            )
            await self.say(
                f"Designed `{result.function_name}` in {result.attempts} attempt(s). "
                f"Sandbox CDDI losses on {len(result.sample_losses)} cells look healthy. "
                "Next: run CDDI Compute on the full batch."
            )
        else:
            await self._emit_code_update(
                code=result.code,
                attempts=result.attempts,
                passed=False,
                last_error=result.last_error,
            )
            await self.failed(result.last_error or "Designer exhausted all attempts")
            await self.say(
                f"Designer exhausted {result.attempts} attempts without producing a "
                "NaN-safe CDDI function. You may retry, switch to Reflective Search mode, or "
                "tweak the LLM provider in Settings."
            )

        return {
            "passed": result.passed,
            "attempts": result.attempts,
            "function_name": result.function_name,
        }

    def _next_version(self) -> int:
        last = int(self.session.context.get("last_designed_version", 1))
        return last + 1

    async def _emit_attempt(self, att: DesignAttempt) -> None:
        await self.emit(
            "thinking",
            text=(
                f"Attempt {att.attempt}: "
                + ("passed sandbox." if att.passed else (
                    "failed — " + (att.error or "").splitlines()[-1][:200]
                ))
            ),
            attempt=att.attempt,
            passed=att.passed,
            duration_ms=att.duration_ms,
        )
        await self.emit(
            "code_update",
            path=str(EVO_GPT_PATH),
            code=att.code or "",
            nan_check={
                "passed": att.passed,
                "attempts": att.attempt,
                "last_error": (att.error or "").splitlines()[-1] if att.error else None,
            },
            sample_losses=att.sample_losses,
            failing_cell=att.failing_cell,
        )

    async def _emit_code_update(
        self,
        *,
        code: str,
        attempts: int,
        passed: bool,
        last_error: Optional[str],
    ) -> None:
        await self.emit(
            "code_update",
            path=str(EVO_GPT_PATH),
            code=code,
            nan_check={
                "passed": passed,
                "attempts": attempts,
                "last_error": (last_error or "").splitlines()[-1] if last_error else None,
            },
            final=True,
        )

    # ==========================================
    # Reflective Search mode
    # ==========================================

    async def _run_reflective_search(
        self,
        cells_filter: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Spawn Evo_Textgrad/main.py and stream generation events.

        Two ergonomic upgrades on top of the attributed EoH operators:

        1. **Light defaults** (REFLECTIVE_SEARCH_LIGHT_DEFAULTS) — keep the demo short.
           The user may override max_fe / pop_size / batch_size via chat
           ("跑 5 代每代 6 个个体"); RouterAgent parses those and writes
           into ``session.context['eoh_user_params']``.
        2. **Historical seeding** — instead of random gen-0, we hand EOH a
           curated seed JSON built from prior winners under
           ``Evo_Textgrad/outputs/output/<timestamp>/``. So even with
           max_fe=3 the user sees movement off a strong starting point.
        """
        ctx = self.session.context
        user_overrides: List[str] = list(ctx.get("eoh_overrides") or [])
        user_params: Dict[str, str] = dict(ctx.get("eoh_user_params") or {})

        # Resolve final flat-key map (user override > user_params > light defaults).
        resolved: Dict[str, str] = dict(REFLECTIVE_SEARCH_LIGHT_DEFAULTS)
        # apply chat-parsed params (max_fe / pop_size / batch_size only)
        for k in ("max_fe", "pop_size", "batch_size"):
            if user_params.get(k):
                resolved[k] = str(user_params[k])
        # then any explicit Hydra-style overrides ("foo=bar") win.
        for ov in user_overrides:
            if "=" in ov and not ov.startswith("+"):
                k, _, v = ov.partition("=")
                resolved[k] = v
        overrides: List[str] = [f"{k}={v}" for k, v in resolved.items()]

        # Auto-select eval.py case based on active dataset.
        # Mapping in problems/battery_mmd/eval.py:
        #   2 = first_200s, 4 = AII_200 / All-200
        ds_id = self.session.active_dataset_id
        selected_case: Optional[int] = None
        if not any(o.startswith("case_num=") for o in overrides):
            selected_case = _dataset_to_search_case(ds_id)
            overrides.append(f"case_num=[{selected_case}]")
        else:
            for ov in overrides:
                if ov.startswith("case_num="):
                    m = re.search(r"\[(\d+)\]", ov)
                    if m:
                        selected_case = int(m.group(1))
                    break

        if cells_filter:
            cleaned_cells = sorted({int(c) for c in cells_filter})
            overrides.append("+eoh_cell_filter=" + ",".join(str(c) for c in cleaned_cells))
            await self._log_llm(
                "info",
                f"Reflective Search cell filter enabled: {cleaned_cells}",
            )

        # ---- historical seeds -------------------------------------------
        # Skip seed injection if the user disabled it ("no seeds" / Hydra override)
        # or already supplied a custom seed path.
        seeds_disabled = (
            ctx.get("eoh_disable_seeds")
            or any(o.startswith("+exp_use_seed=false") for o in overrides)
            or any(o.startswith("exp_use_seed=false") for o in overrides)
        )
        has_custom_seed = any("exp_seed_path=" in o for o in overrides)
        if not seeds_disabled and not has_custom_seed:
            try:
                seed_file = eoh_seeds.build_seeds(case_num=selected_case)
                overrides.append("+exp_use_seed=true")
                overrides.append(f"+exp_seed_path={seed_file}")
                await self._log_llm(
                    "info",
                    f"Search seeds built from case {selected_case} history → {seed_file.name}",
                )
            except Exception as exc:  # noqa: BLE001
                await self._log_llm("warn", f"Seed build failed (continuing without): {exc}")

        await self.thinking(
            f"Starting BATTWISE Reflective Search. dataset={ds_id}, overrides={overrides}. "
            "This may take several minutes; live progress will stream below."
        )

        # snapshot existing output directories so we can detect the new one
        outputs_root = EVO_DIR / "outputs" / "output"
        before = (
            {p.name for p in outputs_root.iterdir() if p.is_dir()}
            if outputs_root.exists()
            else set()
        )

        call_id = await self.tool_call(
            "reflective_search_runner.start_reflective_search",
            {"cwd": str(EVO_DIR), "overrides": overrides},
            status="running",
        )
        start_ts = time.time()

        try:
            job = await reflective_search_runner.start_reflective_search(
                overrides=overrides,
                case_num=None,
            )
        except Exception as exc:
            await self.tool_done(
                call_id, "reflective_search_runner.start_reflective_search", duration_ms=0,
                error=str(exc), result=None,
            )
            await self.failed(f"Failed to launch reflective-search subprocess: {exc}")
            return {"passed": False, "error": str(exc), "mode": REFLECTIVE_SEARCH_MODE}

        ctx["eoh_pid"] = job.pid
        await store.update(self.session)
        await self.log(f"Reflective-search subprocess started (pid={job.pid}).")

        # locate the output dir the child creates
        out_dir = await reflective_search_runner.find_new_output_dir(
            root=outputs_root, before=before, started_after=job.started_at, timeout=30.0,
        )
        if out_dir is None:
            await job.terminate()
            await self.tool_done(
                call_id, "reflective_search_runner.start_reflective_search",
                duration_ms=(time.time() - start_ts) * 1000,
                error="Reflective-search child never created an output directory within 30s",
            )
            await self.failed("Reflective-search child never created an output directory.")
            return {
                "passed": False,
                "error": "no output dir",
                "mode": REFLECTIVE_SEARCH_MODE,
            }

        job.output_dir = out_dir
        log_path = out_dir / "main.log"
        job.log_path = log_path
        await self.log(f"Reflective-search output dir: {out_dir.name}")

        # Tail the log and stream events.
        last_generation = -1
        best_objective: Optional[float] = None
        best_code_path: Optional[Path] = None
        progress_points: List[Dict[str, Any]] = []
        finished_ok = False

        async def watch_cancel() -> None:
            while job.is_running():
                if (
                    self.session.context.get("cancel_reflective_search")
                    or self.session.context.get("cancel_eoh")
                ):
                    await self.log(
                        "Cancellation requested — terminating reflective search.",
                        level="warning",
                    )
                    await job.terminate(kill_after=5.0)
                    self.session.context["cancel_reflective_search"] = False
                    self.session.context["cancel_eoh"] = False
                    await store.update(self.session)
                    return
                await asyncio.sleep(0.5)

        canceller = asyncio.create_task(watch_cancel())

        try:
            # Wait for log file to appear
            for _ in range(60):
                if log_path.exists():
                    break
                if not job.is_running():
                    break
                await asyncio.sleep(0.5)

            # rolling counters for per-generation candidate health (success/inf)
            gen_candidates: Dict[int, Dict[str, int]] = {}

            async for line in reflective_search_runner.tail_log(
                log_path,
                job=job,
                poll=0.5,
                quiet_grace=2.0,
            ):
                # ---- 1. raw log line → push to frontend mini-terminal ----
                # Filter out the chatty httpx access lines (one per LLM API call,
                # 60+ per generation) — the user wants to see the *algorithm*
                # heartbeat, not the HTTP transport. Truncate very long lines
                # so the frontend doesn't get jammed by a stray pandas dump.
                if "[httpx]" not in line and "/v1/chat/completions" not in line:
                    snippet = line.strip()
                    if snippet:
                        # tag the line so the frontend can colour it
                        if "BEST" in snippet or "Objective Value" in snippet:
                            tag = "best"
                        elif "Objective value: inf" in snippet:
                            tag = "inf"
                        elif "successful!" in snippet:
                            tag = "ok"
                        elif "[ERROR]" in snippet or "Traceback" in snippet:
                            tag = "error"
                        elif "[WARNING]" in snippet:
                            tag = "warn"
                        else:
                            tag = "info"
                        await self.emit(
                            "chart_data",
                            chart="reflective_search_log",
                            data={"line": snippet[:500], "tag": tag},
                        )

                ev = reflective_search_runner.parse_log_line(line)
                if ev is None:
                    continue
                # ---- track per-generation success / inf counts ----
                if ev.phase == "iter_eval" and ev.iter_idx is not None:
                    bucket = gen_candidates.setdefault(
                        ev.iter_idx, {"ok": 0, "inf": 0, "total": 0}
                    )
                    bucket["total"] += 1
                    if ev.objective is not None and ev.objective in (float("inf"), float("-inf")):
                        bucket["inf"] += 1
                    else:
                        bucket["ok"] += 1
                    # emit a rolling candidate-stats update once per 5 evals
                    if bucket["total"] % 5 == 0:
                        await self.emit(
                            "chart_data",
                            chart="reflective_search_candidates",
                            data={"generation": ev.iter_idx, **bucket},
                        )
                if ev.phase == "startup":
                    await self.thinking(ev.text or "Reflective search starting…")
                elif ev.phase == "case":
                    await self.log(f"Search case_num={ev.case_num}")
                elif ev.phase == "iter_eval":
                    # too noisy to emit one event per code; only every 20th
                    if ev.response_id is not None and ev.response_id % 20 == 0:
                        await self.emit("chart_data", chart="reflective_search_progress", data={
                            "phase": "iter_eval",
                            "iter": ev.iter_idx,
                            "response_id": ev.response_id,
                            "objective": ev.objective if ev.objective is not None else None,
                        })
                elif ev.phase == "generation":
                    last_generation = ev.generation or last_generation
                    await self.emit("chart_data", chart="reflective_search_progress", data={
                        "phase": "generation",
                        "generation": ev.generation,
                        "total_generations": ev.total_generations,
                        "history": progress_points,
                    })
                elif ev.phase == "objective":
                    if ev.objective is not None and ev.objective != float("inf") and ev.objective != float("-inf"):
                        best_objective = ev.objective
                    progress_points.append({
                        "generation": last_generation,
                        "objective": ev.objective if (ev.objective is not None and not (ev.objective in (float("inf"), float("-inf")))) else None,
                    })
                    await self.emit("chart_data", chart="reflective_search_progress", data={
                        "phase": "best_objective",
                        "generation": last_generation,
                        "objective": progress_points[-1]["objective"],
                        "history": progress_points,
                    })
                elif ev.phase == "best_path":
                    best_code_path = Path(ev.code_path) if ev.code_path else None
                    # Try to load the code from the JSON checkpoint and emit a code_update.
                    if best_code_path and best_code_path.exists():
                        code = reflective_search_runner.load_best_code_from_json(
                            best_code_path
                        )
                        if code:
                            await self.emit(
                                "code_update",
                                path=str(EVO_GPT_PATH),
                                code=code,
                                nan_check={
                                    "passed": True,
                                    "attempts": last_generation if last_generation >= 0 else 0,
                                    "last_error": None,
                                },
                                final=False,
                                generation=ev.generation,
                            )
                            _stash_designed_code(self.session, ds_id, code)
                            await store.update(self.session)
                elif ev.phase == "finished":
                    finished_ok = True
                    if ev.objective is not None and not (ev.objective in (float("inf"), float("-inf"))):
                        best_objective = ev.objective
        finally:
            canceller.cancel()
            try:
                await canceller
            except asyncio.CancelledError:
                pass

        # Make sure the child actually exited
        if job.is_running():
            await job.terminate()
        exit_code = job.process.returncode if job.process.returncode is not None else -1
        elapsed_s = time.time() - start_ts

        await self.tool_done(
            call_id, "reflective_search_runner.start_reflective_search",
            duration_ms=elapsed_s * 1000,
            result={
                "exit_code": exit_code,
                "generations_seen": max(last_generation, 0),
                "best_objective": best_objective,
                "best_code_path": str(best_code_path) if best_code_path else None,
                "output_dir": str(out_dir),
            },
            error=None if exit_code in (0, None) else f"Reflective search exited with {exit_code}",
        )

        # Capture the final generated metric as an artifact.
        if EVO_GPT_PATH.exists() and finished_ok:
            try:
                final_code = EVO_GPT_PATH.read_text(encoding="utf-8")
            except Exception:
                final_code = ""
            if final_code:
                _stash_designed_code(self.session, ds_id, final_code)
                await self.emit(
                    "code_update",
                    path=str(EVO_GPT_PATH),
                    code=final_code,
                    nan_check={"passed": True, "attempts": last_generation, "last_error": None},
                    final=True,
                )
                artifact = Artifact(
                    name="cddi_function.py",
                    kind="py",
                    path=str(EVO_GPT_PATH),
                    size=EVO_GPT_PATH.stat().st_size,
                    preview={
                        "mode": REFLECTIVE_SEARCH_MODE,
                        "generations": last_generation,
                        "best_objective": best_objective,
                        "output_dir": str(out_dir),
                    },
                )
                d = self.session.add_artifact(artifact)
                await store.update(self.session)
                await self.artifact(name=d["name"], kind=d["kind"], path=d["path"], **(d.get("preview") or {}))

        cancelled = exit_code not in (0, None)
        if cancelled:
            await self.failed(
                f"Reflective search exited with code {exit_code} after {elapsed_s:.0f}s "
                f"(generations seen: {last_generation})"
            )
            return {
                "passed": False,
                "error": f"exit_code={exit_code}",
                "mode": REFLECTIVE_SEARCH_MODE,
                "generations_seen": last_generation,
                "best_objective": best_objective,
            }
        await self.finished(
            result={
                "passed": True,
                "mode": REFLECTIVE_SEARCH_MODE,
                "generations_seen": last_generation,
                "best_objective": best_objective,
                "elapsed_s": elapsed_s,
                "output_dir": str(out_dir),
            }
        )
        await self.say(
            f"Reflective Search finished after {last_generation} generation(s) in "
            f"{elapsed_s:.1f}s. Best objective: {best_objective}. "
            "Next: run CDDI Compute on the full batch."
        )
        return {
            "passed": True,
            "mode": REFLECTIVE_SEARCH_MODE,
            "generations_seen": last_generation,
            "best_objective": best_objective,
            "output_dir": str(out_dir),
        }

    # ==========================================

    async def _emit_artifact(self, result: DesignResult) -> None:
        gpt_path = Path(result.gpt_path)
        size = gpt_path.stat().st_size if gpt_path.exists() else 0
        artifact = Artifact(
            name="cddi_function.py",
            kind="py",
            path=str(gpt_path),
            size=size,
            preview={
                "function_name": result.function_name,
                "attempts": result.attempts,
                "sample_losses": result.sample_losses,
            },
        )
        d = self.session.add_artifact(artifact)
        await store.update(self.session)
        await self.artifact(
            name=d["name"],
            kind=d["kind"],
            path=d["path"],
            **(d.get("preview") or {}),
        )
