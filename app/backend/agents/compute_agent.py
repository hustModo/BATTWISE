"""CDDI Compute Agent.

Pipeline:
    1. Pick the CDDI function:
       - inject Evo_Textgrad gpt.py into dianchi03/mmd_tools_new.py
       - reload it
    2. Run mmd_runner.compute_mmd_for_folder on the active dataset:
       - emit one 'tool_call' for the overall job
       - emit incremental 'chart_data(mmd_progress)' every N cells
       - emit per-cell short 'thinking' log lines so the timeline stays live
    3. Save mmd_results.csv into session workspace
    4. Emit chart_data(mmd_hist) + artifact + correlation summary
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from app.backend.agents.base import AgentBase
from app.backend.orchestrator.session_store import Artifact, store
from app.backend.services import data_resolver, mmd_runner
from app.backend.settings import EVO_GPT_PATH, settings


PROGRESS_STRIDE = 1  # push live chart_data every cell — keeps the UI heartbeat
# tight so users never see a "stuck running" feeling. Was 5; bumped down after
# user-reported "compute hung" issues that were actually just buffered ticks.

# Hard ceiling for one Compute run (all 71 cells). Baseline runs are sub-second;
# anything past this is overwhelmingly likely to be an LLM-generated MMD
# function with an infinite loop or pathological complexity. We abort cleanly
# and tell the user to re-design rather than freezing the backend.
#
# Tightened from 90 → 60 s after the user reported "卡死完了，退出了" — when the
# LLM keeps producing pathological functions and the frontend auto-retries
# once, the user-perceived freeze becomes 2 × timeout, so a shorter ceiling
# directly improves recovery time. Healthy runs are ~0.4 s; we still have
# huge headroom for legitimately heavy designs.
COMPUTE_TIMEOUT_S = 60.0

# Heartbeat cadence for the user-facing "…still working" llm_log lines.
# Also drives the maximum time we wait on stdout.readline() before checking
# the deadline. Smaller values give faster responsiveness but spam more
# log lines if the compute is genuinely slow.
HEARTBEAT_S = 2.5

# Path to the repo root (4 levels up from compute_agent.py:
#   .../repo/app/backend/agents/compute_agent.py
# so .parent.parent.parent.parent = repo root). We pass this on PYTHONPATH
# to the worker subprocess so `python -m app.backend.tools.compute_worker`
# resolves correctly regardless of where the user launched the backend
# from.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)


def _histogram(losses: List[float], bins: int = 25) -> Dict[str, List[float]]:
    if not losses:
        return {"bin_edges": [], "counts": []}
    arr = np.asarray(losses, dtype=float)
    counts, edges = np.histogram(arr, bins=bins)
    return {
        "bin_edges": edges.tolist(),
        "counts": counts.astype(int).tolist(),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
    }


class ComputeAgent(AgentBase):
    agent_id = "compute"
    node_id = "compute"

    async def _log_llm(self, tag: str, line: str) -> None:
        """Stream a single 'runtime activity' line into the llm_log frontend channel.

        The frontend `LLMLogStream` component ring-buffers these lines and
        renders them with colour by tag. Tags: info | ok | warn | error.
        We reuse this channel from Designer + Compute + Cluster so users get
        ONE live terminal showing exactly what each agent is doing.
        """
        await self.emit(
            "chart_data",
            chart="llm_log",
            data={"line": line[:500], "tag": tag, "agent": self.agent_id},
        )

    async def run(self, user_text: str = "", **_: Any) -> Dict[str, Any]:  # type: ignore[override]
        await self.started()
        await self._log_llm("info", "Compute agent started.")

        try:
            ds = data_resolver.resolve(self.session.active_dataset_id)
        except FileNotFoundError as exc:
            await self.failed(str(exc))
            return {"passed": False, "error": str(exc)}

        scoped = self.session.context.get("designed_code_by_dataset") or {}
        scoped_entry = scoped.get(ds.dataset_id) if isinstance(scoped, dict) else None
        scoped_code = scoped_entry.get("code") if isinstance(scoped_entry, dict) else None
        if not scoped_code and not EVO_GPT_PATH.exists():
            await self.failed(
                f"No designed CDDI function found for {ds.dataset_id}. Please run the Designer first."
            )
            await self.say(
                f"Please run CDDI Designer for `{ds.name}` before computing. "
                "No dataset-specific CDDI code is available."
            )
            return {"passed": False, "error": "dataset CDDI code missing"}

        await self.thinking(
            f"Injecting designed CDDI function into mmd_tools_new.py and running on dataset "
            f"`{ds.name}` ({ds.cell_count} cells)."
        )
        await self._log_llm("info", f"Reading designed CDDI code for {ds.dataset_id}…")

        inject_call_id = await self.tool_call(
            "mmd_runner.inject_mmd_code",
            {"source": str(EVO_GPT_PATH), "target": "dianchi03/mmd_tools_new.py"},
            status="running",
        )
        try:
            code = str(scoped_code) if scoped_code else EVO_GPT_PATH.read_text(encoding="utf-8")
            await self._log_llm("info", f"Read {len(code)} chars; rewriting mmd_tools_new.py…")
            target = mmd_runner.inject_mmd_code(code)
        except Exception as exc:
            await self.tool_done(
                inject_call_id, "mmd_runner.inject_mmd_code", duration_ms=0, error=str(exc)
            )
            await self._log_llm("error", f"Injection failed: {exc}")
            await self.failed(f"Injection failed: {exc}")
            return {"passed": False, "error": str(exc)}
        await self.tool_done(
            inject_call_id,
            "mmd_runner.inject_mmd_code",
            duration_ms=0,
            result={"written": str(target)},
        )
        await self._log_llm("ok", f"Injected CDDI code into {Path(target).name}.")

        await self._log_llm("info", "Sanity-loading injected function in parent…")
        try:
            # Parent-side load is purely a fail-fast import check: syntax or
            # signature errors should surface NOW (before we spawn). The
            # actual compute happens in the child, which re-loads the same
            # file from disk in its own interpreter.
            _ = mmd_runner.load_injected_function()
        except Exception as exc:
            await self._log_llm("error", f"Sanity import failed: {exc}")
            await self.failed(f"Reload of injected function failed: {exc}")
            return {"passed": False, "error": str(exc)}
        await self._log_llm("ok", "Function imports cleanly. Spawning compute subprocess…")

        out_csv = settings.session_workspace(self.session.session_id) / f"cddi_results_{ds.dataset_id}.csv"

        run_call_id = await self.tool_call(
            "mmd_runner.compute_mmd_for_folder",
            {
                "data_folder": str(ds.folder),
                "model_filename": ds.model_file,
                "char_filename": ds.char_file,
                "out_csv": str(out_csv),
            },
            status="running",
        )

        results_box: Dict[str, Any] = {}

        # ------------------------------------------------------------------
        # Spawn the compute worker as a CLEAN python -m subprocess.
        #
        # Why not multiprocessing? See app/backend/tools/compute_worker.py
        # for the full rationale; tl;dr: spawn is too slow (10-20 s of
        # re-imports), fork deadlocks under multi-threaded uvicorn (lock
        # inheritance from logging/importlib), forkserver was a half
        # solution. A vanilla `asyncio.create_subprocess_exec` gives us
        # the fastest start (<= 1-2 s of pure Python startup + minimum
        # imports), no shared lock state, and clean `proc.kill()` for the
        # 90 s hard timeout.
        # ------------------------------------------------------------------
        worker_cmd = [
            sys.executable,
            "-u",  # unbuffered (defensive — worker also does reconfigure)
            "-m",
            "app.backend.tools.compute_worker",
            "--data-folder", str(ds.folder),
            "--model-file", ds.model_file,
            "--out-csv", str(out_csv),
        ]
        if ds.char_file:
            worker_cmd += ["--char-file", ds.char_file]

        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            _REPO_ROOT + (os.pathsep + existing_pp if existing_pp else "")
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *worker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=_REPO_ROOT,
            )
        except Exception as exc:  # noqa: BLE001
            await self._log_llm("error", f"Failed to launch worker subprocess: {exc}")
            await self.tool_done(
                run_call_id,
                "mmd_runner.compute_mmd_for_folder",
                duration_ms=0,
                error=f"subprocess launch failed: {exc}",
            )
            await self.failed(f"Could not start compute worker: {exc}")
            return {"passed": False, "error": str(exc)}

        await self._log_llm(
            "info",
            f"Compute worker started (subprocess pid={proc.pid}). Streaming progress…",
        )

        # Hold the deadline outside the consumer so we can SIGKILL the child
        # from the same place that emits the timeout message.
        deadline = time.time() + COMPUTE_TIMEOUT_S
        timeout_flag: Dict[str, bool] = {"hit": False}

        async def consume() -> None:
            running_losses: List[float] = []
            running_cells: List[int] = []
            last_push = 0
            last_log_push = 0
            heartbeat_count = 0

            while True:
                # Hard deadline check — covers any worker hang including
                # GIL-bound python loops (since the worker is a separate
                # OS process, the parent's event loop is never blocked by
                # them, but we still must abort the whole compute).
                if time.time() > deadline:
                    timeout_flag["hit"] = True
                    return
                # Cooperative chat-driven cancel. Pipeline sets this flag
                # when the user types "stop" / "abort" / "停" while we're
                # streaming. We kill the worker subprocess in the parent's
                # finally block; here we just exit the consumer loop.
                if self.session.context.get("cancel_compute"):
                    await self._log_llm("warn", "Cancellation requested via chat — terminating compute worker.")
                    self.session.context["cancel_compute"] = False
                    self.session.context["compute_cancelled"] = True
                    await store.update(self.session)
                    return

                # Read one JSON line. We bound the wait by HEARTBEAT_S so
                # we can emit a heartbeat / re-check the deadline even if
                # the worker goes silent for a long time.
                try:
                    line_bytes = await asyncio.wait_for(
                        proc.stdout.readline(),  # type: ignore[union-attr]
                        timeout=HEARTBEAT_S,
                    )
                except asyncio.TimeoutError:
                    heartbeat_count += 1
                    await self._log_llm(
                        "info",
                        f"…still working ({heartbeat_count * HEARTBEAT_S:.0f}s since last event)",
                    )
                    continue

                if not line_bytes:
                    # EOF -> worker exited. Check rc & stderr in the
                    # parent's finally block.
                    return
                # Any real bytes reset the silent-streak counter.
                heartbeat_count = 0

                line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    # The worker only emits JSON; anything else is a stray
                    # print() from a 3rd-party lib. Forward as a low-noise
                    # log line so we can debug if something weird shows up.
                    await self.log(f"worker stdout (non-json): {line[:200]}", level="debug")
                    continue

                phase = item.get("phase")

                # Worker-internal sentinel phases (prefixed with `_`).
                if phase == "_worker_started":
                    await self._log_llm(
                        "ok",
                        f"Worker bootstrapped (pid={item.get('pid')}). Importing dependencies…",
                    )
                    continue
                if phase == "_imports_done":
                    await self._log_llm("ok", "Imports complete. Loading injected CDDI code…")
                    continue
                if phase == "_mmd_loaded":
                    await self._log_llm(
                        "ok",
                        f"Loaded CDDI function `{item.get('fn_name')}`. Running pipeline…",
                    )
                    continue
                if phase == "_done":
                    results_box["summary"] = item.get("summary")
                    return
                if phase == "_failed":
                    results_box["error"] = item.get("error", "unknown")
                    results_box["tb"] = item.get("traceback")
                    return
                if phase == "_error":
                    await self._log_llm("warn", f"Worker reported: {item.get('error')}")
                    continue

                # MMDProgress-style phases. Wrap into attribute access so
                # the existing branching logic below stays unchanged.
                class _P:
                    __slots__ = (
                        "phase", "index", "total", "cell_num", "filename",
                        "mmd_loss", "sample_count", "error", "elapsed_s",
                    )

                p = _P()
                for k in _P.__slots__:
                    setattr(p, k, item.get(k))
                if p.phase == "start":
                    # First tick: model hasn't loaded yet. Emit a heartbeat
                    # so the user doesn't perceive a stall while pandas reads
                    # the (potentially multi-MB) model xlsx.
                    await self._log_llm(
                        "info",
                        f"Loading model file from {Path(p.filename or ds.folder).name}…",
                    )
                elif p.phase == "model_loaded":
                    n_cells = len(mmd_runner._list_cell_files(ds.folder))
                    await self.thinking(
                        f"Model loaded ({p.filename}, {p.sample_count} samples). "
                        f"Computing CDDI for {n_cells} cells..."
                    )
                    await self._log_llm(
                        "ok",
                        f"Model loaded: {p.sample_count} samples. Iterating {n_cells} cells…",
                    )
                elif p.phase == "cell":
                    running_losses.append(float(p.mmd_loss))
                    running_cells.append(int(p.cell_num))
                    # ALWAYS push mmd_progress at PROGRESS_STRIDE=1 (every cell).
                    if p.index - last_push >= PROGRESS_STRIDE or p.index == p.total:
                        last_push = p.index
                        await self.emit(
                            "chart_data",
                            chart="mmd_progress",
                            data={
                                "index": p.index,
                                "total": p.total,
                                "percent": round(100.0 * p.index / max(p.total, 1), 1),
                                "elapsed_s": round(p.elapsed_s, 2),
                                "last_cell": p.cell_num,
                                "last_loss": p.mmd_loss,
                            },
                        )
                    # Also drop a line into the llm_log stream every 5 cells so
                    # the user sees a textual heartbeat even if the chart is
                    # offscreen. The conditional avoids spamming for big runs.
                    if p.index - last_log_push >= 5 or p.index == p.total:
                        last_log_push = p.index
                        await self._log_llm(
                            "info",
                            f"[{p.index}/{p.total}] Cell{p.cell_num} loss={p.mmd_loss:.4f} "
                            f"({p.elapsed_s:.1f}s elapsed)",
                        )
                elif p.phase == "cell_skipped":
                    await self.log(
                        f"skip Cell{p.cell_num} ({p.filename}): {p.error}",
                        level="warning",
                    )
                    await self._log_llm(
                        "warn",
                        f"skip Cell{p.cell_num}: {p.error}",
                    )
                elif p.phase == "done":
                    results_box["progress"] = p
                    results_box["losses"] = running_losses
                    results_box["cells"] = running_cells
                    await self._log_llm(
                        "ok",
                        f"Compute finished in {p.elapsed_s:.1f}s, "
                        f"{len(running_losses)}/{p.total} cells computed.",
                    )

        start_ts = time.time()
        consumer = asyncio.create_task(consume())
        try:
            await consumer
        finally:
            # If we hit the deadline OR the consumer fell out because the
            # worker died unexpectedly, force-kill the subprocess before
            # we return so it cannot keep grinding CPU after the agent is
            # gone. SIGTERM first; SIGKILL if it doesn't quit in 2 s.
            if proc.returncode is None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass
            # Drain any remaining stderr for diagnostics if the worker
            # crashed (best-effort; ignore failures).
            if "error" in results_box and proc.stderr is not None:
                try:
                    stderr_data = await asyncio.wait_for(
                        proc.stderr.read(), timeout=1.0
                    )
                    if stderr_data:
                        await self._log_llm(
                            "warn",
                            f"Worker stderr: {stderr_data.decode('utf-8','replace')[:400]}",
                        )
                except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                    pass

        # ---- post-mortem: did we time out / cancel / fail / succeed? ----
        if self.session.context.pop("compute_cancelled", False):
            await self.tool_done(
                run_call_id,
                "mmd_runner.compute_mmd_for_folder",
                duration_ms=(time.time() - start_ts) * 1000,
                error="cancelled by user",
            )
            await self.emit(
                "agent_finished",
                error="cancelled by user",
                status="cancelled",
                reason="user_cancel",
            )
            self.session.set_node(self.node_id, "idle")
            await store.update(self.session)
            await self.say("Compute cancelled. Tell me what to do next.")
            return {"passed": False, "error": "cancelled", "cancelled": True}

        if timeout_flag["hit"]:
            await self._log_llm(
                "error",
                f"Compute exceeded {COMPUTE_TIMEOUT_S:.0f}s — the designed "
                "CDDI function is likely stuck (infinite loop or O(n³+) "
                "complexity). Subprocess killed. Re-run Designer or switch to "
                "Reflective Search mode for a more robust function.",
            )
            await self.tool_done(
                run_call_id,
                "mmd_runner.compute_mmd_for_folder",
                duration_ms=(time.time() - start_ts) * 1000,
                error=f"timeout >{COMPUTE_TIMEOUT_S:.0f}s (child killed)",
            )
            # Mark this as a compute_timeout in the failed event payload so
            # the frontend can decide whether to auto-retry (Design again →
            # Compute). The retry counter lives in the frontend store; we
            # only signal the cause here.
            await self.emit(
                "agent_finished",
                error=f"compute timeout >{COMPUTE_TIMEOUT_S:.0f}s",
                status="error",
                reason="compute_timeout",
            )
            self.session.set_node(self.node_id, "error")
            await self.say(
                f"Compute aborted after {COMPUTE_TIMEOUT_S:.0f}s: the latest "
                "designed CDDI function hangs on this dataset. "
                "Auto-retrying with a fresh Designer attempt (LLM is stochastic, "
                "another try usually works). If this fails twice in a row, "
                "toggle **Mode → Reflective Search** for a more thorough search."
            )
            return {"passed": False, "error": "compute timeout"}

        if "error" in results_box:
            err = results_box["error"]
            await self._log_llm("error", f"Subprocess error: {err}")
            await self.tool_done(
                run_call_id,
                "mmd_runner.compute_mmd_for_folder",
                duration_ms=(time.time() - start_ts) * 1000,
                error=err,
            )
            await self.failed(f"Compute subprocess failed: {err}")
            return {"passed": False, "error": err}

        summary = results_box.get("summary")
        if summary is None:
            await self._log_llm("error", "Subprocess returned no summary; treating as failure.")
            await self.failed("Compute subprocess produced no result.")
            return {"passed": False, "error": "no summary"}

        elapsed = time.time() - start_ts
        await self.tool_done(
            run_call_id,
            "mmd_runner.compute_mmd_for_folder",
            duration_ms=elapsed * 1000,
            result={
                "saved_to": summary["saved_to"],
                "total_cells": summary["total_cells"],
                "skipped": len(summary["skipped"]),
                "computed": len(summary["results"]),
            },
        )

        losses = [r["MMD_Loss"] for r in summary["results"]]
        hist = _histogram(losses, bins=25)
        per_cell = [
            {"cell_num": r["Cell_num"], "mmd_loss": r["MMD_Loss"]}
            for r in summary["results"]
        ]
        await self.emit(
            "chart_data",
            chart="mmd_hist",
            data={
                "histogram": hist,
                "per_cell": per_cell,
                "skipped": summary["skipped"],
            },
        )

        if summary.get("correlation"):
            await self.emit(
                "chart_data",
                chart="mmd_correlation",
                data=summary["correlation"],
            )

        artifact = Artifact(
            name=Path(summary["saved_to"]).name,
            kind="csv",
            path=str(summary["saved_to"]),
            size=Path(summary["saved_to"]).stat().st_size,
            preview={
                "rows": len(summary["results"]),
                "skipped": len(summary["skipped"]),
                "stats": {k: v for k, v in hist.items() if k not in {"bin_edges", "counts"}},
                "correlation": {
                    k: v for k, v in (summary.get("correlation") or {}).items()
                    if k != "features"
                },
            },
        )
        d = self.session.add_artifact(artifact)
        self.session.context["last_cddi_csv"] = d["path"]
        by_dataset = dict(self.session.context.get("ddia_csv_by_dataset") or {})
        by_dataset.update(self.session.context.get("cddi_csv_by_dataset") or {})
        by_dataset[ds.dataset_id] = d["path"]
        self.session.context["cddi_csv_by_dataset"] = by_dataset
        self.session.context["last_mmd_stats"] = hist
        await store.update(self.session)
        await self.artifact(name=d["name"], kind=d["kind"], path=d["path"], **(d.get("preview") or {}))

        corr = summary.get("correlation") or {}
        # mean_sig now uses p_threshold=1.0 (matching the upstream EOH project,
        # see services/mmd_runner.py docstring) — so mean_sig == mean |r| across
        # all indicator features. We also report the count of features that pass
        # the stricter p<0.1 bar for additional context.
        mean_sig = corr.get("mean_sig", 0.0)
        all_mean_abs = corr.get("all_mean_abs", 0.0)
        n_features = len(corr.get("features") or [])
        n_sig_strict = sum(
            1 for f in (corr.get("features") or []) if f.get("p", 1.0) < 0.1
        )

        if mean_sig < 0.10:
            advisory = (
                " CDDI ↔ indicator correlation is very weak. "
                "Consider re-running Designer, switching to Reflective Search, "
                "or checking that the dataset's indicator file is meaningful. "
                "You can still type `Run clustering` to proceed."
            )
        elif mean_sig < 0.25:
            advisory = (
                f" Signal is modest. "
                "Clustering will work, but Reflective Search usually produces a stronger CDDI signal."
            )
        else:
            advisory = " Next: run clustering, or adjust clustering parameters first."

        msg = (
            f"Computed CDDI for {len(summary['results'])} cells in {elapsed:.1f}s "
            f"(skipped {len(summary['skipped'])}). "
            f"mean |r|={mean_sig:.3f} across {n_features} indicator features "
            f"({n_sig_strict} pass p<0.1).{advisory}"
        )
        await self.say(msg)

        await self.finished(
            result={
                "passed": True,
                "computed": len(summary["results"]),
                "skipped": len(summary["skipped"]),
                "csv": d["path"],
                "stats": {k: v for k, v in hist.items() if k not in {"bin_edges", "counts"}},
                "mean_sig": mean_sig,
            }
        )
        return {"passed": True, "computed": len(summary["results"]), "csv": d["path"]}
