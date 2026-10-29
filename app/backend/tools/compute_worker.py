"""Standalone worker that runs MMD compute for one Compute-Agent call.

Why a separate Python module launched via ``asyncio.create_subprocess_exec``
instead of ``multiprocessing.{spawn,fork,forkserver}``?

In production we hit two repeated failures with ``multiprocessing``:

  1. ``spawn`` mode re-imports the entire ``app.backend.*`` tree (FastAPI,
     openai, pandas, scipy, our own code…). That added ~10-20 s of pure
     import latency where nothing was written to the progress queue — the
     UI showed a static "running" tool_call card and the user (rightly)
     thought the backend was hung.
  2. ``fork`` / ``forkserver`` are near-instant but **not safe** under a
     multi-threaded asyncio server: the child inherits every lock the
     parent held at fork(), but not the threads that own them, so any
     ``logging`` / ``importlib`` lock that happened to be held while
     uvicorn was juggling a request would deadlock the child forever.

A vanilla ``asyncio.create_subprocess_exec(sys.executable, "-m", ...)``
side-steps both problems:

  * No multiprocessing IPC bookkeeping; communication is JSON-per-line on
    stdout — flushed eagerly so the very first ``{"phase": "start"}``
    line lands in the parent within a fraction of a second.
  * The child is a completely fresh Python interpreter that imports
    *only* the small subset it actually needs (``app.backend.services.
    mmd_runner`` plus its transitive deps). Imports happen in a clean
    process with no inherited locks.
  * ``proc.kill()`` is a clean SIGKILL that immediately reaps every
    resource — perfect for our 90 s hard-timeout escape hatch.

Protocol (worker -> parent over stdout):
    Each progress event is **one line** of compact JSON. The parent reads
    them with ``await proc.stdout.readline()``. Any line that fails to
    parse as JSON is forwarded to ``log`` as a diagnostic.

Required CLI arguments::

    python -m app.backend.tools.compute_worker \\
        --data-folder /path/to/dianchi/First_200s \\
        --model-file  Model_4.xlsx \\
        --out-csv     /path/to/session/mmd_results.csv \\
        [--char-file First_200s_indicator.xlsx]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# IMPORTANT: line-buffered output. Without this, Python may buffer stdout in
# chunks (since stdout is a pipe, not a TTY) and the parent's `readline()`
# would block for tens of seconds waiting for the buffer to flush — defeating
# the whole point of running in a subprocess. `reconfigure` is 3.7+ and we
# pin Python >= 3.9, so it's safe.
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]


def _emit(payload: dict) -> None:
    """Write one JSON line to stdout (parent reads it via readline)."""
    try:
        print(json.dumps(payload, ensure_ascii=False), flush=True)
    except Exception:
        # last-resort: if even the serialiser fails, write a fallback line so
        # the parent can keep its read loop alive.
        sys.stdout.write('{"phase":"_error","error":"serialise failed"}\n')
        sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", required=True, type=Path)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--char-file", default=None)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-end", type=int, default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args(argv)

    _emit({"phase": "_worker_started", "pid": __import__("os").getpid(), "t": time.time()})

    # Import mmd_runner LATE so an import failure can still be reported
    # via a clean JSON line (rather than dying silently before we have a
    # protocol).
    try:
        from app.backend.services import mmd_runner
    except Exception as exc:  # noqa: BLE001
        _emit({
            "phase": "_failed",
            "error": f"failed to import mmd_runner: {exc}",
            "traceback": traceback.format_exc(limit=4),
        })
        return 2

    _emit({"phase": "_imports_done", "t": time.time()})

    # Load the LLM-injected MMD_cal_new from dianchi03/mmd_tools_new.py.
    try:
        mmd_fn = mmd_runner.load_injected_function()
    except Exception as exc:  # noqa: BLE001
        _emit({
            "phase": "_failed",
            "error": f"load_injected_function failed: {exc}",
            "traceback": traceback.format_exc(limit=4),
        })
        return 3

    _emit({"phase": "_mmd_loaded", "fn_name": getattr(mmd_fn, "__name__", "?"), "t": time.time()})

    def on_progress(prog: "mmd_runner.MMDProgress") -> None:
        # Translate the dataclass into a small JSON-able dict. Field set
        # mirrors what the parent's `_log_llm` / consume() code already
        # expects.
        try:
            _emit({
                "phase": prog.phase,
                "index": int(prog.index),
                "total": int(prog.total),
                "cell_num": prog.cell_num,
                "filename": prog.filename,
                "mmd_loss": prog.mmd_loss,
                "sample_count": prog.sample_count,
                "error": prog.error,
                "elapsed_s": float(prog.elapsed_s),
            })
        except Exception:
            _emit({"phase": "_error", "error": "on_progress dump failed"})

    try:
        summary = mmd_runner.run_full_pipeline(
            data_folder=args.data_folder,
            model_filename=args.model_file,
            char_filename=args.char_file,
            out_csv=args.out_csv,
            mmd_fn=mmd_fn,
            row_start=args.row_start,
            row_end=args.row_end,
            scale=args.scale,
            on_progress=on_progress,
        )
    except Exception as exc:  # noqa: BLE001
        _emit({
            "phase": "_failed",
            "error": f"run_full_pipeline raised: {exc}",
            "traceback": traceback.format_exc(limit=6),
        })
        return 4

    # Final summary message. `correlation` may contain numpy types that we
    # need to coerce before json.dumps; rely on default=str as the safety
    # net.
    try:
        _emit({"phase": "_done", "summary": summary})
    except TypeError:
        _emit({
            "phase": "_done",
            "summary": json.loads(json.dumps(summary, default=str)),
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
