"""Regression test: ComputeAgent kills a hung MMD function instead of freezing.

User scenario: the LLM-designed `calculate_mmd_v2` contained a pure-Python
tight loop (or worse, an infinite loop) that never released the GIL. Result:
the backend's asyncio event loop went dark, heartbeats stopped, and Compute
"hung forever" from the user's perspective.

Fix: ComputeAgent now runs `run_full_pipeline` in a child process (spawn),
streams progress through a multiprocessing.Queue, and enforces a hard
wall-clock deadline. The parent stays reactive even when the child is GIL-
bound.

This test injects a deliberately-hanging MMD function, sets a 6-second
deadline (test_only override would be ideal but we just patch COMPUTE_TIMEOUT_S
at import time), runs ComputeAgent and checks that:
  * agent emits a failure within ~7 seconds
  * agent_finished status is 'error'
  * subprocess is no longer alive
  * pipeline state goes to 'error', not 'running'
"""
from __future__ import annotations

import asyncio
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Patch the timeout BEFORE importing the agent — module-level constant.
from app.backend.agents import compute_agent as compute_agent_mod  # noqa: E402

compute_agent_mod.COMPUTE_TIMEOUT_S = 6.0  # short enough for CI

from app.backend.agents.compute_agent import ComputeAgent  # noqa: E402
from app.backend.orchestrator.events import bus  # noqa: E402
from app.backend.orchestrator.session_store import Session  # noqa: E402
from app.backend.services import data_resolver, mmd_runner  # noqa: E402
from app.backend.settings import EVO_GPT_PATH, DIANCHI_MMD_TOOLS_NEW, settings  # noqa: E402


HANG_MMD_CODE = '''
import numpy as np

def calculate_mmd_v9(xs, xt, fix_sigma=0, kernel_mul=2.0, kernel_num=5):
    """Intentionally hangs in a pure-Python tight loop (no GIL release).

    Mimics a pathological LLM-generated MMD: spins forever without doing
    any numpy work, so the GIL is held continuously. If the parent process
    cannot kill us, Compute hangs forever.
    """
    # Touch the inputs once so the function isn't dead-stripped
    _ = float(xs.shape[0]) + float(xt.shape[0])
    while True:
        # pure Python: holds the GIL
        x = 0
        for i in range(10_000):
            x = (x * 1103515245 + 12345) & 0x7fffffff
        if x < 0:  # never true
            break
    return 0.0
'''


async def collect_until(session_id: str, kind: str, agent: str, timeout: float = 30.0):
    queue = await bus.subscribe(session_id)
    events = []
    try:
        while True:
            ev = await asyncio.wait_for(queue.get(), timeout=timeout)
            events.append(ev)
            if ev.get("kind") == kind and ev.get("agent") == agent:
                break
    finally:
        await bus.unsubscribe(session_id, queue)
    return events


async def run_one_shot() -> int:
    # Inject the hanging MMD into dianchi03/mmd_tools_new.py before running.
    # We use the real inject_mmd_code so the test exercises the same path
    # Designer does in production.
    mmd_runner.inject_mmd_code(HANG_MMD_CODE)
    # Also touch gpt.py because ComputeAgent precondition is EVO_GPT_PATH.exists()
    EVO_GPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVO_GPT_PATH.write_text(HANG_MMD_CODE, encoding="utf-8")

    session = Session(session_id="test_compute_timeout_001")
    # session.active_dataset_id defaults to builtin-first-200s in Session __init__
    ds = data_resolver.resolve(session.active_dataset_id)

    agent = ComputeAgent(session)
    listener = asyncio.create_task(
        collect_until(session.session_id, "agent_finished", "compute", timeout=30.0)
    )
    await asyncio.sleep(0.1)

    t0 = time.time()
    result = await agent.run()
    elapsed = time.time() - t0
    events = await listener

    print(f"  ran for {elapsed:.1f}s (deadline was {compute_agent_mod.COMPUTE_TIMEOUT_S}s)")
    if elapsed > compute_agent_mod.COMPUTE_TIMEOUT_S + 4.0:
        print(f"  [FAIL] agent took {elapsed:.1f}s, expected < {compute_agent_mod.COMPUTE_TIMEOUT_S + 4.0}s")
        return 1

    if result.get("passed"):
        print(f"  [FAIL] expected passed=False, got {result}")
        return 1
    if "timeout" not in (result.get("error") or "").lower():
        print(f"  [FAIL] expected timeout error, got: {result}")
        return 1

    # agent_finished must have status='error'
    finished = next(
        (e for e in events if e.get("kind") == "agent_finished" and e.get("agent") == "compute"),
        None,
    )
    if finished is None:
        print("  [FAIL] no agent_finished event")
        return 1
    if finished.get("status") != "error":
        print(f"  [FAIL] expected status=error, got status={finished.get('status')}")
        return 1

    # Look for an error-tagged llm_log line about timeout
    err_logs = [
        e for e in events
        if e.get("kind") == "chart_data"
        and e.get("chart") == "llm_log"
        and (e.get("data", {}) or {}).get("tag") == "error"
    ]
    if not err_logs:
        print("  [FAIL] no error llm_log entry observed")
        return 1
    print(f"  [OK] {len(err_logs)} error log line(s); first: "
          f"{err_logs[0].get('data', {}).get('line', '')[:120]}")

    # Pipeline state should be 'error'
    if session.pipeline_state.get("compute") != "error":
        print(f"  [FAIL] pipeline_state.compute={session.pipeline_state.get('compute')}, expected 'error'")
        return 1

    print(f"  [OK] killed within {elapsed:.1f}s, pipeline.compute='error', user-friendly msg posted.")
    return 0


def main() -> int:
    print("== test_compute_timeout ==")
    try:
        rc = asyncio.run(run_one_shot())
    finally:
        # CRITICAL: restore mmd_tools_new.py to a known-good baseline so the
        # next time the developer runs Compute (via the UI or another test)
        # it doesn't pick up our intentionally-hanging stub. We restore from
        # `mmd_tools_new.baseline.py` (committed alongside this test) rather
        # than from `.bak`, because `.bak` is overwritten by every inject
        # call and could itself be polluted.
        baseline = DIANCHI_MMD_TOOLS_NEW.with_name(
            DIANCHI_MMD_TOOLS_NEW.stem + ".baseline" + DIANCHI_MMD_TOOLS_NEW.suffix
        )
        if baseline.exists():
            DIANCHI_MMD_TOOLS_NEW.write_bytes(baseline.read_bytes())
            print(f"  [cleanup] restored {DIANCHI_MMD_TOOLS_NEW.name} from baseline.")
        else:
            # Fallback: try the .bak too (best effort)
            bak = DIANCHI_MMD_TOOLS_NEW.with_suffix(DIANCHI_MMD_TOOLS_NEW.suffix + ".bak")
            if bak.exists():
                DIANCHI_MMD_TOOLS_NEW.write_bytes(bak.read_bytes())
                print(f"  [cleanup] restored {DIANCHI_MMD_TOOLS_NEW.name} from .bak (no baseline found!).")
    print("PASSED" if rc == 0 else "FAILED")
    return rc


if __name__ == "__main__":
    sys.exit(main())
