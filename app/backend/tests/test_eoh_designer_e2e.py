"""End-to-end test for DesignerAgent in full_eoh mode.

Doesn't actually run battwise_cddi_search/main.py (that would take minutes and need
LLM credentials). Instead we monkey-patch ``eoh_runner.start_eoh`` to spawn a
*fake* worker that mimics the directory + main.log layout the real EOH child
would produce.

The fake worker:
    * creates ``<EVO_DIR>/outputs/output/<test-timestamp>/``
    * writes a believable main.log line-by-line with sleeps in between
    * writes ``best_population_generation_N.json`` files

This exercises:
    * DesignerAgent._run_full_eoh control flow
    * find_new_output_dir
    * tail_log → parse_log_line streaming
    * code_update + chart_data(eoh_progress) emissions
    * the cancel hook (session.context['cancel_eoh'])
"""
from __future__ import annotations

import asyncio
import datetime
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.agents.designer_agent import DesignerAgent  # noqa: E402
from app.backend.orchestrator.events import bus  # noqa: E402
from app.backend.orchestrator.session_store import Session  # noqa: E402
from app.backend.services import reflective_search_runner  # noqa: E402
from app.backend.settings import EVO_DIR  # noqa: E402


_FAKE_CODE = '''import numpy as np

def calculate_mmd_v2(xs: np.ndarray, xt: np.ndarray, fix_sigma=0, kernel_mul=2.0, kernel_num=5) -> float:
    return float(np.sum((xs.mean(axis=0) - xt.mean(axis=0)) ** 2))
'''


async def _fake_worker(out_dir: Path, total_generations: int, line_delay: float = 0.03) -> int:
    """Mimic battwise_cddi_search/main.py producing main.log and best_population_*.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "main.log"

    def stamp() -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,000")

    def write(line: str) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{stamp()}][root][INFO] - {line}\n")

    write("Starting BATTWISE Reflective Metric Search...")
    write("Case num: [2]")
    write("BATTWISE Reflective Metric Search initialized successfully")
    await asyncio.sleep(line_delay)

    for gen in range(total_generations + 1):
        if gen == 0:
            write("Generation 0 (Initial) - BEST SOLUTION:")
        else:
            write(f"Generation {gen}/{total_generations} - BEST SOLUTION:")
        # objective improves slightly each generation
        obj = round(0.5 + 0.03 * gen, 4)
        write(f"  Objective Value: {obj}")
        write("  Algorithm Description: stubbed for test")
        ckpt = out_dir / f"best_population_generation_{gen}.json"
        ckpt.write_text(
            json.dumps(
                {"algorithm": "stub", "code": _FAKE_CODE, "objective": obj},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        write(f"  Saved to: {ckpt}")
        await asyncio.sleep(line_delay)

    write("Best Code Overall: import numpy as np ...")
    write(f"Best Code Path Overall: {out_dir}/best_population_generation_{total_generations}.json")
    write(f"Best Objective Overall: {round(0.5 + 0.03 * total_generations, 4)}")
    write("BATTWISE Reflective Metric Search completed")
    return 0


class _FakeProcess:
    """Mimics asyncio.subprocess.Process to the extent eoh_runner uses it."""

    def __init__(self, task: asyncio.Task) -> None:
        self._task = task
        self.returncode = None
        self.pid = 999_999  # placeholder
        self._done_lock = asyncio.Lock()

    async def wait(self) -> int:
        await self._task
        return self.returncode

    def terminate(self) -> None:
        if not self._task.done():
            self._task.cancel()
        if self.returncode is None:
            self.returncode = -15

    def kill(self) -> None:
        self.terminate()


def _make_fake_start_eoh(total_generations: int = 3, line_delay: float = 0.03):
    async def fake_start_eoh(**kwargs):
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_test")
        out_dir = EVO_DIR / "outputs" / "output" / ts
        task = asyncio.create_task(_fake_worker(out_dir, total_generations, line_delay))

        proc = _FakeProcess(task)

        async def _settle() -> None:
            try:
                rc = await task
            except asyncio.CancelledError:
                rc = -15
            except Exception:
                rc = 1
            proc.returncode = rc

        # background settler updates returncode when fake worker finishes
        asyncio.create_task(_settle())

        job = reflective_search_runner.ReflectiveSearchJob(
            process=proc,  # type: ignore[arg-type]
            started_at=time.time(),
            pid=proc.pid,
        )
        return job

    return fake_start_eoh


async def run_happy_path() -> int:
    session = Session(session_id="test_eoh_happy_001")
    session.mode = "full_eoh"
    session.active_dataset_id = "builtin-first-200s"

    saved_start = reflective_search_runner.start_reflective_search
    reflective_search_runner.start_reflective_search = _make_fake_start_eoh(
        total_generations=3,
        line_delay=0.02,
    )
    try:
        agent = DesignerAgent(session)
        history_before = len(bus.history(session.session_id))
        result = await agent.run(user_text="evolve a better MMD")
        history = bus.history(session.session_id)
        new_events = history[history_before:]

        # Inspect
        assert result.get("passed") is True, f"happy-path failed: {result}"
        assert result.get("mode") == "reflective_search"
        assert result.get("generations_seen") == 3, result

        eoh_chart_events = [
            e for e in new_events
            if e.get("kind") == "chart_data"
            and e.get("chart") == "reflective_search_progress"
        ]
        gen_events = [e for e in eoh_chart_events if e.get("data", {}).get("phase") == "generation"]
        obj_events = [e for e in eoh_chart_events if e.get("data", {}).get("phase") == "best_objective"]
        code_updates = [e for e in new_events if e.get("kind") == "code_update"]
        artifacts = [e for e in new_events if e.get("kind") == "artifact"]
        finished = [e for e in new_events if e.get("kind") == "agent_finished"]

        assert len(gen_events) == 4, f"expected 4 generation events, got {len(gen_events)}"
        assert obj_events, "expected best_objective events"
        # objectives should be 0.5, 0.53, 0.56, 0.59
        objs = [e.get("data", {}).get("objective") for e in obj_events]
        assert objs == [0.5, 0.53, 0.56, 0.59], objs
        assert len(code_updates) >= 4, f"expected ≥4 code_updates, got {len(code_updates)}"
        assert artifacts, "expected at least one artifact event"
        assert session.pipeline_state.get("designer") == "done", session.pipeline_state
        assert session.context.get("last_designed_code"), "last_designed_code not stored"

        print(f"  events: {len(new_events)} (chart={len(eoh_chart_events)}, "
              f"code_update={len(code_updates)}, artifact={len(artifacts)})")
        print(f"  objectives: {objs}")
        print(f"  pipeline_state: {session.pipeline_state}")
        print(f"  agent_finished result: {finished[-1].get('result') if finished else None}")
        return 0
    finally:
        reflective_search_runner.start_reflective_search = saved_start


async def run_cancel_path() -> int:
    session = Session(session_id="test_eoh_cancel_001")
    session.mode = "full_eoh"
    session.active_dataset_id = "builtin-first-200s"

    saved_start = reflective_search_runner.start_reflective_search
    # Long fake run so we can cancel partway through.
    reflective_search_runner.start_reflective_search = _make_fake_start_eoh(
        total_generations=50,
        line_delay=0.2,
    )
    try:
        agent = DesignerAgent(session)

        async def fire_cancel() -> None:
            await asyncio.sleep(1.5)
            session.context["cancel_eoh"] = True

        canceller = asyncio.create_task(fire_cancel())
        result = await agent.run(user_text="evolve")
        await canceller

        assert result.get("passed") is False, f"expected cancelled run, got {result}"
        # We should have seen at least one generation event before cancellation
        history = bus.history(session.session_id)
        gen_events = [
            e for e in history
            if e.get("kind") == "chart_data"
            and e.get("chart") == "reflective_search_progress"
            and e.get("data", {}).get("phase") == "generation"
        ]
        assert gen_events, "expected at least one generation event before cancel"
        # Pipeline state should be 'error' since cancellation produced non-zero exit
        assert session.pipeline_state.get("designer") == "error", session.pipeline_state

        print(f"  cancel result: {result}")
        print(f"  generations observed before cancel: {len(gen_events)}")
        print(f"  pipeline_state: {session.pipeline_state}")
        return 0
    finally:
        reflective_search_runner.start_reflective_search = saved_start


def main() -> int:
    print("== test_eoh_designer_e2e ==")
    print("[1/2] happy path (3 generations)")
    rc = asyncio.run(run_happy_path())
    if rc != 0:
        return rc
    print("[2/2] cancel mid-evolution")
    rc = asyncio.run(run_cancel_path())
    if rc != 0:
        return rc
    print("\nAll EOH designer E2E tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
