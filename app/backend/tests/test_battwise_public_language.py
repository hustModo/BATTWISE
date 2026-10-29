"""Prevent deprecated method names from leaking into public-facing text."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

PUBLIC_FILES = (
    ROOT / "app/backend/agents/chat_agent.py",
    ROOT / "app/backend/agents/compute_agent.py",
    ROOT / "app/backend/agents/designer_agent.py",
    ROOT / "app/backend/orchestrator/pipeline.py",
    ROOT / "app/backend/services/conversation_memory.py",
    ROOT / "app/backend/services/llm_router.py",
    ROOT / "app/backend/services/narrator.py",
    ROOT / "app/backend/services/mmd_runner.py",
    ROOT / "app/backend/services/planner.py",
    ROOT / "app/backend/main.py",
    ROOT / "app/frontend/src/components/TopBar.tsx",
    ROOT / "app/frontend/src/components/MainStage.tsx",
    ROOT / "app/frontend/src/components/planStepSchema.ts",
    ROOT / "app/frontend/src/components/charts/ReflectiveSearchLogStream.tsx",
    ROOT / "app/frontend/src/components/charts/ReflectiveSearchProgressChart.tsx",
    ROOT / "app/frontend/src/pages/Landing.tsx",
)

FORBIDDEN = (
    "BatRecycler-MAS",
    "EOH-Textgrad",
    "EOH-TextGrad",
    "Full EOH",
    "Full-EOH",
    "transferable skill",
    "cross-task transfer",
)


def main() -> None:
    failures: list[str] = []
    combined = ""
    for path in PUBLIC_FILES:
        text = path.read_text(encoding="utf-8")
        combined += text
        for token in FORBIDDEN:
            if token.casefold() in text.casefold():
                failures.append(f"{path.relative_to(ROOT)} contains {token!r}")

    assert "BATTWISE" in combined, "public source does not expose the BATTWISE name"
    assert not failures, "\n".join(failures)
    print("test_battwise_public_language: PASS")


if __name__ == "__main__":
    main()
