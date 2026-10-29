"""Source-contract tests for the BATTWISE reflective-search entry point."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SEARCH_FILE = (
    ROOT
    / "Evo_Textgrad_battery_eoh_deepseek"
    / "reflective_metric_search.py"
)
MAIN_FILE = ROOT / "Evo_Textgrad_battery_eoh_deepseek" / "main.py"


def _class_names(source: str) -> set[str]:
    tree = ast.parse(source)
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    }


def main() -> int:
    assert SEARCH_FILE.exists(), "primary reflective-search module is missing"
    source = SEARCH_FILE.read_text(encoding="utf-8")
    classes = _class_names(source)

    assert "HierarchicalReflectionMemory" in classes
    assert "ReflectiveMetricSearch" in classes
    assert "EOH_Textgrad" not in classes
    assert "ReflectionLibrary" not in classes

    lowered = source.lower()
    assert "backward propagation" not in lowered
    assert "backward_propagation" not in lowered
    assert "transferable" not in lowered
    assert "cross-task transfer" not in lowered

    # Attribution is deliberately retained: BATTWISE's search implementation
    # builds on the repository's bundled EoH baseline operators.
    assert "baselines.eoh" in source

    main_source = MAIN_FILE.read_text(encoding="utf-8")
    assert "reflective_metric_search" in main_source
    assert "ReflectiveMetricSearch" in main_source
    print("test_reflective_search_source: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
