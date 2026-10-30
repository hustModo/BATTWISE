"""Run synthetic code checks without measured datasets or an LLM service."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Also test optional PyTorch representations (requires requirements-benchmark.txt).")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error("Python 3.10 or newer is required.")
    tests = ["experiments/fair_1d_benchmark/tests"] if args.full else [
        "experiments/fair_1d_benchmark/tests/test_data.py",
        "experiments/fair_1d_benchmark/tests/test_metrics.py",
        "experiments/fair_1d_benchmark/tests/test_battwise_cddi.py",
    ]
    tests.append("app/backend/tests/test_search_layout.py")
    checks = [
        (ROOT, ["scripts/check_release.py"]),
        (ROOT, ["examples/demo_cddi.py"]),
        (ROOT, ["-m", "pytest", *tests, "-q", "-p", "no:cacheprovider"]),
        (ROOT, ["app/backend/tests/test_cddi_compatibility.py"]),
        (ROOT, ["app/backend/tests/test_reflective_search_source.py"]),
        (ROOT, ["app/backend/tests/test_plan_step_edit.py"]),
        (ROOT / "battwise_cddi_search", ["main.py", "--help"]),
        (ROOT, ["experiments/statistics/analyze_fair_benchmark.py", "--help"]),
    ]
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    for cwd, command in checks:
        print("\n> python " + " ".join(command), flush=True)
        try:
            result = subprocess.run([sys.executable, "-B", *command], cwd=cwd, env=env, timeout=120)
        except subprocess.TimeoutExpired:
            print("FAIL: check exceeded 120 seconds.", file=sys.stderr)
            return 1
        if result.returncode:
            return result.returncode
    print(f"\nPASS: {len(checks)} offline checks; no measured-data experiment or LLM search was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
