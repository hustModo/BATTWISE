"""Source-tree invariants that prevent credential and frontend regressions."""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{12,}")


def test_no_embedded_api_keys() -> None:
    checked = [
        ROOT / "battwise_cddi_search" / "utils" / "utils.py",
        ROOT / "docs" / "RELEASE_SCOPE.md",
    ]
    offenders = [
        str(path.relative_to(ROOT))
        for path in checked
        if KEY_PATTERN.search(path.read_text(encoding="utf-8", errors="replace"))
    ]
    assert offenders == [], f"credential-shaped tokens remain in: {offenders}"


def test_frontend_has_no_emitted_source_twins() -> None:
    frontend = ROOT / "app" / "frontend"
    twins = []
    for js_path in (frontend / "src").rglob("*.js"):
        if js_path.with_suffix(".ts").exists() or js_path.with_suffix(".tsx").exists():
            twins.append(str(js_path.relative_to(ROOT)))
    assert twins == [], f"generated JavaScript shadows TypeScript: {twins}"
    assert not (frontend / "vite.config.js").exists()
    assert not (frontend / "tsconfig.tsbuildinfo").exists()


def test_frontend_build_never_emits_into_source() -> None:
    frontend = ROOT / "app" / "frontend"
    tsconfig = json.loads((frontend / "tsconfig.json").read_text(encoding="utf-8"))
    package = json.loads((frontend / "package.json").read_text(encoding="utf-8"))
    assert tsconfig["compilerOptions"].get("noEmit") is True
    assert package["scripts"]["build"] == "tsc --noEmit && vite build"


def main() -> None:
    test_no_embedded_api_keys()
    test_frontend_has_no_emitted_source_twins()
    test_frontend_build_never_emits_into_source()
    print("test_source_hygiene: PASS")


if __name__ == "__main__":
    main()

