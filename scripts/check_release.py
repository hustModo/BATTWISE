"""Check that a review distribution contains code and no private runtime state."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DENIED_SUFFIXES = {".csv", ".xls", ".xlsx", ".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg", ".log", ".pkl", ".pt", ".pth", ".key", ".enc"}
DENIED_PARTS = {"workspace", "node_modules", "dist", "results", "smoke_results", ".cache1"}
PATTERNS = {
    "credential-shaped string": re.compile(r"(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)"),
    "private home path": re.compile(r"(?:/home/[A-Za-z0-9_.-]+/|[A-Za-z]:[\\/](?:Users|users)[\\/])"),
}


def main() -> int:
    problems = []
    files = []
    manifest = dict(line.split("  ", 1)[::-1] for line in
                    (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines())
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in {".git", "__pycache__", ".pytest_cache", ".venv", ".venv-benchmark"} for part in relative.parts):
            continue
        if not path.is_file():
            continue
        files.append(relative.as_posix())
        source_data = (relative.parts[0] == "figures" or
                       relative.parts[:2] == ("battery_compute", "AII_200_new"))
        approved_data = (source_data and path.suffix.lower() in {".csv", ".xls", ".xlsx"}
                         and manifest.get(relative.as_posix()) == hashlib.sha256(path.read_bytes()).hexdigest())
        if (path.suffix.lower() in DENIED_SUFFIXES and not approved_data) or any(part in DENIED_PARTS for part in relative.parts):
            problems.append((relative.as_posix(), "excluded artifact type or directory"))
        if path.name.startswith(".env") and path.name != ".env.example":
            problems.append((relative.as_posix(), "runtime environment file"))
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".py":
            try:
                ast.parse(text)
            except SyntaxError as exc:
                problems.append((relative.as_posix(), f"syntax error at line {exc.lineno}"))
        # Pattern definitions themselves are not secret-bearing source values.
        if path.resolve() != Path(__file__).resolve():
            for label, pattern in PATTERNS.items():
                if pattern.search(text):
                    problems.append((relative.as_posix(), label))
    program = ROOT / "programs/cddi/calculate_mmd_v2.py"
    source = program.read_text(encoding="utf-8").strip()
    expected = "f18815885200043f9f14371d169b6bc600757d8b19e70f262fd175d2d6bbfda7"
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != expected:
        problems.append(("programs/cddi/calculate_mmd_v2.py", "frozen source hash mismatch"))
    if json.loads((ROOT / "programs/cddi/program.json").read_text(encoding="utf-8"))["code"] != source:
        problems.append(("programs/cddi/program.json", "program source differs"))
    print(json.dumps({"checked_files": len(files), "problems": problems}, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
