from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def write_benchmark_artifacts(
    *,
    output_dir: Path,
    embeddings: pd.DataFrame,
    assignments: pd.DataFrame,
    metrics: pd.DataFrame,
    electrochemical_metrics: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings.to_csv(output_dir / "embeddings.csv", index=False)
    assignments.to_csv(output_dir / "assignments.csv", index=False)
    metrics.to_csv(output_dir / "metrics_long.csv", index=False)
    electrochemical_metrics.to_csv(
        output_dir / "electrochemical_metrics.csv",
        index=False,
    )
    with pd.ExcelWriter(output_dir / "summary.xlsx", engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="summary", index=False)
        electrochemical_metrics.to_excel(
            writer,
            sheet_name="electrochemical",
            index=False,
        )
        embeddings.to_excel(writer, sheet_name="embeddings", index=False)
        assignments.to_excel(writer, sheet_name="assignments", index=False)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
