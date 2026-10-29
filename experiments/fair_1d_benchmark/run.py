from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from typing import Iterable

import numpy as np
import pandas as pd
import sklearn
import yaml

from .clustering import cluster_representation
from .data import load_all200_dataset
from .metrics import compute_partition_metrics
from .registry import load_reducer
from .report import write_benchmark_artifacts


def _resolve_root(config_path: Path, configured: str | None) -> Path:
    if configured is None:
        return config_path.parent.resolve()
    root = Path(configured)
    return root.resolve() if root.is_absolute() else (config_path.parent / root).resolve()


def _resolve_under(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def run_benchmark(
    config_path: str | Path,
    *,
    method_names: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
) -> dict:
    config_path = Path(config_path).resolve()
    config_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    if not isinstance(config, dict):
        raise ValueError("Benchmark config must be a YAML mapping")
    project_root = _resolve_root(config_path, config.get("project_root"))

    dataset_config = dict(config.get("dataset") or {})
    dataset = load_all200_dataset(
        _resolve_under(project_root, dataset_config["data_dir"]),
        expected_cells=int(dataset_config.get("expected_cells", 71)),
        expected_points=int(dataset_config.get("expected_points", 200)),
    )
    clustering_config = dict(config.get("clustering") or {})
    selected_seeds = [
        int(seed)
        for seed in (
            seeds
            if seeds is not None
            else clustering_config.get("seeds", list(range(30)))
        )
    ]
    if not selected_seeds:
        raise ValueError("At least one random seed is required")

    method_specs = dict(config.get("methods") or {})
    selected_methods = (
        list(method_names) if method_names is not None else list(method_specs)
    )
    unknown = [name for name in selected_methods if name not in method_specs]
    if unknown:
        raise KeyError(f"Unknown benchmark methods: {unknown}")

    embedding_rows: list[dict] = []
    assignment_rows: list[dict] = []
    metric_rows: list[dict] = []
    electro_frames: list[pd.DataFrame] = []
    run_diagnostics: list[dict] = []

    for method_key in selected_methods:
        reducer = load_reducer(
            method_specs[method_key],
            project_root=project_root,
        )
        for seed in selected_seeds:
            representation = reducer.fit_transform(dataset, seed=seed)
            use_z = representation.diagnostics.get("use_z", True) is not False
            clustered = cluster_representation(
                dataset,
                representation,
                k=int(clustering_config.get("k", 6)),
                seed=seed,
                pop_size=int(clustering_config.get("pop_size", 200)),
                max_iter=int(clustering_config.get("max_iter", 80)),
                use_z=use_z,
            )
            summary, electro = compute_partition_metrics(dataset, clustered)
            summary["method_key"] = method_key
            summary["representation_elapsed_s"] = float(
                representation.diagnostics.get("elapsed_s", 0.0)
            )
            metric_rows.append(summary)
            electro.insert(0, "method_key", method_key)
            electro_frames.append(electro)

            for cell_id, z1 in zip(
                representation.cell_ids,
                representation.z1,
                strict=True,
            ):
                embedding_rows.append(
                    {
                        "method_key": method_key,
                        "method": representation.method,
                        "seed": seed,
                        "Cell_num": int(cell_id),
                        "z1": float(z1),
                    }
                )
            for row_index, (cell_id, label) in enumerate(
                zip(clustered.cell_ids, clustered.labels, strict=True)
            ):
                assignment = {
                    "method_key": method_key,
                    "method": clustered.method,
                    "seed": seed,
                    "Cell_num": int(cell_id),
                    "Cluster_label": int(label),
                    "OCV_norm": float(clustered.cluster_matrix[row_index, 0]),
                    "IR_norm": float(clustered.cluster_matrix[row_index, 1]),
                    "z1": float(representation.z1[row_index]),
                }
                assignment["z1_norm"] = (
                    float(clustered.cluster_matrix[row_index, 2])
                    if clustered.cluster_matrix.shape[1] == 3
                    else np.nan
                )
                assignment_rows.append(assignment)
            run_diagnostics.append(
                {
                    "method_key": method_key,
                    "method": representation.method,
                    "seed": seed,
                    "diagnostics": representation.diagnostics,
                }
            )

    embeddings = pd.DataFrame(embedding_rows)
    assignments = pd.DataFrame(assignment_rows)
    metrics = pd.DataFrame(metric_rows)
    electrochemical_metrics = pd.concat(electro_frames, ignore_index=True)
    output_dir = _resolve_under(project_root, str(config["output_dir"]))
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "project_root": str(project_root),
        "dataset": {
            "n_cells": int(len(dataset.cell_ids)),
            "curve_shape": list(dataset.curves.shape),
            "n_training_features": int(dataset.train_targets.shape[1]),
            "n_external_electrochemical_features": int(
                dataset.electrochemical.shape[1]
            ),
            "cell_ids": dataset.cell_ids.tolist(),
        },
        "methods": selected_methods,
        "seeds": selected_seeds,
        "clustering": clustering_config,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "runs": run_diagnostics,
    }
    write_benchmark_artifacts(
        output_dir=output_dir,
        embeddings=embeddings,
        assignments=assignments,
        metrics=metrics,
        electrochemical_metrics=electrochemical_metrics,
        manifest=manifest,
    )
    return {
        "methods": selected_methods,
        "seeds": selected_seeds,
        "output_dir": str(output_dir),
        "n_runs": len(metric_rows),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the All_200 fair one-dimensional battery benchmark."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--seeds", nargs="*", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_benchmark(
        args.config,
        method_names=args.methods,
        seeds=args.seeds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
