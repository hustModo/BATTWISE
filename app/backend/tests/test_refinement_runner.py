"""Offline tests for the refinement-specific service additions:
    - mmd_runner.compute_mmd_for_folder(cells_filter=...)
    - clustering_runner.cluster_subset(...)
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.backend.services import clustering_runner, mmd_runner  # noqa: E402
from app.backend.services.clustering_runner import cluster_subset  # noqa: E402
from app.backend.settings import BUILTIN_DATASETS, EVO_GPT_PATH  # noqa: E402

GOOD_CODE = '''
import numpy as np

def calculate_mmd_v2(xs: np.ndarray, xt: np.ndarray, fix_sigma=0, kernel_mul=2.0, kernel_num=5):
    xs = np.asarray(xs, dtype=float)
    xt = np.asarray(xt, dtype=float)
    if xs.size == 0 or xt.size == 0:
        return 0.0
    mu_s, mu_t = xs.mean(axis=0), xt.mean(axis=0)
    return float(np.sum((mu_s - mu_t) ** 2))
'''


def test_cells_filter_limits_compute() -> None:
    ds = BUILTIN_DATASETS["builtin-first-200s"]
    folder = Path(ds["folder"])
    target_cells = [10, 11, 12, 13]
    mmd_runner.inject_mmd_code(GOOD_CODE)
    fn = mmd_runner.load_injected_function()
    with tempfile.TemporaryDirectory() as td:
        out_csv = Path(td) / "subset.csv"
        result = mmd_runner.run_full_pipeline(
            data_folder=folder,
            model_filename=ds["model_file"],
            char_filename=None,
            out_csv=out_csv,
            mmd_fn=fn,
            cells_filter=target_cells,
        )
        cells = sorted(r["Cell_num"] for r in result["results"])
        assert set(cells).issubset(set(target_cells)), cells
        # We expect exactly the cells we asked for (assuming they exist in builtin)
        assert set(cells) == set(target_cells), f"expected {target_cells}, got {cells}"
    print("[1] cells_filter restricts compute_mmd_for_folder OK")


def test_cluster_subset_smaller_k() -> None:
    """End-to-end small smoke test: compute MMD on a subset, then sub-cluster."""
    ds = BUILTIN_DATASETS["builtin-first-200s"]
    folder = Path(ds["folder"])
    indicator = Path(ds["indicator_path"])
    target_cells = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    mmd_runner.inject_mmd_code(GOOD_CODE)
    fn = mmd_runner.load_injected_function()

    with tempfile.TemporaryDirectory() as td:
        out_csv = Path(td) / "subset_mmd.csv"
        out_label = Path(td) / "subset_label.xlsx"
        mmd = mmd_runner.run_full_pipeline(
            data_folder=folder,
            model_filename=ds["model_file"],
            char_filename=None,
            out_csv=out_csv,
            mmd_fn=fn,
            cells_filter=target_cells,
        )
        assert mmd["results"], "no MMD results computed for subset"
        sub = cluster_subset(
            mmd_csv=out_csv,
            indicator_xlsx=indicator,
            cell_nums=target_cells,
            out_xlsx=out_label,
            pop_size=80,
            max_iter=20,
            min_k=2,
            max_k=3,
        )
        assert 2 <= sub["k"] <= 3, sub["k"]
        assert sub["n_samples"] >= 4
        assert len(sub["labels"]) == sub["n_samples"]
        assert out_label.exists()
        df = pd.read_excel(out_label, engine="calamine")
        assert "Cluster_label" in df.columns
        assert "Cell_num" in df.columns
        # Every Cell_num in output should be in target_cells
        out_cells = set(int(x) for x in df["Cell_num"].dropna().tolist())
        assert out_cells.issubset(set(target_cells)), out_cells - set(target_cells)
        # summary structure
        assert "scatter" in sub["summary"]
        assert "boxplot" in sub["summary"]
        assert "fingerprint" in sub["summary"]
        assert "per_cluster" in sub["summary"]
        print(
            f"[2] cluster_subset: n={sub['n_samples']}, k={sub['k']}, "
            f"dbi={sub['dbi']:.3f}, sizes={[c['size'] for c in sub['summary']['per_cluster']]}"
        )


if __name__ == "__main__":
    test_cells_filter_limits_compute()
    test_cluster_subset_smaller_k()
    print("All refinement-runner tests passed.")
