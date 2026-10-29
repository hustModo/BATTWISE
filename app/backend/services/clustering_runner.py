"""Function-ised version of First_200_select/cluster.py.

Removes matplotlib/Tk interactions and pure-text I/O. Everything returns
JSON-serialisable dicts. Public API:

    merge_mmd_into_indicator(mmd_csv, indicator_xlsx, out_xlsx=None) -> pd.DataFrame
    run_adaptive_kmeans(data_norm, cap_norm, ...)                    -> ClusterRunResult
    summarize_clusters(merged_df, labels, ...)                       -> dict (JSON-safe)
    cluster_dataset(...)                                             -> dict bundle (used by the agent)

Behaviour is preserved relative to ``First_200_select/cluster.py`` but
parameters default to a faster "live" setting (pop=200, iter=80). Original
slow settings can be reproduced via the kwargs.
"""
from __future__ import annotations

import copy
import logging
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler

from app.backend.services.mmd_designer import _load_data

os.environ.setdefault("OMP_NUM_THREADS", "1")  # avoid KMeans memory-leak warning


logger = logging.getLogger(__name__)

_TRAIN_NON_FEATURE_COLS = {
    "Cell_num",
    "Filename",
    "Cluster_label",
    "Merged_label",
    "Merged_display",
    "Final_label",
    "Final_display",
    "Label_source",
}

_DEFAULT_FEATURE_WEIGHTS: Dict[str, float] = {
    "ocv": 1.0,
    "ir": 1.0,
    "ddia": 1.0,
}


# ==========================================
# 1. Data structures
# ==========================================

class Individual:
    """Cluster individual — first row of `matrix` holds K, rows 1..K are centroids."""

    __slots__ = ("matrix", "fitness", "dbi", "compactness", "cluster_result", "cap_std")

    def __init__(self, matrix: Optional[np.ndarray] = None) -> None:
        self.matrix = matrix
        self.fitness: float = float("inf")
        self.dbi: float = float("inf")
        self.compactness: float = float("inf")
        self.cap_std: float = 0.0
        self.cluster_result: Optional[np.ndarray] = None


@dataclass
class ClusterRunResult:
    labels: List[int]
    k: int
    fitness: float
    dbi: float
    compactness: float
    cap_std: float
    history: List[Dict[str, float]] = field(default_factory=list)
    elapsed_s: float = 0.0


# ==========================================
# 2. helpers
# ==========================================

def _get_cen_num(mat: np.ndarray) -> int:
    return int(mat[0, 0])


def _get_centroids(mat: np.ndarray) -> np.ndarray:
    n = _get_cen_num(mat)
    return mat[1:n + 1, :]


def _create_indi_matrix(centroids: np.ndarray) -> np.ndarray:
    k, dim = centroids.shape
    mat = np.zeros((k + 1, dim))
    mat[0, 0] = k
    mat[1:, :] = centroids
    return mat


def _kmeans_plus_plus_init(data: np.ndarray, k: int, rng: random.Random) -> np.ndarray:
    n_samples, n_features = data.shape
    if k > n_samples:
        k = n_samples
    centers = np.empty((k, n_features))
    centers[0] = data[rng.randrange(n_samples)]
    for i in range(1, k):
        dists_sq = np.min(
            [np.sum((data - center) ** 2, axis=1) for center in centers[:i]], axis=0
        )
        total = dists_sq.sum()
        if total <= 0:
            centers[i] = data[rng.randrange(n_samples)]
            continue
        probs = dists_sq / total
        r = rng.random()
        cum = np.cumsum(probs)
        idx = int(np.searchsorted(cum, r))
        idx = min(idx, n_samples - 1)
        centers[i] = data[idx]
    return centers


# ==========================================
# 3. fitness / mutation / crossover (preserved logic)
# ==========================================

def _calc_fitness(
    indi: Individual,
    data: np.ndarray,
    cap: np.ndarray,
    use_capacity: bool = False,
    feedback_metric: str = "combined",
) -> Tuple[np.ndarray, float, float, float, float]:
    """GA fitness = weighted sum of DBI + intra-distance + capacity-std.

    `feedback_metric` lets the user steer which signal dominates the
    selection pressure. Set via chat ("use variance feedback") or kept
    at the default ``combined`` (current production weights).

    Weights (per metric):
        combined:  2*DBI + 1*intra + 5*cap_std        (default)
        dbi:       4*DBI + 1*intra + 0*cap_std        (geometric quality)
        variance:  1*DBI + 0.5*intra + 12*cap_std     (in-cluster capacity homogeneity)
    """
    mat = indi.matrix
    cen_num = _get_cen_num(mat)
    centroids = _get_centroids(mat)
    m, n = data.shape

    diff = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    min_dists = np.min(dists, axis=1)
    labels = np.argmin(dists, axis=1)

    unique_labels, counts = np.unique(labels, return_counts=True)
    penalty = 0
    if len(unique_labels) < cen_num:
        penalty += 10000
    if np.any(counts < 4):
        penalty += 5000

    if len(unique_labels) > 1:
        score_dbi = float(davies_bouldin_score(data, labels))
    else:
        score_dbi = 10.0
    avg_intra_dist = float(np.mean(min_dists))

    valid_stds = []
    for i in range(cen_num):
        indices = np.where(labels == i)[0]
        if len(indices) > 1:
            valid_stds.append(float(np.std(cap[indices], ddof=1)))
    cap_std = float(np.mean(valid_stds)) if valid_stds else 0.0

    k_penalty = 0.05 * cen_num
    metric = (feedback_metric or "combined").lower()
    if metric == "dbi":
        w_dbi, w_intra, w_cap = 4.0, 1.0, 0.0
    elif metric == "variance":
        # Capacity is forbidden as a clustering signal. In variance mode,
        # push on within-cluster feature compactness instead.
        w_dbi, w_intra, w_cap = 1.0, 4.0, 0.0
    else:  # combined (default — current production weights)
        w_dbi, w_intra, w_cap = 2.0, 1.5, 0.0
    if not use_capacity:
        w_cap = 0.0
    fitness = (w_dbi * score_dbi) + (w_intra * avg_intra_dist) + (w_cap * cap_std) + k_penalty + penalty

    re = np.zeros((m, n + 2))
    re[:, 0:n] = data
    re[:, n] = min_dists
    re[:, n + 1] = labels
    return re, float(fitness), score_dbi, avg_intra_dist, cap_std


def _indi_update_fast(indi: Individual, data: np.ndarray) -> Individual:
    mat = indi.matrix
    cen_num = _get_cen_num(mat)
    current_centers = _get_centroids(mat)
    if cen_num < 2:
        return indi
    try:
        km = KMeans(n_clusters=cen_num, init=current_centers, n_init=1, max_iter=10)
        km.fit(data)
        return Individual(_create_indi_matrix(km.cluster_centers_))
    except Exception:
        return indi


def _mutation(
    indi: Individual,
    data: np.ndarray,
    rng: random.Random,
    min_k: int = 2,
    max_k: int = 20,
) -> Individual:
    mat = indi.matrix.copy()
    cen_num = _get_cen_num(mat)
    centroids = _get_centroids(mat)

    # Respect hard K bounds. This matters when the user pins an exact
    # cluster count (min_k == max_k): the old logic forced a split at
    # cen_num <= min_k, so a fixed k=4 run could mutate into k=5.
    can_split = cen_num < max_k
    can_merge = cen_num > min_k
    if not can_split and not can_merge:
        return Individual(_create_indi_matrix(centroids.copy()))
    if can_split and not can_merge:
        action = "split"
    elif can_merge and not can_split:
        action = "merge"
    else:
        action = "split" if rng.random() < 0.5 else "merge"

    new_centroids = centroids.copy()
    if action == "split":
        diff = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        labels = np.argmin(np.linalg.norm(diff, axis=2), axis=1)
        max_var = -1.0
        target_idx = -1
        for i in range(cen_num):
            indices = np.where(labels == i)[0]
            if len(indices) > 2:
                var = float(np.mean(np.linalg.norm(data[indices] - centroids[i], axis=1)))
                if var > max_var:
                    max_var = var
                    target_idx = i
        if target_idx != -1:
            target_cen = centroids[target_idx]
            shift = 0.05 * np.std(data, axis=0)
            new_c1 = target_cen + shift
            new_c2 = target_cen - shift
            new_centroids = np.delete(new_centroids, target_idx, axis=0)
            new_centroids = np.vstack([new_centroids, new_c1, new_c2])
    else:  # merge
        if cen_num > min_k:
            min_dist = float("inf")
            pair = (-1, -1)
            for i in range(cen_num):
                for j in range(i + 1, cen_num):
                    d = float(np.linalg.norm(centroids[i] - centroids[j]))
                    if d < min_dist:
                        min_dist = d
                        pair = (i, j)
            if pair[0] != -1:
                idx1, idx2 = pair
                merged = (centroids[idx1] + centroids[idx2]) / 2.0
                new_centroids = np.delete(new_centroids, idx2, axis=0)
                new_centroids = np.delete(new_centroids, idx1, axis=0)
                new_centroids = np.vstack([new_centroids, merged])
    return Individual(_create_indi_matrix(new_centroids))


def _crossover(
    indi_list: List[Individual], cro_num: int, rng: random.Random
) -> List[Individual]:
    out: List[Individual] = []
    n = len(indi_list)
    if n < 2:
        return out
    indices = list(range(n))
    rng.shuffle(indices)
    iter_num = min(cro_num // 2, n // 2)
    for i in range(iter_num):
        ind1 = indi_list[indices[2 * i]]
        ind2 = indi_list[indices[2 * i + 1]]
        cens1 = _get_centroids(ind1.matrix).copy()
        cens2 = _get_centroids(ind2.matrix).copy()
        k1, k2 = len(cens1), len(cens2)
        min_k = min(k1, k2)
        if min_k > 1:
            exc_num = rng.randint(1, max(1, min_k // 2))
            idx1 = np.array(rng.sample(range(k1), exc_num))
            idx2 = np.array(rng.sample(range(k2), exc_num))
            temp = cens1[idx1].copy()
            cens1[idx1] = cens2[idx2]
            cens2[idx2] = temp
        out.append(Individual(_create_indi_matrix(cens1)))
        out.append(Individual(_create_indi_matrix(cens2)))
    return out


# ==========================================
# 4. main adaptive K driver
# ==========================================

def run_adaptive_kmeans(
    data_norm: np.ndarray,
    cap_norm: np.ndarray,
    *,
    pop_size: int = 200,
    max_iter: int = 80,
    min_k: int = 2,
    max_k: Optional[int] = None,
    copy_rate: float = 0.05,
    upd_rate: float = 0.55,
    cros_rate: float = 0.2,
    use_capacity: bool = False,
    seed: Optional[int] = 42,
    on_iter: Optional[Callable[[int, Individual], None]] = None,
    feedback_metric: str = "combined",
) -> ClusterRunResult:
    """Adaptive GA-KMeans matching the algorithm in First_200_select/cluster.py.

    `feedback_metric` selects which signal dominates GA fitness:
        - "combined" (default): current production weights
        - "dbi": prefer geometrically-tight clusters
        - "variance": prefer in-cluster capacity homogeneity
    """
    import time

    rng = random.Random(seed)
    np.random.seed(seed)

    n_samples = data_norm.shape[0]
    if max_k is None:
        max_k = max(min_k, min(3, n_samples // 2))
    if max_k < min_k:
        max_k = min_k

    pop: List[Individual] = []
    for _ in range(pop_size):
        k_init = rng.randint(min_k, max_k)
        centroids = _kmeans_plus_plus_init(data_norm, k_init, rng)
        pop.append(Individual(_create_indi_matrix(centroids)))

    best_fitness = float("inf")
    best: Optional[Individual] = None
    history: List[Dict[str, float]] = []
    t0 = time.time()

    for it in range(1, max_iter + 1):
        for ind in pop:
            res, fit, dbi, comp, cstd = _calc_fitness(
                ind, data_norm, cap_norm, use_capacity, feedback_metric=feedback_metric,
            )
            ind.cluster_result = res
            ind.fitness = fit
            ind.dbi = dbi
            ind.compactness = comp
            ind.cap_std = cstd
        pop.sort(key=lambda x: x.fitness)

        if pop[0].fitness < best_fitness:
            best_fitness = pop[0].fitness
            best = copy.deepcopy(pop[0])

        history.append({
            "iter": it,
            "k": _get_cen_num(pop[0].matrix),
            "fitness": pop[0].fitness,
            "dbi": pop[0].dbi,
            "cap_std": pop[0].cap_std,
            "compactness": pop[0].compactness,
        })
        if on_iter and best is not None:
            on_iter(it, best)

        new_pop: List[Individual] = []
        copy_n = math.ceil(pop_size * copy_rate)
        for i in range(copy_n):
            new_pop.append(copy.deepcopy(pop[i]))
        upd_n = math.ceil(pop_size * upd_rate)
        for cand in pop[copy_n:copy_n + upd_n]:
            new_pop.append(_indi_update_fast(cand, data_norm))
        cro_n = math.ceil(pop_size * cros_rate)
        if cro_n % 2 != 0:
            cro_n += 1
        pool = pop[copy_n + upd_n:]
        if len(pool) >= 2:
            while len(pool) < cro_n:
                pool.extend(pool)
            new_pop.extend(_crossover(pool[:cro_n], cro_n, rng))
        # mutation fills the rest
        while len(new_pop) < pop_size:
            parent = rng.choice(pop)
            new_pop.append(_mutation(parent, data_norm, rng, min_k, max_k))
        pop = new_pop[:pop_size]

    assert best is not None
    final_mat = best.matrix
    final_centers = _get_centroids(final_mat)
    diff = data_norm[:, np.newaxis, :] - final_centers[np.newaxis, :, :]
    labels = np.argmin(np.linalg.norm(diff, axis=2), axis=1)

    return ClusterRunResult(
        labels=labels.tolist(),
        k=_get_cen_num(final_mat),
        fitness=float(best.fitness),
        dbi=float(best.dbi),
        compactness=float(best.compactness),
        cap_std=float(best.cap_std),
        history=history,
        elapsed_s=time.time() - t0,
    )


# ==========================================
# 5. data merge
# ==========================================

# Derived clustering input convention:
#   Cell_num | OCV | IR | CDDI
# Internally CDDI is normalised to MMD_Loss for backwards-compatible summary code.
_INDICATOR_NON_FEATURE_COLS = {
    "Cell_num", "Cluster_label", "Merged_label", "Merged_display",
    "Final_label", "Final_display", "Label_source", "Parent_cluster",
    "Sub_cluster", "Sample_count", "Cell_ID_Match",
}

# Electrochemical / equivalent-circuit-model columns. These are NOT
# measurable in the deployed application (the user's complaint: 实际应用
# 中电化学特征没有测出). Excluded from the fingerprint radar so the
# right-bottom panel only shows curve-derived features (gradients, ICA,
# CV, Q…). They're STILL included in the dispersion table so the analyst
# can still inspect their per-cluster spread offline.
_ELECTROCHEMICAL_COLS = {
    "OCV", "IR",
    # Equivalent-circuit-model EIS fit parameters
    "Rs", "CPE1-T", "Rp", "CPE3-T", "R4", "CPE2-T", "R3", "Wo-R", "Wo-T",
}


def format_curve_dispersion_markdown(
    dispersion: Dict[str, Any],
    *,
    top_n: int = 5,
) -> Optional[str]:
    """Build a markdown block summarising curve-feature dispersion.

    The cluster chat message already calls out per-cluster *capacity*
    spread (μ / σ / CV) and the F-style separability. This helper
    focuses on the *curve features* (V_init_mean, V_drop, …) and
    surfaces:

    * A short "noisiest features" line (top-N by mean within-cluster σ).
    * A markdown table of per-cluster σ for those features, plus a
      "Mean σ" column so the user can eyeball variability at a glance.

    Returns ``None`` if the dispersion payload has no usable curve
    features (e.g. clustering ran on Capacity alone), so the caller can
    skip the block instead of posting an empty table.
    """
    features = dispersion.get("features") or []
    # Exclude Capacity — it gets its own dedicated block already.
    curve_features = [f for f in features if f != "Capacity"]
    if not curve_features:
        return None

    feature_mean_std = dispersion.get("feature_mean_std") or {}
    # Rank by mean within-cluster σ (largest = noisiest). Skip features
    # with no stats so a dataset with missing columns doesn't fake-rank.
    ranked = sorted(
        (
            (f, feature_mean_std.get(f))
            for f in curve_features
            if isinstance(feature_mean_std.get(f), (int, float))
        ),
        key=lambda kv: kv[1] or 0.0,
        reverse=True,
    )
    if not ranked:
        return None
    top = ranked[: max(1, int(top_n))]

    per_cluster = dispersion.get("per_cluster") or []
    if not per_cluster:
        return None

    # Headline line — name the noisiest features so the user sees the
    # message even on small screens that fold the table away.
    noisy_bits = [f"`{name}` (σ̄={val:.3g})" for name, val in top if val is not None]
    headline = "**Curve-feature dispersion** (noisiest first): " + ", ".join(noisy_bits) + "."

    # Markdown table: rows = top features, cols = clusters + Mean σ.
    cluster_ids = [r.get("cluster") for r in per_cluster if r.get("cluster") is not None]
    header = ["Feature"] + [f"Cluster {c} σ" for c in cluster_ids] + ["Mean σ"]
    align = ["---"] * len(header)
    rows: List[str] = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    for fname, mean_std in top:
        cells: List[str] = [f"`{fname}`"]
        for r in per_cluster:
            stat = (r.get(fname) or {}) if isinstance(r, dict) else {}
            std = stat.get("std") if isinstance(stat, dict) else None
            cells.append(f"{float(std):.3g}" if isinstance(std, (int, float)) else "—")
        cells.append(f"{float(mean_std):.3g}" if isinstance(mean_std, (int, float)) else "—")
        rows.append("| " + " | ".join(cells) + " |")
    table = "\n".join(rows)

    return f"{headline}\n\n{table}".replace("MMD_Loss", "CDDI").replace("DDIA", "CDDI")


def write_dispersion_xlsx(
    dispersion: Dict[str, Any],
    out_path: Path,
) -> Path:
    """Write the per-cluster × per-feature dispersion stats as a wide XLSX.

    Layout:
        cluster | size | <feature>_mean | <feature>_std | <feature>_cv | …
    One sheet "by_cluster", plus a "by_feature" sheet listing the mean-of-std
    per feature.
    """
    features = dispersion.get("features") or []
    rows = dispersion.get("per_cluster") or []
    flat: List[Dict[str, Any]] = []
    for r in rows:
        row = {"cluster": r.get("cluster"), "size": r.get("size")}
        for f in features:
            stats = r.get(f) or {}
            label = "CDDI" if f in {"MMD_Loss", "DDIA", "DDI"} else f
            row[f"{label}_mean"] = stats.get("mean")
            row[f"{label}_std"] = stats.get("std")
            row[f"{label}_cv"] = stats.get("cv")
        flat.append(row)
    df_by_cluster = pd.DataFrame(flat)

    feat_stds = dispersion.get("feature_mean_std") or {}
    df_by_feature = pd.DataFrame(
        [{"feature": "CDDI" if k in {"MMD_Loss", "DDIA", "DDI"} else k,
          "mean_of_std": v} for k, v in feat_stds.items()]
    )

    cap_aggr = dispersion.get("capacity_aggregate") or {}
    df_capacity = pd.DataFrame([{
        "metric": k,
        "value": v,
    } for k, v in cap_aggr.items()])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        df_by_cluster.to_excel(xw, sheet_name="by_cluster", index=False)
        df_by_feature.to_excel(xw, sheet_name="by_feature", index=False)
        if not df_capacity.empty:
            df_capacity.to_excel(xw, sheet_name="capacity_aggregate", index=False)
    return out_path


def _normalise_indicator_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    aliases = ("CDDI", "MMD_Loss", "DDIA", "CDDI_Loss", "DDIA_Loss", "DDI")
    column = next((name for name in aliases if name in out.columns), None)
    if column is not None:
        out = out.drop(columns=[name for name in aliases if name != column], errors="ignore")
        out = out.rename(columns={column: "MMD_Loss"})
    return out


def load_cluster_input(input_xlsx: Path) -> pd.DataFrame:
    """Load a derived clustering input workbook.

    Expected user-facing columns are Cell_num, OCV, IR, CDDI. Internally we
    normalise CDDI to MMD_Loss so older summary code keeps working.
    """
    df = _load_data(Path(input_xlsx))
    if df is None:
        raise FileNotFoundError(f"Could not load clustering input: {input_xlsx}")
    df = _normalise_indicator_columns(df)
    required = {"Cell_num", "OCV", "IR", "MMD_Loss"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{Path(input_xlsx).name} missing required columns: {sorted(missing)}")
    df["Cell_num"] = pd.to_numeric(df["Cell_num"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Cell_num"]).reset_index(drop=True)
    df["Cell_num"] = df["Cell_num"].astype(int)
    return df


def _attach_posthoc_capacity(df: pd.DataFrame, posthoc_xlsx: Optional[Path]) -> pd.DataFrame:
    """Attach Capacity after clustering input is built.

    Capacity is post-hoc only: never part of feature construction or GA
    fitness, but useful for boxplots, decisions, and final reports.
    """
    if posthoc_xlsx is None:
        return df
    meta = _load_data(Path(posthoc_xlsx))
    if meta is None:
        return df
    meta = meta.copy()
    meta.columns = [str(c).strip() for c in meta.columns]
    if "Cell_num" not in meta.columns or "Capacity" not in meta.columns:
        return df
    cap = meta[["Cell_num", "Capacity"]].copy()
    cap["Cell_num"] = pd.to_numeric(cap["Cell_num"], errors="coerce").astype("Int64")
    cap = cap.dropna(subset=["Cell_num"]).copy()
    cap["Cell_num"] = cap["Cell_num"].astype(int)
    cap["Capacity"] = pd.to_numeric(cap["Capacity"], errors="coerce")
    out = df.drop(columns=["Capacity"], errors="ignore").copy()
    return pd.merge(out, cap, on="Cell_num", how="left")


def merge_mmd_into_indicator(
    mmd_csv: Path,
    indicator_xlsx: Path,
    out_xlsx: Optional[Path] = None,
) -> pd.DataFrame:
    """Replace the MMD_Loss column in indicator_xlsx with values from mmd_csv (by Cell_num)."""
    df_mmd = _normalise_indicator_columns(pd.read_csv(mmd_csv))
    if "Cell_num" not in df_mmd.columns or "MMD_Loss" not in df_mmd.columns:
        raise KeyError("mmd_csv must contain Cell_num and MMD_Loss columns")
    df_mmd = df_mmd[["Cell_num", "MMD_Loss"]].copy()
    df_mmd["Cell_num"] = df_mmd["Cell_num"].astype(int)

    df_ind = _load_data(Path(indicator_xlsx))
    if df_ind is None:
        raise FileNotFoundError(f"Could not load indicator file: {indicator_xlsx}")
    df_ind = _normalise_indicator_columns(df_ind)
    df_ind["Cell_num"] = pd.to_numeric(df_ind["Cell_num"], errors="coerce").astype("Int64")
    df_ind = df_ind.dropna(subset=["Cell_num"]).reset_index(drop=True)
    df_ind["Cell_num"] = df_ind["Cell_num"].astype(int)

    df_ind = df_ind.drop(columns=["MMD_Loss"], errors="ignore")
    merged = pd.merge(df_ind, df_mmd, on="Cell_num", how="inner")

    cols = list(merged.columns)
    if "MMD_Loss" in cols and "Cluster_label" in cols:
        cols.remove("MMD_Loss")
        idx = cols.index("Cluster_label") + 1
        cols = cols[:idx] + ["MMD_Loss"] + cols[idx:]
        merged = merged[cols]

    if out_xlsx is not None:
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)
        merged.rename(columns={"MMD_Loss": "CDDI"}).to_excel(out_xlsx, index=False)
    return merged


# ==========================================
# 6. summarisation (per-cluster JSON-safe payload)
# ==========================================

def _safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def normalize_feature_weights(raw: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Return positive OCV/IR/CDDI weights, defaulting to equal 1:1:1.

    Weights are applied *after* MinMax scaling, so they change relative
    importance without reintroducing unit-scale bias.
    """
    out = dict(_DEFAULT_FEATURE_WEIGHTS)
    if isinstance(raw, dict):
        aliases = {
            "ocv": "ocv",
            "ocv_weight": "ocv",
            "ir": "ir",
            "ir_weight": "ir",
            "cddi": "ddia",
            "cddi_weight": "ddia",
            "ddi": "ddia",
            "ddia": "ddia",
            "ddi_weight": "ddia",
            "ddia_weight": "ddia",
            "mmd": "ddia",
            "mmd_loss": "ddia",
        }
        for k, v in raw.items():
            key = aliases.get(str(k).strip().lower())
            if not key:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv) and fv > 0:
                out[key] = max(0.05, min(20.0, fv))
    return out


def _feature_group(col: str) -> str:
    low = col.lower()
    if low == "ocv":
        return "ocv"
    if low == "ir":
        return "ir"
    if low in {"mmd_loss", "cddi", "cddi_loss", "ddia", "ddi", "ddi_loss", "ddia_loss"}:
        return "ddia"
    return "curve"


def _numeric_feature_columns(df: pd.DataFrame, *, mode: str) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c in _TRAIN_NON_FEATURE_COLS:
            continue
        if c == "Capacity":
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if mode == "main":
            if c in {"OCV", "IR", "MMD_Loss"}:
                cols.append(c)
        else:
            # All-200 refinement uses curve-derived feature tables. Exclude
            # electrochemical-only fields if they happen to appear, but keep
            # Urest/tCC/tCV/dU/dt... and CDDI loss.
            if c in _ELECTROCHEMICAL_COLS and c != "MMD_Loss":
                continue
            cols.append(c)
    if mode == "main":
        missing = [c for c in ("OCV", "IR", "MMD_Loss") if c not in cols]
        if missing:
            raise KeyError(f"Main clustering requires columns: {missing}")
    else:
        if "MMD_Loss" in df.columns and "MMD_Loss" not in cols:
            cols.append("MMD_Loss")
        if len(cols) < 2:
            raise KeyError("Curve-feature clustering needs at least two numeric features")
    return cols


def _weighted_minmax(
    frame: pd.DataFrame,
    *,
    feature_weights: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    weights = normalize_feature_weights(feature_weights)
    data = frame.astype(float).values
    data_norm = MinMaxScaler().fit_transform(data)
    mult = []
    for c in frame.columns:
        group = _feature_group(str(c))
        mult.append(weights.get(group, 1.0))
    return data_norm * np.asarray(mult, dtype=float), weights


def _build_training_inputs(
    merged: pd.DataFrame,
    *,
    mode: str,
    feature_weights: Optional[Dict[str, Any]] = None,
    require_capacity: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], Dict[str, float], bool]:
    feature_cols = _numeric_feature_columns(merged, mode=mode)
    train = merged[feature_cols].apply(pd.to_numeric, errors="coerce")
    has_capacity = "Capacity" in merged.columns
    cap = pd.DataFrame({"Capacity": np.zeros(len(merged), dtype=float)})

    valid = ~train.isna().any(axis=1)
    if require_capacity:
        # Capacity is no longer allowed to drive clustering. Keep the argument
        # for backwards compatibility, but do not add capacity filtering or
        # fitness pressure.
        logger.info("Ignoring require_capacity=True; Capacity is post-hoc only.")

    if valid.sum() < 4:
        raise RuntimeError(f"Not enough valid rows for clustering: {valid.sum()}")

    filtered = merged.loc[valid].reset_index(drop=True)
    train = train.loc[valid].reset_index(drop=True)
    cap = cap.loc[valid].reset_index(drop=True)
    data_norm, weights = _weighted_minmax(train, feature_weights=feature_weights)
    cap_norm = np.zeros(len(filtered), dtype=float)
    return filtered, data_norm, cap_norm, feature_cols, weights, bool(has_capacity)


def _choose_scatter_axes(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[str, str]:
    if "OCV" in df.columns and "IR" in df.columns:
        return "OCV", "IR"
    if "IR" in df.columns and "Capacity" in df.columns:
        return "IR", "Capacity"
    numeric = [
        c for c in feature_cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(numeric) >= 2:
        ranked = sorted(
            numeric,
            key=lambda c: float(pd.to_numeric(df[c], errors="coerce").std(skipna=True) or 0.0),
            reverse=True,
        )
        return ranked[0], ranked[1]
    if len(numeric) == 1:
        return numeric[0], numeric[0]
    return "x", "y"


def _compute_dispersion(
    df: pd.DataFrame,
    feature_cols: List[str],
    cluster_ids: List[int],
) -> Dict[str, Any]:
    """Per-cluster × per-feature spread stats + global aggregates.

    Returns a JSON-safe dict so the chat panel and the downloadable XLSX
    can both consume it. The "capacity_summary" line is what we drop into
    the chat as the headline; the full table is what the user clicks
    Download on.

    Stats per (cluster, feature):
        mean, std (sample), cv = std/|mean| (0 if mean==0)

    Capacity-specific aggregates:
        tightest_cluster: id with smallest capacity-std
        loosest_cluster:  id with largest capacity-std
        f_ratio:          variance(group means) / mean(within-group var)
                          — Fisher-style separability index; >1 means the
                          groups are more spread apart than within.
    """
    out_rows: List[Dict[str, Any]] = []
    cap_stats_per_cluster: Dict[int, Dict[str, Optional[float]]] = {}

    # Capacity: always reported as a top-level feature even though it's
    # excluded from feature_cols (it's the prediction target).
    all_features = ["Capacity"] + list(feature_cols)

    cap_within_vars: List[float] = []
    cap_group_means: List[float] = []
    overall_cap_mean: Optional[float] = None
    cap_all = df["Capacity"].dropna().astype(float) if "Capacity" in df.columns else pd.Series(dtype=float)
    if len(cap_all):
        overall_cap_mean = float(cap_all.mean())

    for c in cluster_ids:
        mask = df["Cluster_label"] == c
        sub = df[mask]
        size = int(len(sub))
        row: Dict[str, Any] = {"cluster": c, "size": size}
        for f in all_features:
            if f not in sub.columns:
                continue
            vals = pd.to_numeric(sub[f], errors="coerce").dropna()
            mean = float(vals.mean()) if len(vals) else None
            std = float(vals.std(ddof=1)) if len(vals) > 1 else None
            if mean is not None and std is not None and abs(mean) > 1e-9:
                cv = std / abs(mean)
            else:
                cv = None
            row[f] = {
                "mean": _safe_float(mean),
                "std": _safe_float(std),
                "cv": _safe_float(cv),
            }
            if f == "Capacity":
                cap_stats_per_cluster[c] = row[f]
                if std is not None:
                    cap_within_vars.append(std * std)
                if mean is not None:
                    cap_group_means.append(mean)
        out_rows.append(row)

    # Aggregates over capacity
    cap_aggr: Dict[str, Any] = {}
    if cap_stats_per_cluster:
        tightest = min(
            (c for c, s in cap_stats_per_cluster.items() if s.get("std") is not None),
            key=lambda c: cap_stats_per_cluster[c]["std"],  # type: ignore[arg-type, index]
            default=None,
        )
        loosest = max(
            (c for c, s in cap_stats_per_cluster.items() if s.get("std") is not None),
            key=lambda c: cap_stats_per_cluster[c]["std"],  # type: ignore[arg-type, index]
            default=None,
        )
        cap_aggr["tightest_cluster"] = tightest
        cap_aggr["loosest_cluster"] = loosest
        cap_aggr["mean_of_std"] = (
            float(np.mean([s["std"] for s in cap_stats_per_cluster.values() if s.get("std") is not None]))
            if any(s.get("std") is not None for s in cap_stats_per_cluster.values()) else None
        )
        # F-style ratio: between-group / within-group spread on capacity.
        if cap_within_vars and len(cap_group_means) >= 2:
            between = float(np.var(cap_group_means, ddof=1))
            within_mean = float(np.mean(cap_within_vars))
            cap_aggr["f_ratio"] = float(between / within_mean) if within_mean > 0 else None

    # Per-feature mean-of-stds (so the chat can flag the noisiest feature).
    feature_mean_std: Dict[str, Optional[float]] = {}
    for f in all_features:
        stds = []
        for r in out_rows:
            cell = r.get(f)
            if isinstance(cell, dict) and cell.get("std") is not None:
                stds.append(cell["std"])
        feature_mean_std[f] = float(np.mean(stds)) if stds else None

    return {
        "features": all_features,
        "per_cluster": out_rows,
        "capacity_aggregate": cap_aggr,
        "feature_mean_std": feature_mean_std,
        "overall_capacity_mean": overall_cap_mean,
    }


def summarize_clusters(
    merged_df: pd.DataFrame,
    labels: List[int],
    *,
    training_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Bundle everything the frontend needs to render the cluster view.

    Output keys:
        scatter:    points + cluster colours (OCV vs IR)
        boxplot:    per-cluster capacity arrays
        fingerprint: per-cluster normalised mean of electrochemical indicators
        per_cluster: stats + cell IDs
    """
    df = _normalise_indicator_columns(merged_df.reset_index(drop=True).copy())
    df["Cluster_label"] = labels[: len(df)]

    feature_cols = [
        c for c in df.columns
        if c not in _INDICATOR_NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    if "Capacity" in feature_cols:
        feature_cols.remove("Capacity")
    # Fingerprint should explain the actual clustering basis. In the current
    # system that is OCV/IR/CDDI only; Capacity is excluded above and remains
    # post-hoc.
    fingerprint_feature_cols = [
        c for c in (training_features or feature_cols)
        if c in df.columns and c != "Capacity"
    ]

    scatter_x, scatter_y = _choose_scatter_axes(
        df,
        training_features or fingerprint_feature_cols or feature_cols,
    )
    scatter_points = []
    for _, row in df.iterrows():
        ocv = _safe_float(row.get("OCV"))
        ir = _safe_float(row.get("IR"))
        cap = _safe_float(row.get("Capacity"))
        mmd = _safe_float(row.get("MMD_Loss"))
        x_val = _safe_float(row.get(scatter_x))
        y_val = _safe_float(row.get(scatter_y))
        scatter_points.append({
            "cell_num": int(row["Cell_num"]),
            "cluster": int(row["Cluster_label"]),
            "ocv": ocv,
            "ir": ir,
            "capacity": cap,
            "mmd_loss": mmd,
            "x": x_val,
            "y": y_val,
        })

    cluster_ids = sorted(int(c) for c in df["Cluster_label"].unique())

    boxplot = []
    for c in cluster_ids:
        if "Capacity" in df.columns:
            caps = (
                df.loc[df["Cluster_label"] == c, "Capacity"]
                .dropna()
                .astype(float)
                .tolist()
            )
        else:
            caps = []
        boxplot.append({"cluster": c, "values": caps})

    # Fingerprint: per-cluster mean of normalised clustering features.
    fp_data: Dict[str, Any] = {
        "features": fingerprint_feature_cols,
        "feature_kind": "cluster_features",
        "clusters": [],
    }
    if fingerprint_feature_cols:
        norm = df[fingerprint_feature_cols].apply(pd.to_numeric, errors="coerce")
        col_min = norm.min(axis=0)
        col_max = norm.max(axis=0)
        denom = (col_max - col_min).replace(0, np.nan)
        norm = (norm - col_min) / denom
        norm = norm.fillna(0.5)
        for c in cluster_ids:
            mask = df["Cluster_label"] == c
            means = norm.loc[mask, fingerprint_feature_cols].mean(axis=0)
            fp_data["clusters"].append({
                "cluster": c,
                "values": [
                    None if np.isnan(v) else float(v) for v in means.tolist()
                ],
            })

    per_cluster = []
    for c in cluster_ids:
        sub = df[df["Cluster_label"] == c]
        caps = (
            sub["Capacity"].dropna().astype(float)
            if "Capacity" in sub.columns else pd.Series(dtype=float)
        )
        mmd = sub["MMD_Loss"].dropna().astype(float)
        cell_ids = sub["Cell_num"].astype(int).tolist()
        cap_mean = _safe_float(caps.mean()) if len(caps) else None
        cap_std = _safe_float(caps.std(ddof=1)) if len(caps) > 1 else None
        per_cluster.append({
            "cluster": c,
            "size": int(len(sub)),
            "cell_ids": cell_ids,
            "capacity": {
                "mean": cap_mean,
                "std": cap_std,
                "min": _safe_float(caps.min()) if len(caps) else None,
                "max": _safe_float(caps.max()) if len(caps) else None,
            },
            "mmd_loss": {
                "mean": _safe_float(mmd.mean()) if len(mmd) else None,
                "std": _safe_float(mmd.std(ddof=1)) if len(mmd) > 1 else None,
            },
        })

    # ---- dispersion: per-cluster × per-feature std/cv, plus aggregates ----
    # Used by the chat-side summary ("Cluster 3 has the tightest capacity:
    # std=0.018, CV=0.93%") and by the downloadable XLSX artifact.
    dispersion = _compute_dispersion(df, feature_cols, cluster_ids)

    return {
        "scatter": scatter_points,
        "boxplot": boxplot,
        "fingerprint": fp_data,
        "per_cluster": per_cluster,
        "dispersion": dispersion,
        # IR vs Capacity is the canonical battery-aging scatter (paper-style):
        # internal resistance rises and usable capacity drops together. Each
        # scatter_point carries OCV/IR/Capacity/MMD so the frontend can swap.
        "axes": {"x": scatter_x, "y": scatter_y},
        "n_samples": int(len(df)),
        "k": len(cluster_ids),
    }


# ==========================================
# 7. one-shot helper used by ClusterAgent
# ==========================================

def cluster_dataset(
    *,
    mmd_csv: Optional[Path] = None,
    indicator_xlsx: Path,
    out_xlsx: Optional[Path] = None,
    pop_size: int = 200,
    max_iter: int = 80,
    min_k: int = 2,
    max_k: Optional[int] = None,
    use_capacity: bool = False,
    seed: Optional[int] = 42,
    on_iter: Optional[Callable[[int, Individual], None]] = None,
    feedback_metric: str = "combined",
    feature_weights: Optional[Dict[str, Any]] = None,
    posthoc_xlsx: Optional[Path] = None,
) -> Dict[str, Any]:
    if mmd_csv is None:
        merged = load_cluster_input(indicator_xlsx)
        if out_xlsx is not None:
            out_xlsx.parent.mkdir(parents=True, exist_ok=True)
            merged.rename(columns={"MMD_Loss": "CDDI"}).to_excel(out_xlsx, index=False)
    else:
        merged = merge_mmd_into_indicator(mmd_csv, indicator_xlsx, out_xlsx=out_xlsx)

    if "MMD_Loss" not in merged.columns:
        raise KeyError("After merge, MMD_Loss column missing")

    merged, data_norm, cap_norm, feature_cols, weights, has_capacity = _build_training_inputs(
        merged,
        mode="main",
        feature_weights=feature_weights,
        require_capacity=False,
    )

    result = run_adaptive_kmeans(
        data_norm, cap_norm,
        pop_size=pop_size, max_iter=max_iter,
        min_k=min_k, max_k=max_k,
        use_capacity=False, seed=seed,
        on_iter=on_iter,
        feedback_metric=feedback_metric,
    )

    display_df = _attach_posthoc_capacity(merged, posthoc_xlsx)
    summary = summarize_clusters(display_df, result.labels, training_features=feature_cols)

    cluster_xlsx: Optional[str] = None
    if out_xlsx is not None:
        cluster_xlsx_path = out_xlsx.with_name(out_xlsx.stem + "_Cluster_label.xlsx")
        out_df = display_df.drop(columns=["Cluster_label"], errors="ignore").copy()
        out_df.insert(1, "Cluster_label", result.labels[: len(out_df)])
        out_df = out_df.rename(columns={"MMD_Loss": "CDDI"})
        cluster_xlsx_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_excel(cluster_xlsx_path, index=False)
        cluster_xlsx = str(cluster_xlsx_path)

    return {
        "k": result.k,
        "fitness": result.fitness,
        "dbi": result.dbi,
        "elapsed_s": result.elapsed_s,
        "history": result.history,
        "labels": result.labels,
        "summary": summary,
        "cluster_xlsx": cluster_xlsx,
        "merged_xlsx": str(out_xlsx) if out_xlsx else None,
        "n_samples": int(len(merged)),
        "cell_nums": [int(c) for c in merged["Cell_num"].tolist()] if "Cell_num" in merged.columns else [],
        "feature_cols": feature_cols,
        "feature_weights": weights,
    }


# ==========================================
# 8. sub-clustering on a subset (used by RefinementAgent)
# ==========================================

def cluster_subset(
    *,
    mmd_csv: Optional[Path] = None,
    indicator_xlsx: Path,
    cell_nums: List[int],
    out_xlsx: Optional[Path] = None,
    pop_size: int = 120,
    max_iter: int = 60,
    min_k: int = 2,
    max_k: int = 4,
    use_capacity: bool = False,
    seed: Optional[int] = 42,
    on_iter: Optional[Callable[[int, Individual], None]] = None,
    feedback_metric: str = "combined",
    feature_weights: Optional[Dict[str, Any]] = None,
    posthoc_xlsx: Optional[Path] = None,
) -> Dict[str, Any]:
    """Re-cluster only the rows whose Cell_num is in ``cell_nums``.

    Used by RefinementAgent to produce a sub-cluster overlay for a chosen
    parent cluster. Returns the same shape as :func:`cluster_dataset` but
    with smaller ``max_k`` defaults appropriate for a subset.
    """
    wanted = set(int(c) for c in cell_nums)
    if len(wanted) < 4:
        raise RuntimeError(
            f"Sub-cluster needs at least 4 cells, got {len(wanted)} — refinement skipped."
        )

    if mmd_csv is None:
        merged_full = load_cluster_input(indicator_xlsx)
    else:
        merged_full = merge_mmd_into_indicator(mmd_csv, indicator_xlsx, out_xlsx=None)
    merged_full["Cell_num"] = merged_full["Cell_num"].astype(int)
    merged = merged_full[merged_full["Cell_num"].isin(wanted)].reset_index(drop=True)
    if len(merged) < 4:
        raise RuntimeError(
            f"After merge only {len(merged)} of the requested {len(wanted)} cells "
            "have valid indicator + MMD rows; refinement skipped."
        )

    if "MMD_Loss" not in merged.columns:
        raise KeyError("After merge, MMD_Loss column missing")

    merged, data_norm, cap_norm, feature_cols, weights, has_capacity = _build_training_inputs(
        merged,
        mode="main",
        feature_weights=feature_weights,
        require_capacity=False,
    )

    capped_max_k = min(max_k, max(min_k, len(merged) // 2))
    result = run_adaptive_kmeans(
        data_norm, cap_norm,
        pop_size=pop_size, max_iter=max_iter,
        min_k=min_k, max_k=capped_max_k,
        use_capacity=False, seed=seed,
        on_iter=on_iter,
        feedback_metric=feedback_metric,
    )

    display_df = _attach_posthoc_capacity(merged, posthoc_xlsx)
    summary = summarize_clusters(display_df, result.labels, training_features=feature_cols)

    cluster_xlsx: Optional[str] = None
    if out_xlsx is not None:
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)
        out_df = display_df.drop(columns=["Cluster_label"], errors="ignore").copy()
        out_df.insert(1, "Cluster_label", result.labels[: len(out_df)])
        out_df = out_df.rename(columns={"MMD_Loss": "CDDI"})
        out_df.to_excel(out_xlsx, index=False)
        cluster_xlsx = str(out_xlsx)

    return {
        "k": result.k,
        "fitness": result.fitness,
        "dbi": result.dbi,
        "elapsed_s": result.elapsed_s,
        "history": result.history,
        "labels": result.labels,
        "summary": summary,
        "cluster_xlsx": cluster_xlsx,
        "n_samples": int(len(merged)),
        "cell_nums": [int(c) for c in merged["Cell_num"].tolist()],
        "feature_cols": feature_cols,
        "feature_weights": weights,
    }
