from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from sklearn.metrics import adjusted_rand_score


ROOT = Path(__file__).resolve().parents[2] / "outputs" / "benchmark"
OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "statistics"

METHOD_ORDER = [
    "battwise_cddi",
    "kpca_1d",
    "fc_vae_1d",
    "decm_1d",
    "ocv_ir",
]

METHOD_NAMES = {
    "battwise_cddi": "BATTWISE-CDDI",
    "kpca_1d": "KPCA-1D",
    "fc_vae_1d": "FC-VAE-1D",
    "decm_1d": "DECM-1D",
    "ocv_ir": "OCV-IR",
}

METRICS = {
    "capacity_mean_cluster_variance": "lower",
    "capacity_pooled_within_variance": "lower",
    "capacity_npwv": "lower",
    "electrochemical_mean_npwv": "lower",
    "electrochemical_median_npwv": "lower",
    "dbi": "lower",
    "calinski_harabasz": "higher",
    "silhouette": "higher",
    "sse": "lower",
    "ga_fitness": "lower",
    "ga_compactness": "lower",
    "ga_elapsed_s": "lower",
    "representation_elapsed_s": "lower",
}

PRIMARY_METRICS = [
    "capacity_mean_cluster_variance",
    "capacity_npwv",
    "electrochemical_mean_npwv",
    "dbi",
    "silhouette",
    "sse",
    "ga_fitness",
]

ECM_FEATURES = {
    "Rs",
    "CPE1-T",
    "Rp",
    "CPE3-T",
    "R4",
    "CPE2-T",
    "R3",
    "Wo-R",
    "Wo-T",
}


def validate(
    metrics: pd.DataFrame,
    assignments: pd.DataFrame,
    embeddings: pd.DataFrame,
    electro: pd.DataFrame,
) -> dict[str, object]:
    expected_seeds = set(range(30))
    expected_methods = set(METHOD_ORDER)
    assert len(metrics) == 150
    assert set(metrics["method_key"]) == expected_methods
    assert set(metrics["seed"]) == expected_seeds
    assert not metrics.duplicated(["method_key", "seed"]).any()
    assert (metrics["n_cells"] == 71).all()
    assert (metrics["k"] == 6).all()
    assert np.isfinite(metrics[list(METRICS)]).all().all()

    for frame, name in ((assignments, "assignments"), (embeddings, "embeddings")):
        assert len(frame) == 150 * 71, name
        sizes = frame.groupby(["method_key", "seed"]).size()
        assert (sizes == 71).all(), name
        assert set(frame["method_key"]) == expected_methods
        assert set(frame["seed"]) == expected_seeds

    assert assignments.groupby(["method_key", "seed"])["Cluster_label"].nunique().eq(6).all()
    assert np.isfinite(assignments[["OCV_norm", "IR_norm", "z1"]]).all().all()
    assert not electro.duplicated(["method_key", "seed", "feature"]).any()
    assert electro.groupby(["method_key", "seed"]).size().eq(33).all()
    assert np.isfinite(electro["npwv"]).all()

    return {
        "metrics_rows": int(len(metrics)),
        "assignment_rows": int(len(assignments)),
        "embedding_rows": int(len(embeddings)),
        "electrochemical_rows": int(len(electro)),
        "methods": len(expected_methods),
        "seeds_per_method": len(expected_seeds),
        "cells_per_run": 71,
        "clusters_per_run": 6,
    }


def descriptive_statistics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, direction in METRICS.items():
        pivot = metrics.pivot(index="seed", columns="method_key", values=metric)
        oriented = pivot if direction == "lower" else -pivot
        ranks = oriented.rank(axis=1, method="average", ascending=True)
        for method in METHOD_ORDER:
            values = pivot[method].to_numpy(dtype=float)
            rows.append(
                {
                    "metric": metric,
                    "direction": direction,
                    "method_key": method,
                    "method": METHOD_NAMES[method],
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "mean_rank": float(ranks[method].mean()),
                }
            )
    return pd.DataFrame(rows)


def friedman_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, direction in METRICS.items():
        pivot = metrics.pivot(index="seed", columns="method_key", values=metric)
        statistic, p_value = friedmanchisquare(
            *(pivot[method].to_numpy(dtype=float) for method in METHOD_ORDER)
        )
        oriented = pivot if direction == "lower" else -pivot
        ranks = oriented.rank(axis=1, method="average", ascending=True)
        row: dict[str, object] = {
            "metric": metric,
            "direction": direction,
            "friedman_chi2": float(statistic),
            "p_value": float(p_value),
        }
        for method in METHOD_ORDER:
            row[f"rank_{method}"] = float(ranks[method].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def rank_biserial(improvements: np.ndarray) -> float:
    improvements = np.asarray(improvements, dtype=float)
    nonzero = improvements[~np.isclose(improvements, 0.0, rtol=0.0, atol=1e-15)]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank_index, original_index in enumerate(order):
        candidate = min(1.0, (count - rank_index) * p_values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted.tolist()


def paired_battwise_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baselines = [method for method in METHOD_ORDER if method != "battwise_cddi"]
    for metric in PRIMARY_METRICS:
        direction = METRICS[metric]
        pivot = metrics.pivot(index="seed", columns="method_key", values=metric)
        metric_rows: list[dict[str, object]] = []
        raw_p_values: list[float] = []
        for baseline in baselines:
            battwise = pivot["battwise_cddi"].to_numpy(dtype=float)
            comparison = pivot[baseline].to_numpy(dtype=float)
            improvement = (
                comparison - battwise
                if direction == "lower"
                else battwise - comparison
            )
            if np.allclose(improvement, 0.0, rtol=0.0, atol=1e-15):
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(
                    improvement,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                statistic, p_value = float(result.statistic), float(result.pvalue)
            tolerance = 1e-12
            wins = int((improvement > tolerance).sum())
            ties = int((np.abs(improvement) <= tolerance).sum())
            losses = int((improvement < -tolerance).sum())
            denominator = np.maximum(np.abs(comparison), 1e-15)
            relative = 100.0 * improvement / denominator
            row = {
                "metric": metric,
                "direction": direction,
                "baseline_key": baseline,
                "baseline": METHOD_NAMES[baseline],
                "battwise_mean": float(battwise.mean()),
                "baseline_mean": float(comparison.mean()),
                "mean_improvement": float(improvement.mean()),
                "median_improvement": float(np.median(improvement)),
                "mean_relative_improvement_pct": float(relative.mean()),
                "median_relative_improvement_pct": float(np.median(relative)),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "rank_biserial": rank_biserial(improvement),
            }
            metric_rows.append(row)
            raw_p_values.append(p_value)
        adjusted = holm_adjust(raw_p_values)
        for row, adjusted_p in zip(metric_rows, adjusted, strict=True):
            row["holm_p_within_metric"] = adjusted_p
            rows.append(row)
    return pd.DataFrame(rows)


def canonical_partition(labels: np.ndarray) -> str:
    labels = np.asarray(labels, dtype=int)
    coassignment = labels[:, None] == labels[None, :]
    signature = np.packbits(coassignment[np.triu_indices(len(labels), k=1)])
    return hashlib.sha256(signature.tobytes()).hexdigest()


def clustering_stability(assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    label_by_method_seed: dict[tuple[str, int], np.ndarray] = {}
    for (method, seed), group in assignments.groupby(["method_key", "seed"]):
        ordered = group.sort_values("Cell_num")
        label_by_method_seed[(str(method), int(seed))] = ordered[
            "Cluster_label"
        ].to_numpy(dtype=int)

    for method in METHOD_ORDER:
        seeds = sorted(
            seed
            for current_method, seed in label_by_method_seed
            if current_method == method
        )
        aris = [
            adjusted_rand_score(
                label_by_method_seed[(method, left)],
                label_by_method_seed[(method, right)],
            )
            for left, right in combinations(seeds, 2)
        ]
        signatures = {
            canonical_partition(label_by_method_seed[(method, seed)])
            for seed in seeds
        }
        rows.append(
            {
                "method_key": method,
                "method": METHOD_NAMES[method],
                "pair_count": len(aris),
                "mean_pairwise_ari": float(np.mean(aris)),
                "std_pairwise_ari": float(np.std(aris, ddof=1)),
                "min_pairwise_ari": float(np.min(aris)),
                "max_pairwise_ari": float(np.max(aris)),
                "unique_partitions": len(signatures),
            }
        )

    matrix = pd.DataFrame(
        np.eye(len(METHOD_ORDER)),
        index=METHOD_ORDER,
        columns=METHOD_ORDER,
        dtype=float,
    )
    for left, right in combinations(METHOD_ORDER, 2):
        values = [
            adjusted_rand_score(
                label_by_method_seed[(left, seed)],
                label_by_method_seed[(right, seed)],
            )
            for seed in range(30)
        ]
        matrix.loc[left, right] = float(np.mean(values))
        matrix.loc[right, left] = float(np.mean(values))
    matrix.index.name = "method_key"
    return pd.DataFrame(rows), matrix


def per_seed_best_counts(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in PRIMARY_METRICS:
        direction = METRICS[metric]
        pivot = metrics.pivot(index="seed", columns="method_key", values=metric)
        best = pivot.min(axis=1) if direction == "lower" else pivot.max(axis=1)
        for method in METHOD_ORDER:
            count = int(
                np.isclose(
                    pivot[method].to_numpy(dtype=float),
                    best.to_numpy(dtype=float),
                    rtol=0.0,
                    atol=1e-12,
                ).sum()
            )
            rows.append(
                {
                    "metric": metric,
                    "direction": direction,
                    "method_key": method,
                    "method": METHOD_NAMES[method],
                    "best_or_tied_seeds": count,
                }
            )
    return pd.DataFrame(rows)


def electrochemical_feature_comparison(electro: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_means = (
        electro.groupby(["method_key", "feature"], as_index=False)["npwv"]
        .mean()
        .pivot(index="feature", columns="method_key", values="npwv")
    )
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    battwise = feature_means["battwise_cddi"]
    for baseline in [method for method in METHOD_ORDER if method != "battwise_cddi"]:
        improvement = feature_means[baseline] - battwise
        relative = 100.0 * improvement / feature_means[baseline].abs().clip(lower=1e-15)
        tolerance = 1e-12
        summary_rows.append(
            {
                "baseline_key": baseline,
                "baseline": METHOD_NAMES[baseline],
                "features_battwise_better": int((improvement > tolerance).sum()),
                "features_tied": int((improvement.abs() <= tolerance).sum()),
                "features_battwise_worse": int((improvement < -tolerance).sum()),
                "mean_absolute_improvement": float(improvement.mean()),
                "median_relative_improvement_pct": float(relative.median()),
            }
        )
        for feature in feature_means.index:
            detail_rows.append(
                {
                    "feature": feature,
                    "baseline_key": baseline,
                    "battwise_mean_npwv": float(battwise.loc[feature]),
                    "baseline_mean_npwv": float(feature_means.loc[feature, baseline]),
                    "improvement": float(improvement.loc[feature]),
                    "relative_improvement_pct": float(relative.loc[feature]),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def electrochemical_subgroup_analysis(
    electro: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = electro.copy()
    grouped["feature_group"] = np.where(
        grouped["feature"].isin(ECM_FEATURES),
        "ECM_EIS_9",
        "curve_derived_24",
    )
    per_run = (
        grouped.groupby(
            ["method_key", "method", "seed", "feature_group"],
            as_index=False,
        )["npwv"]
        .mean()
        .rename(columns={"npwv": "mean_npwv"})
    )
    descriptive = (
        per_run.groupby(
            ["feature_group", "method_key", "method"],
            as_index=False,
        )["mean_npwv"]
        .agg(["mean", "std", "median"])
        .reset_index()
    )

    paired_rows: list[dict[str, object]] = []
    for feature_group in ("ECM_EIS_9", "curve_derived_24"):
        pivot = per_run[
            per_run["feature_group"].eq(feature_group)
        ].pivot(
            index="seed",
            columns="method_key",
            values="mean_npwv",
        )
        raw_p_values: list[float] = []
        group_rows: list[dict[str, object]] = []
        for baseline in [
            method for method in METHOD_ORDER if method != "battwise_cddi"
        ]:
            improvement = (
                pivot[baseline].to_numpy(dtype=float)
                - pivot["battwise_cddi"].to_numpy(dtype=float)
            )
            if np.allclose(improvement, 0.0, rtol=0.0, atol=1e-15):
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(
                    improvement,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                statistic, p_value = float(result.statistic), float(result.pvalue)
            tolerance = 1e-12
            row = {
                "feature_group": feature_group,
                "baseline_key": baseline,
                "baseline": METHOD_NAMES[baseline],
                "battwise_mean": float(pivot["battwise_cddi"].mean()),
                "baseline_mean": float(pivot[baseline].mean()),
                "wins": int((improvement > tolerance).sum()),
                "ties": int((np.abs(improvement) <= tolerance).sum()),
                "losses": int((improvement < -tolerance).sum()),
                "median_relative_improvement_pct": float(
                    np.median(
                        100.0
                        * improvement
                        / np.maximum(
                            np.abs(pivot[baseline].to_numpy(dtype=float)),
                            1e-15,
                        )
                    )
                ),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "rank_biserial": rank_biserial(improvement),
            }
            group_rows.append(row)
            raw_p_values.append(p_value)
        adjusted = holm_adjust(raw_p_values)
        for row, adjusted_p in zip(group_rows, adjusted, strict=True):
            row["holm_p_within_group"] = adjusted_p
            paired_rows.append(row)
    return descriptive, pd.DataFrame(paired_rows)


def main() -> None:
    metrics = pd.read_csv(ROOT / "metrics_long.csv")
    assignments = pd.read_csv(ROOT / "assignments.csv")
    embeddings = pd.read_csv(ROOT / "embeddings.csv")
    electro = pd.read_csv(ROOT / "electrochemical_metrics.csv")

    validation = validate(metrics, assignments, embeddings, electro)
    descriptive = descriptive_statistics(metrics)
    friedman = friedman_tests(metrics)
    paired = paired_battwise_tests(metrics)
    stability, cross_method_ari = clustering_stability(assignments)
    best_counts = per_seed_best_counts(metrics)
    feature_summary, feature_detail = electrochemical_feature_comparison(electro)
    subgroup_descriptive, subgroup_paired = electrochemical_subgroup_analysis(
        electro
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    descriptive.to_csv(OUTPUT / "descriptive_statistics.csv", index=False)
    friedman.to_csv(OUTPUT / "friedman_tests.csv", index=False)
    paired.to_csv(OUTPUT / "paired_battwise_tests.csv", index=False)
    stability.to_csv(OUTPUT / "clustering_stability.csv", index=False)
    cross_method_ari.to_csv(OUTPUT / "cross_method_ari.csv")
    best_counts.to_csv(OUTPUT / "per_seed_best_counts.csv", index=False)
    feature_summary.to_csv(OUTPUT / "electrochemical_feature_summary.csv", index=False)
    feature_detail.to_csv(OUTPUT / "electrochemical_feature_detail.csv", index=False)
    subgroup_descriptive.to_csv(
        OUTPUT / "electrochemical_subgroup_descriptive.csv",
        index=False,
    )
    subgroup_paired.to_csv(
        OUTPUT / "electrochemical_subgroup_paired.csv",
        index=False,
    )

    primary_descriptive = descriptive[
        descriptive["metric"].isin(PRIMARY_METRICS)
    ].copy()
    primary_descriptive["rank_by_mean"] = np.nan
    for metric in PRIMARY_METRICS:
        metric_mask = primary_descriptive["metric"].eq(metric)
        direction = METRICS[metric]
        primary_descriptive.loc[metric_mask, "rank_by_mean"] = (
            primary_descriptive.loc[metric_mask, "mean"].rank(
                method="min",
                ascending=direction == "lower",
            )
        )

    summary = {
        "validation": validation,
        "primary_descriptive": primary_descriptive.to_dict(orient="records"),
        "friedman_primary": friedman[
            friedman["metric"].isin(PRIMARY_METRICS)
        ].to_dict(orient="records"),
        "paired_battwise_primary": paired.to_dict(orient="records"),
        "stability": stability.to_dict(orient="records"),
        "cross_method_ari": cross_method_ari.to_dict(),
        "best_counts": best_counts.to_dict(orient="records"),
        "electrochemical_feature_summary": feature_summary.to_dict(
            orient="records"
        ),
        "electrochemical_subgroup_descriptive": subgroup_descriptive.to_dict(
            orient="records"
        ),
        "electrochemical_subgroup_paired": subgroup_paired.to_dict(
            orient="records"
        ),
    }
    (OUTPUT / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print("\nPRIMARY DESCRIPTIVE")
    print(
        primary_descriptive[
            ["metric", "method", "mean", "std", "mean_rank"]
        ].to_string(index=False)
    )
    print("\nFRIEDMAN")
    print(
        friedman[friedman["metric"].isin(PRIMARY_METRICS)][
            ["metric", "friedman_chi2", "p_value"]
        ].to_string(index=False)
    )
    print("\nBATTWISE PAIRED")
    print(
        paired[
            [
                "metric",
                "baseline",
                "wins",
                "ties",
                "losses",
                "median_relative_improvement_pct",
                "holm_p_within_metric",
                "rank_biserial",
            ]
        ].to_string(index=False)
    )
    print("\nSTABILITY")
    print(stability.to_string(index=False))
    print("\nELECTROCHEMICAL FEATURE COUNTS")
    print(feature_summary.to_string(index=False))
    print("\nELECTROCHEMICAL SUBGROUPS")
    print(subgroup_descriptive.to_string(index=False))
    print("\nELECTROCHEMICAL SUBGROUP PAIRED")
    print(subgroup_paired.to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze saved benchmark CSVs; never runs models or clustering.")
    parser.add_argument("--input", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    ROOT, OUTPUT = args.input.resolve(), args.output.resolve()
    main()
