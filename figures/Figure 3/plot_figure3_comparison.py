"""Figure 3 new: statistical comparison of BATTWISE-CDDI and baselines.

The figure reproduces the visual language of Figure 2 while reading only the
saved formal 30-seed outputs and precomputed statistical tables.  It does not
rerun representation learning, clustering, or hypothesis tests.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from style import (
    METHOD_COLORS,
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    apply_style,
    metric_title,
    panel_label,
    save_figure,
)


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
METHOD_ORDER = ["BATTWISE-CDDI", "KPCA-1D", "FC-VAE-1D", "DECM-1D", "OCV-IR"]
BASELINES = ["KPCA-1D", "FC-VAE-1D", "DECM-1D", "OCV-IR"]
PRIMARY_METRICS = [
    "capacity_mean_cluster_variance",
    "capacity_npwv",
    "electrochemical_mean_npwv",
]
SHORT_METRICS = {
    "capacity_mean_cluster_variance": "Capacity variance",
    "capacity_npwv": "Capacity nPWV",
    "electrochemical_mean_npwv": "33-feature mean nPWV",
}


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def _closed_frame(ax: plt.Axes) -> None:
    """Use a closed rectangular frame for every panel."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#7A7F85")
        spine.set_linewidth(0.7)


def _distribution_panel(
    ax: plt.Axes, metrics: pd.DataFrame, metric: str, friedman: pd.DataFrame
) -> None:
    series = [metrics.loc[metrics["method"] == method, metric].to_numpy() for method in METHOD_ORDER]
    box = ax.boxplot(
        series,
        positions=np.arange(len(METHOD_ORDER)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.1},
        whiskerprops={"color": "#80868D", "linewidth": 0.8},
        capprops={"color": "#80868D", "linewidth": 0.8},
    )
    for patch, method in zip(box["boxes"], METHOD_ORDER):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.22)
        patch.set_edgecolor(METHOD_COLORS[method])
    jitter = np.linspace(-0.18, 0.18, 30)
    for position, (method, values) in enumerate(zip(METHOD_ORDER, series)):
        ax.scatter(
            position + jitter,
            values,
            s=12,
            facecolor=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.25,
            alpha=0.78,
            zorder=3,
        )
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels(["BATTWISE", "KPCA", "FC-VAE", "DECM", "OCV–IR"], rotation=24, ha="right")
    ax.set_ylabel(metric_title(metric))
    row = friedman.loc[friedman["metric"] == metric].iloc[0]
    ax.text(0.02, 0.98, f"30 paired seeds\nFriedman p={row['p_value']:.1e}", transform=ax.transAxes, ha="left", va="top", fontsize=6.5)


def _improvement_heatmap(ax: plt.Axes, paired: pd.DataFrame) -> None:
    matrix = np.full((len(PRIMARY_METRICS), len(BASELINES)), np.nan)
    for i, metric in enumerate(PRIMARY_METRICS):
        for j, baseline in enumerate(BASELINES):
            row = paired.loc[(paired["metric"] == metric) & (paired["baseline"] == baseline)].iloc[0]
            matrix[i, j] = row["median_relative_improvement_pct"]
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=sns.diverging_palette(15, 145, s=75, l=50, as_cmap=True),
        center=0,
        annot=True,
        fmt=".1f",
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        xticklabels=["KPCA", "FC-VAE", "DECM", "OCV–IR"],
        yticklabels=[SHORT_METRICS[m] for m in PRIMARY_METRICS],
        annot_kws={"fontsize": 6.7},
    )
    ax.set_title("Median paired improvement (%)")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)


def _stability_panel(ax: plt.Axes, stability: pd.DataFrame) -> None:
    table = stability.set_index("method").loc[METHOD_ORDER].reset_index()
    for y, row in table.iterrows():
        method = row["method"]
        ax.hlines(y, row["min_pairwise_ari"], row["max_pairwise_ari"], color=METHOD_COLORS[method], lw=1.1, alpha=0.55)
        ax.errorbar(row["mean_pairwise_ari"], y, xerr=row["std_pairwise_ari"], fmt="o", color=METHOD_COLORS[method], capsize=2, ms=5)
        # Keep the annotation inside the panel frame and right-align it so it
        # cannot overlap the whisker/mean marker, especially for the
        # high-stability BATTWISE and OCV–IR rows.
        ax.text(
            1.215,
            y,
            f"{int(row['unique_partitions'])} partitions",
            ha="right",
            va="center",
            fontsize=6.5,
            clip_on=True,
        )
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels(["BATTWISE", "KPCA", "FC-VAE", "DECM", "OCV–IR"])
    ax.invert_yaxis()
    ax.set_xlim(0.68, 1.23)
    ax.set_xlabel("Pairwise ARI across seeds")
    ax.set_title("Stability across seeds")


def _win_loss_panel(ax: plt.Axes, paired: pd.DataFrame) -> None:
    rows = []
    for metric in PRIMARY_METRICS:
        for baseline in BASELINES:
            row = paired.loc[(paired["metric"] == metric) & (paired["baseline"] == baseline)].iloc[0]
            short_label = {
                "capacity_mean_cluster_variance": "Cap. var.",
                "capacity_npwv": "Cap. nPWV",
                "electrochemical_mean_npwv": "33-feat. nPWV",
            }[metric]
            rows.append({"label": f"{short_label} · {baseline.replace('-1D', '')}", "wins": row["wins"], "ties": row["ties"], "losses": row["losses"]})
    table = pd.DataFrame(rows)
    y = np.arange(len(table))
    ax.barh(y, table["wins"], color=POSITIVE, label="Win")
    ax.barh(y, table["ties"], left=table["wins"], color=NEUTRAL, label="Tie")
    ax.barh(y, table["losses"], left=table["wins"] + table["ties"], color=NEGATIVE, label="Loss")
    ax.set_yticks(y)
    ax.set_yticklabels(table["label"], fontsize=5.3)
    ax.invert_yaxis()
    ax.set_xlim(0, 30)
    ax.set_xlabel("Paired seeds")
    ax.set_title(
        "Per-seed win/tie/loss analysis\nBATTWISE vs. baselines (n=30)",
        pad=20,
        fontsize=8.8,
    )
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.2)


def build_figure() -> plt.Figure:
    apply_style()
    metrics = _load("metrics_long.csv")
    paired = _load("paired_battwise_tests.csv")
    friedman = _load("friedman_tests.csv")
    stability = _load("clustering_stability.csv")
    # Use the Figure 2 visual language but a larger canvas so long metric
    # labels and the per-comparison win/tie/loss rows remain readable.  A
    # fixed subplot grid gives all six panels the same horizontal and vertical
    # gaps instead of letting label extents change the spacing.
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 7.6), constrained_layout=False)
    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.105,
        top=0.925,
        wspace=0.60,
        hspace=0.52,
    )
    for ax, metric, label in zip(axes[0], PRIMARY_METRICS, ["a", "b", "c"]):
        _distribution_panel(ax, metrics, metric, friedman)
        panel_label(ax, label)
    _improvement_heatmap(axes[1, 0], paired)
    panel_label(axes[1, 0], "d", x=-0.24, y=1.04)
    _stability_panel(axes[1, 1], stability)
    panel_label(axes[1, 1], "e", x=-0.24, y=1.04)
    _win_loss_panel(axes[1, 2], paired)
    panel_label(axes[1, 2], "f", x=-0.24, y=1.04)
    for ax in axes.ravel():
        _closed_frame(ax)
    return fig


def export_summary_tables() -> None:
    metrics = _load("metrics_long.csv")
    metrics.groupby("method", sort=False)[PRIMARY_METRICS].agg(["mean", "std", "median"]).to_csv(DATA / "method_metric_summary.csv")
    paired = _load("paired_battwise_tests.csv")
    paired.loc[paired["metric"].isin(PRIMARY_METRICS)].to_csv(DATA / "primary_paired_statistics.csv", index=False)


def main() -> None:
    export_summary_tables()
    figure = build_figure()
    save_figure(figure, HERE, "Figure3_comparison_large")
    plt.close(figure)
    print(f"Saved {HERE / 'Figure3_comparison_large.pdf'}")
    print(f"Saved {HERE / 'Figure3_comparison_large.png'}")


if __name__ == "__main__":
    main()
