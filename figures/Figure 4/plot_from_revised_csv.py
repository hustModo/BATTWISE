"""Draw the revised six-panel figure from the saved 45-model CSV only."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "plot_data" / "revised_45_models_all_panels.csv"
DATASETS = ["first_100s", "first_200s", "first_500s"]
LABELS = ["First 100 s", "First 200 s", "First 500 s"]
COLORS = ["#D55E00", "#0072B2", "#009E73"]
PANELS = [
    ("score", "a  Search performance", "Final score (higher is better)"),
    ("cddi_33feature_distance_spearman", "b  CDDI vs 33-feature distance",
     "Pairwise Spearman rho (higher is better)"),
    ("cddi_capacity_distance_spearman", "c  CDDI vs capacity difference",
     "Pairwise Spearman rho (higher is better)"),
    ("mean_within_cluster_capacity_sd", "d  Mean within-cluster capacity SD",
     "Mean capacity SD (lower is better)"),
    ("cluster_agreement_mean", "e  Clustering stability",
     "Mean adjusted Rand index (higher is better)"),
    ("mean_normalized_within_cluster_33feature_sd", "f  33-feature within-cluster dispersion",
     "Mean normalized SD (lower is better)"),
]


def draw_panel(ax: plt.Axes, frame: pd.DataFrame, metric: str,
               title: str, ylabel: str) -> None:
    values = [frame.loc[frame["dataset"].eq(dataset), metric].to_numpy(float)
              for dataset in DATASETS]
    boxplot = ax.boxplot(
        values,
        positions=np.arange(len(DATASETS)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202124", "linewidth": 1.35},
        whiskerprops={"color": "#666666", "linewidth": 0.9},
        capprops={"color": "#666666", "linewidth": 0.9},
    )
    for patch, color in zip(boxplot["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.25)

    for index, (array, color) in enumerate(zip(values, COLORS)):
        order = np.argsort(array, kind="stable")
        jitter = np.linspace(-0.16, 0.16, len(array))
        x_positions = np.empty(len(array))
        x_positions[order] = index + jitter
        ax.scatter(
            x_positions, array, s=22, color=color, alpha=0.88,
            edgecolor="white", linewidth=0.4, zorder=3,
        )
        data_range = float(np.ptp(array))
        offset = data_range * 0.025 if data_range > 0 else max(abs(float(array[0])) * 0.01, 0.005)
        ax.text(index, float(np.max(array)) + offset, f"n={len(array)}",
                ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(np.arange(len(DATASETS)))
    ax.set_xticklabels(LABELS, rotation=20, ha="right")
    ax.set_title(title, loc="left", pad=7)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.margins(y=0.08)
    for spine in ax.spines.values():
        spine.set_color("#70757A")
        spine.set_linewidth(0.75)


def main() -> None:
    frame = pd.read_csv(SOURCE)
    metrics = [panel[0] for panel in PANELS]
    missing_columns = sorted(set(metrics + ["dataset", "sample_id"]) - set(frame.columns))
    if missing_columns:
        raise RuntimeError(f"Missing required columns: {missing_columns}")
    counts = frame.groupby("dataset").size().reindex(DATASETS)
    if not (counts == 15).all():
        raise RuntimeError(f"Expected 15 models per dataset, found {counts.to_dict()}")
    if frame[metrics].isna().any().any():
        raise RuntimeError("The plotting metrics contain missing values")
    if frame.duplicated(["dataset", "sample_id"]).any():
        raise RuntimeError("Duplicate model IDs found")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
    })
    figure, axes = plt.subplots(2, 3, figsize=(12.8, 8.1))
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.11, top=0.90,
                           wspace=0.36, hspace=0.44)
    for axis, panel in zip(axes.flat, PANELS):
        draw_panel(axis, frame, *panel)

    figure.suptitle(
        "CDDI model performance and electrochemical validity (n=15 per dataset)",
        y=0.973, fontsize=13, fontweight="bold",
    )
    figure.text(
        0.5, 0.02,
        "Panels b and f use 33 CV/EIS/PITT/ICA features; boxes show median and IQR, and points show all models.",
        ha="center", fontsize=8, color="#4C566A",
    )
    png = HERE / "Figure4_revised_45_models_from_csv.png"
    pdf = HERE / "Figure4_revised_45_models_from_csv.pdf"
    figure.savefig(png, dpi=600, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    print({"source": str(SOURCE), "rows": len(frame),
           "counts": counts.to_dict(), "png": str(png), "pdf": str(pdf)})


if __name__ == "__main__":
    main()
