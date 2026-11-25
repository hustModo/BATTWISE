# -*- coding: utf-8 -*-
"""Redraw every panel from Figure 3 new and Figure S3 as standalone files."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
COMPARISON_DATA = ROOT / "Figure 3 new" / "data"
DISCOVERY_DATA = ROOT / "Figure S3" / "data"
OUT_COMPARISON = OUT / "Figure3_comparison_panels"
OUT_DISCOVERY = OUT / "FigureS3_discovery_panels"
PANEL_SIZE_INCHES = 6.0
# Identical axes bounds for all nine square panels. This makes every black
# plotting frame line up exactly after the standalone panels are assembled.
COMMON_PANEL_ADJUST = dict(left=0.30, right=0.96, bottom=0.22, top=0.95)
DISCOVERY_ROW_ADJUST = COMMON_PANEL_ADJUST
COMPARISON_ROW_ADJUST = COMMON_PANEL_ADJUST
BOTTOM_ROW_ADJUST = COMMON_PANEL_ADJUST
TEXT_SIZE = 14.0
PANEL_LABEL_SIZE = TEXT_SIZE

METHOD_ORDER = ["BATTWISE-CDDI", "KPCA-1D", "FC-VAE-1D", "DECM-1D", "OCV-IR"]
BASELINES = ["KPCA-1D", "FC-VAE-1D", "DECM-1D", "OCV-IR"]
METHOD_LABELS = ["BATTWISE", "KPCA", "FC-VAE", "DECM", "OCV–IR"]
METHOD_COLORS = {
    "BATTWISE-CDDI": "#E64B35",
    "KPCA-1D": "#4DBBD5",
    "FC-VAE-1D": "#00A087",
    "DECM-1D": "#8E6C8A",
    "OCV-IR": "#7E7E7E",
}
PRIMARY_METRICS = [
    "capacity_mean_cluster_variance",
    "capacity_npwv",
    "electrochemical_mean_npwv",
]
METRIC_LABELS = {
    "capacity_mean_cluster_variance": "Mean within-cluster\ncapacity variance (Ah²)",
    "capacity_npwv": "Capacity nPWV",
    "electrochemical_mean_npwv": "Mean nPWV across 33\nevaluation variables",
}
SHORT_METRICS = {
    "capacity_mean_cluster_variance": "Capacity\nvariance",
    "capacity_npwv": "Capacity\nnPWV",
    "electrochemical_mean_npwv": "Electrochemical\nfeature nPWV",
}
TEXT = "#171717"
GRID = "#D8D8D8"
ORANGE = "#E64B35"
BLUE = "#3C78D8"
TEAL = "#00A087"
PURPLE = "#8E6C8A"
GOLD = "#E5A000"


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": TEXT_SIZE,
        "axes.titlesize": TEXT_SIZE,
        "axes.titleweight": "normal",
        "axes.labelsize": TEXT_SIZE,
        "xtick.labelsize": TEXT_SIZE,
        "ytick.labelsize": TEXT_SIZE,
        "legend.fontsize": TEXT_SIZE,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "#202020",
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "text.color": TEXT,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        # Keep a fixed square artboard for straightforward panel assembly.
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
    })


def style_axes(ax: plt.Axes, grid: str | None = "y") -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#202020")
        spine.set_linewidth(1.0)
    ax.tick_params(direction="in", length=4, width=0.9, top=False, right=False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.55, alpha=0.65, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.15, 1.035, label, transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE,
            fontweight="bold", ha="left", va="top", clip_on=False)


def save_panel(fig: plt.Figure, folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(PANEL_SIZE_INCHES, PANEL_SIZE_INCHES, forward=True)
    fig.savefig(folder / f"{stem}.png", dpi=600, bbox_inches=None, pad_inches=0)
    fig.savefig(folder / f"{stem}.pdf", bbox_inches=None, pad_inches=0)
    fig.savefig(folder / f"{stem}.svg", bbox_inches=None, pad_inches=0)
    plt.close(fig)


def new_panel(figsize=(4.45, 3.75)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(**COMPARISON_ROW_ADJUST)
    return fig, ax


def comparison_distribution(metrics, friedman, metric, label, stem):
    fig, ax = new_panel((4.65, 4.05))
    values = [metrics.loc[metrics["method"] == m, metric].to_numpy() for m in METHOD_ORDER]
    box = ax.boxplot(values, positions=np.arange(5), widths=0.58, patch_artist=True,
                     showfliers=False, medianprops={"color": "#171717", "linewidth": 1.5},
                     whiskerprops={"color": "#555555", "linewidth": 1.0},
                     capprops={"color": "#555555", "linewidth": 1.0})
    for patch, method in zip(box["boxes"], METHOD_ORDER):
        patch.set_facecolor(METHOD_COLORS[method]); patch.set_alpha(0.30)
        patch.set_edgecolor(METHOD_COLORS[method]); patch.set_linewidth(1.2)
    jitter = np.linspace(-0.19, 0.19, 30)
    for x, (method, ys) in enumerate(zip(METHOD_ORDER, values)):
        ax.scatter(x + jitter, ys, s=23, color=METHOD_COLORS[method], edgecolor="white",
                   linewidth=0.35, alpha=0.88, zorder=3)
    row = friedman.loc[friedman["metric"] == metric].iloc[0]
    ax.text(0.03, 0.97, f"30 paired seeds\nFriedman $P$ = {row['p_value']:.1e}",
            transform=ax.transAxes, ha="left", va="top", fontsize=TEXT_SIZE)
    ax.set_xticks(range(5)); ax.set_xticklabels(METHOD_LABELS, rotation=35, ha="right")
    ax.set_ylabel(METRIC_LABELS[metric])
    panel_label(ax, label); style_axes(ax)
    save_panel(fig, OUT_COMPARISON, stem)


def comparison_improvement(paired):
    fig, ax = new_panel((4.85, 4.05))
    fig.subplots_adjust(**BOTTOM_ROW_ADJUST)
    matrix = np.array([[paired.loc[(paired.metric == m) & (paired.baseline == b),
                                   "median_relative_improvement_pct"].iloc[0]
                        for b in BASELINES] for m in PRIMARY_METRICS])
    cmap = sns.diverging_palette(245, 20, s=82, l=55, center="light", as_cmap=True)
    sns.heatmap(matrix, ax=ax, cmap=cmap, center=0, annot=True, fmt=".1f", cbar=False,
                linewidths=1.2, linecolor="white", annot_kws={"fontsize": TEXT_SIZE},
                xticklabels=["KPCA", "FC-VAE", "DECM", "OCV–IR"],
                yticklabels=[SHORT_METRICS[m] for m in PRIMARY_METRICS],
                )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    panel_label(ax, "d")
    for spine in ax.spines.values(): spine.set_visible(True); spine.set_linewidth(1.0)
    save_panel(fig, OUT_COMPARISON, "d_median_paired_improvement")


def comparison_stability(stability):
    fig, ax = new_panel((5.35, 4.05))
    fig.subplots_adjust(**BOTTOM_ROW_ADJUST)
    table = stability.set_index("method").loc[METHOD_ORDER].reset_index()
    for y, row in table.iterrows():
        c = METHOD_COLORS[row.method]
        ax.hlines(y, row.min_pairwise_ari, row.max_pairwise_ari, color=c, lw=3.0, alpha=0.35)
        ax.errorbar(row.mean_pairwise_ari, y, xerr=row.std_pairwise_ari, fmt="o", color=c,
                    ecolor=c, capsize=3.5, elinewidth=1.5, ms=7, mec="white", mew=0.5)
        ax.text(1.155, y, f"{int(row.unique_partitions)} partitions",
                ha="right", va="center", fontsize=TEXT_SIZE, clip_on=True)
    ax.set_yticks(np.arange(5)); ax.set_yticklabels(METHOD_LABELS); ax.invert_yaxis()
    ax.set_xlim(0.68, 1.17); ax.set_xlabel("Pairwise ARI across seeds")
    panel_label(ax, "e"); style_axes(ax, grid="x")
    save_panel(fig, OUT_COMPARISON, "e_clustering_stability")


def comparison_win_loss(paired):
    fig, ax = new_panel((5.8, 5.4))
    fig.subplots_adjust(**BOTTOM_ROW_ADJUST)
    rows = []
    abbrev = {
        PRIMARY_METRICS[0]: "Cap.var.",
        PRIMARY_METRICS[1]: "Q nPWV",
        PRIMARY_METRICS[2]: "EC nPWV",
    }
    compact_baseline = {"KPCA-1D": "KPCA", "FC-VAE-1D": "FCVAE",
                        "DECM-1D": "DECM", "OCV-IR": "OCV-IR"}
    for metric in PRIMARY_METRICS:
        for baseline in BASELINES:
            r = paired.loc[(paired.metric == metric) & (paired.baseline == baseline)].iloc[0]
            rows.append((f"{abbrev[metric]}–{compact_baseline[baseline]}", r.wins, r.ties, r.losses))
    table = pd.DataFrame(rows, columns=["label", "wins", "ties", "losses"])
    y = np.arange(len(table))
    ax.barh(y, table.wins, color="#2E8B57", label="Win")
    ax.barh(y, table.ties, left=table.wins, color="#E5A000", label="Tie")
    ax.barh(y, table.losses, left=table.wins + table.ties, color="#D55E5E", label="Loss")
    ax.set_yticks(y); ax.set_yticklabels(table.label, fontsize=TEXT_SIZE); ax.invert_yaxis()
    ax.set_xlim(0, 30); ax.set_xlabel("Paired seeds")
    ax.xaxis.set_label_coords(0.5, -0.09)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.19), frameon=False)
    ax.text(-0.15, 1.035, "f", transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight="bold",
            ha="left", va="top", clip_on=False)
    style_axes(ax, grid="x")
    save_panel(fig, OUT_COMPARISON, "f_win_tie_loss")


def search_trace() -> pd.DataFrame:
    raw = pd.read_csv(DISCOVERY_DATA / "candidate_records.csv")
    raw["objective_num"] = pd.to_numeric(raw["objective"], errors="coerce")
    rows = []
    for gen, group in raw.groupby("generation", sort=True):
        score = -group.objective_num.dropna()
        rows.append((int(gen), len(group), len(score), score.max(), score.median()))
    return pd.DataFrame(rows, columns=["generation", "population_size", "finite_count", "best", "median"])


def frozen_cddi() -> pd.DataFrame:
    emb = pd.read_csv(DISCOVERY_DATA / "embeddings.csv")
    out = emb.loc[(emb.method == "BATTWISE-CDDI") & (emb.seed == 0), ["Cell_num", "z1"]].copy()
    out.z1 = pd.to_numeric(out.z1, errors="coerce")
    return out.dropna().sort_values("z1").reset_index(drop=True)


def descriptor_correlations() -> pd.DataFrame:
    desc = pd.read_excel(DISCOVERY_DATA / "Charging_curve_characteristics.xlsx", sheet_name="Sheet2")
    merged = desc.merge(frozen_cddi(), on="Cell_num", validate="one_to_one")
    features = [c for c in desc.columns if c != "Cell_num"]
    rows = []
    for i, feature in enumerate(features):
        stage = "Early charge" if i < 4 else ("CC" if i < 12 else "CV")
        rows.append((feature, float(merged[feature].corr(merged.z1)), stage))
    return pd.DataFrame(rows, columns=["feature", "r", "stage"])


def discovery_search():
    trace = search_trace(); fig, ax = new_panel((4.6, 4.0))
    fig.subplots_adjust(**DISCOVERY_ROW_ADJUST)
    ax.plot(trace.generation, trace.best, color=ORANGE, marker="o", ms=5, lw=2.2, label="Best feasible")
    ax.plot(trace.generation, trace["median"], color=PURPLE, marker="s", ms=4.3, lw=1.7,
            ls="--", label="Population median")
    ax.fill_between(trace.generation, trace["median"], trace.best, color=GOLD, alpha=0.16)
    lo = min(trace.best.min(), trace["median"].min()); hi = max(trace.best.max(), trace["median"].max())
    ax.set_ylim(lo - max(.02, .18*(hi-lo)), hi + max(.02, .18*(hi-lo)))
    ax.set_xticks([0,5,10,15,20]); ax.set_xlabel("Generation"); ax.set_ylabel("Mean |Pearson r|")
    ax.text(.04,.96,f"{trace.finite_count.sum()}/{trace.population_size.sum()} finite objectives\n{len(trace)} generations",
            transform=ax.transAxes, va="top", fontsize=TEXT_SIZE)
    ax.legend(loc="lower right", frameon=False)
    panel_label(ax,"a"); style_axes(ax)
    save_panel(fig, OUT_DISCOVERY, "a_candidate_search_trajectory")


def discovery_distribution():
    cddi=frozen_cddi(); fig,ax=new_panel((4.6,4.0)); ranks=np.arange(1,len(cddi)+1)
    fig.subplots_adjust(**DISCOVERY_ROW_ADJUST)
    colors=plt.get_cmap("viridis")(np.linspace(.12,.88,len(cddi)))
    ax.scatter(ranks,cddi.z1,s=28,c=colors,edgecolor="white",linewidth=.35,zorder=3)
    ax.plot(ranks,cddi.z1,color=TEAL,alpha=.38,lw=1.3)
    q1,med,q3=cddi.z1.quantile([.25,.5,.75]); ax.axhspan(q1,q3,color="#4DBBD5",alpha=.13)
    ax.axhline(med,color=ORANGE,lw=1.5,ls="--",label="Median")
    ax.set_xlabel("Cell rank (sorted by CDDI)"); ax.set_ylabel("CDDI")
    ax.text(.04,.96,f"n = {len(cddi)}\nmedian = {med:,.0f}\nIQR = {q1:,.0f}–{q3:,.0f}",
            transform=ax.transAxes,va="top",fontsize=TEXT_SIZE)
    ax.legend(loc="lower right",frameon=False)
    panel_label(ax,"b"); style_axes(ax)
    save_panel(fig,OUT_DISCOVERY,"b_cddi_distribution")


def discovery_correlations():
    corr=descriptor_correlations(); fig,ax=new_panel((5.5,6.4)); y=np.arange(len(corr))[::-1]
    fig.subplots_adjust(**DISCOVERY_ROW_ADJUST)
    colors={"Early charge":BLUE,"CC":GOLD,"CV":ORANGE}
    labels={
        "tChar":r"$t_{\mathrm{Char}}$", "Urest":r"$U_{\mathrm{rest}}$",
        "UCC,init":r"$U_{\mathrm{CC,init}}$", "∆Urest,CC":r"$\Delta U_{\mathrm{rest,CC}}$",
        "tCC":r"$t_{\mathrm{CC}}$", "dU/dtCC,fin":r"$\mathrm{d}U/\mathrm{d}t_{\mathrm{CC,fin}}$",
        "dU/dtCC,init":r"$\mathrm{d}U/\mathrm{d}t_{\mathrm{CC,init}}$",
        "tCC,slow":r"$t_{\mathrm{CC,slow}}$", "UCC,slow,init":r"$U_{\mathrm{CC,slow,init}}$",
        "UCC,slow,fin":r"$U_{\mathrm{CC,slow,fin}}$", "tCC,slow,init":r"$t_{\mathrm{CC,slow,init}}$",
        "tCC,slow,fin":r"$t_{\mathrm{CC,slow,fin}}$", "tCV":r"$t_{\mathrm{CV}}$",
        "dI/dtCV,max":r"$\mathrm{d}I/\mathrm{d}t_{\mathrm{CV,max}}$",
        "dI/dtCV,min":r"$\mathrm{d}I/\mathrm{d}t_{\mathrm{CV,min}}$",
        "tCV,0.5V":r"$t_{\mathrm{CV,0.5V}}$", "tCV,2.5~0.5V":r"$t_{\mathrm{CV,2.5\sim0.5V}}$"}
    for yi,row in zip(y,corr.itertuples(index=False)):
        c=colors[row.stage]; ax.hlines(yi,0,row.r,color=c,lw=3.0)
        ax.scatter(row.r,yi,color=c,s=42,zorder=3,edgecolor="white",linewidth=.4)
        if row.r <= -0.75:
            tx, ha = row.r + 0.025, "left"
        elif row.r < 0:
            tx, ha = row.r - 0.025, "right"
        elif row.r >= 0.75:
            tx, ha = row.r - 0.025, "right"
        else:
            tx, ha = row.r + 0.025, "left"
        ax.text(tx, yi, f"{row.r:+.2f}", va="center", ha=ha, fontsize=TEXT_SIZE,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.25})
    ax.axvline(0,color="#555555",lw=1.0); ax.axhline(12.5,color=GRID,lw=1.0); ax.axhline(4.5,color=GRID,lw=1.0)
    ax.set_xlim(-1.08,1.08); ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks(y); ax.set_yticklabels([labels.get(f,f) for f in corr.feature])
    ax.set_xlabel("Correlation coefficient")
    ax.xaxis.set_label_coords(0.5, -0.09)
    handles=[plt.Line2D([0],[0],color=c,lw=3,label=s) for s,c in colors.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.19),
              ncol=3, frameon=False)
    ax.text(-0.15, 1.035, "c", transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight="bold",
            ha="left", va="top", clip_on=False)
    style_axes(ax)
    save_panel(fig,OUT_DISCOVERY,"c_descriptor_correlations")


def build_composite_preview() -> None:
    paths = [
        OUT_DISCOVERY / "a_candidate_search_trajectory.png",
        OUT_DISCOVERY / "b_cddi_distribution.png",
        OUT_DISCOVERY / "c_descriptor_correlations.png",
        OUT_COMPARISON / "a_capacity_variance.png",
        OUT_COMPARISON / "b_capacity_npwv.png",
        OUT_COMPARISON / "c_33feature_mean_npwv.png",
        OUT_COMPARISON / "d_median_paired_improvement.png",
        OUT_COMPARISON / "e_clustering_stability.png",
        OUT_COMPARISON / "f_win_tie_loss.png",
    ]
    tile = 1200
    canvas = Image.new("RGB", (tile * 3, tile * 3), "white")
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            panel = source.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
        canvas.paste(panel, ((index % 3) * tile, (index // 3) * tile))
    canvas.save(OUT / "Revised_Figure_3x3_preview.png", dpi=(300, 300))


def main() -> None:
    apply_style()
    metrics=pd.read_csv(COMPARISON_DATA/"metrics_long.csv")
    paired=pd.read_csv(COMPARISON_DATA/"paired_battwise_tests.csv")
    friedman=pd.read_csv(COMPARISON_DATA/"friedman_tests.csv")
    stability=pd.read_csv(COMPARISON_DATA/"clustering_stability.csv")
    for metric,label,stem in zip(PRIMARY_METRICS,"abc",[
        "a_capacity_variance","b_capacity_npwv","c_33feature_mean_npwv"]):
        comparison_distribution(metrics,friedman,metric,label,stem)
    comparison_improvement(paired); comparison_stability(stability); comparison_win_loss(paired)
    discovery_search(); discovery_distribution(); discovery_correlations()
    build_composite_preview()
    print(f"Saved standalone panels under: {OUT}")


if __name__ == "__main__":
    main()
