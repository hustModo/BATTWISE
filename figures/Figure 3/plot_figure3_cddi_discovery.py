"""Plot Figure 3: BATTWISE search, frozen CDDI and descriptor associations.

The script is self-contained: all evidence files are copied into ``data/``.
Run from any directory with ``python plot_figure3_cddi_discovery.py``.
Outputs ``Figure3_CDDI_discovery.pdf`` and ``Figure3_CDDI_discovery.png``.
"""

from __future__ import annotations

import sys
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse the canonical Figure 2 style so that this discovery figure matches the
# formal Results figures (Arial-compatible sans serif, panel labels, colours,
# line weights and export settings).
STYLE_ROOT = Path(__file__).resolve().parents[1] / "05_FIGURES" / "code"
if str(STYLE_ROOT) not in sys.path:
    sys.path.insert(0, str(STYLE_ROOT))
from paper_figures.style import GRID, NEGATIVE, POSITIVE, TEXT, apply_style, panel_label, save_figure


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT_PDF = HERE / "Figure3_CDDI_discovery.pdf"
OUT_PNG = HERE / "Figure3_CDDI_discovery.png"
PROGRAM_CANDIDATES = [
    DATA / "calculate_mmd_v2.py",
    HERE.parent / "02_BATTWISE_METHOD" / "selected_programs" / "global_all200" / "calculate_mmd_v2.py",
]

ORANGE = "#D55E00"
BLUE = "#0072B2"
TEAL = POSITIVE
YELLOW = "#E69F00"
GREY = "#777777"


def _search_trace() -> pd.DataFrame:
    raw = pd.read_csv(DATA / "candidate_records.csv")
    raw["objective_num"] = pd.to_numeric(raw["objective"], errors="coerce")
    rows = []
    for generation, group in raw.groupby("generation", sort=True):
        finite = group["objective_num"].dropna()
        score = -finite
        rows.append(
            {
                "generation": int(generation),
                "population_size": int(len(group)),
                "finite_count": int(len(finite)),
                "best": float(score.max()) if len(score) else np.nan,
                "median": float(score.median()) if len(score) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _frozen_cddi() -> pd.DataFrame:
    emb = pd.read_csv(DATA / "embeddings.csv")
    out = emb.loc[(emb["method"] == "BATTWISE-CDDI") & (emb["seed"] == 0), ["Cell_num", "z1"]].copy()
    out["z1"] = pd.to_numeric(out["z1"], errors="coerce")
    return out.dropna().sort_values("z1").reset_index(drop=True)


def _correlations() -> pd.DataFrame:
    desc = pd.read_excel(DATA / "Charging_curve_characteristics.xlsx", sheet_name="Sheet2")
    emb = _frozen_cddi()
    merged = desc.merge(emb, on="Cell_num", validate="one_to_one")
    if len(merged) != 71:
        raise ValueError(f"Expected all 71 cells for descriptor correlations, found {len(merged)}")
    features = [c for c in desc.columns if c != "Cell_num"]
    rows = []
    for feature in features:
        rows.append({"feature": feature, "r": float(merged[feature].corr(merged["z1"]))})
    out = pd.DataFrame(rows)

    # The 17 descriptors are displayed as three contiguous blocks, matching
    # the charging-process stages requested for panel c.
    def stage(feature: str) -> str:
        index = features.index(feature)
        if index < 4:
            return "Early charge (1–4)"
        if index < 12:
            return "CC (5–12)"
        return "CV (13–17)"

    out["stage"] = out["feature"].map(stage)
    stage_order = {"Early charge (1–4)": 0, "CC (5–12)": 1, "CV (13–17)": 2}
    out["stage_order"] = out["stage"].map(stage_order)
    feature_order = {feature: i for i, feature in enumerate(features)}
    out["feature_order"] = out["feature"].map(feature_order)
    return out.sort_values("feature_order").reset_index(drop=True)


def build_figure() -> plt.Figure:
    trace = _search_trace()
    cddi = _frozen_cddi()
    corr = _correlations()

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.95), constrained_layout=True)
    ax_d, ax_e, ax_f = axes

    # a: search trajectory
    ax_d.plot(trace["generation"], trace["best"], color=ORANGE, marker="o", ms=3.2, lw=1.8, label="Best feasible")
    ax_d.plot(trace["generation"], trace["median"], color=GREY, marker="s", ms=2.8, lw=1.25, ls="--", label="Population median")
    ax_d.fill_between(trace["generation"], trace["median"], trace["best"], color=ORANGE, alpha=0.10, linewidth=0)
    ax_d.set_xlabel("Generation")
    ax_d.set_ylabel("Mean |Pearson r|")
    ax_d.set_title("Candidate search trajectory")
    ax_d.set_xticks([0, 5, 10, 15, 20])
    # Keep the narrow objective trajectory visually centred rather than
    # compressing it against the top of a zero-based axis.
    lo = float(np.nanmin(pd.concat([trace["best"], trace["median"]])))
    hi = float(np.nanmax(pd.concat([trace["best"], trace["median"]])))
    pad = max(0.02, 0.18 * (hi - lo))
    ax_d.set_ylim(lo - pad, hi + pad)
    ax_d.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax_d.text(0.04, 0.96, f"{int(trace.finite_count.sum())}/{int(trace.population_size.sum())} finite objectives\n{len(trace)} generations", transform=ax_d.transAxes, va="top", fontsize=6.5, color=TEXT)
    ax_d.grid(axis="y", color=GRID, lw=0.5, alpha=0.7)
    panel_label(ax_d, "a")

    # b: sorted CDDI values with distribution summary.  The frozen program is
    # applied to all 71 cells; seed 0 is the reproducible frozen output used
    # for the formal figure (other seeds are retained in embeddings.csv).
    ranks = np.arange(1, len(cddi) + 1)
    ax_e.scatter(ranks, cddi["z1"], s=17, color=TEAL, alpha=0.82, edgecolor="white", linewidth=0.25)
    ax_e.plot(ranks, cddi["z1"], color=TEAL, alpha=0.28, lw=0.9)
    q1, med, q3 = cddi["z1"].quantile([0.25, 0.5, 0.75])
    ax_e.axhspan(q1, q3, color=TEAL, alpha=0.08, zorder=0)
    ax_e.axhline(med, color=TEAL, lw=1.1, ls="--")
    ax_e.set_xlabel("Cell rank (sorted by CDDI)")
    ax_e.set_ylabel("CDDI")
    ax_e.set_title("CDDI distribution")
    ax_e.text(0.03, 0.96, f"n = {len(cddi)}\nmedian = {med:,.0f}\nIQR = {q1:,.0f}–{q3:,.0f}", transform=ax_e.transAxes, va="top", fontsize=6.5, color=TEXT)
    ax_e.grid(axis="y", color=GRID, lw=0.5, alpha=0.7)
    panel_label(ax_e, "b")

    # c: descriptor correlation lollipop, computed across all 71 cells
    y = np.arange(len(corr))[::-1]
    colours = {"Early charge (1–4)": BLUE, "CC (5–12)": YELLOW, "CV (13–17)": ORANGE}
    feature_labels = {
        "tChar": r"$t_{\mathrm{Char}}$",
        "Urest": r"$U_{\mathrm{rest}}$",
        "UCC,init": r"$U_{\mathrm{CC,init}}$",
        "∆Urest,CC": r"$\Delta U_{\mathrm{rest,CC}}$",
        "tCC": r"$t_{\mathrm{CC}}$",
        "dU/dtCC,fin": r"$\mathrm{d}U/\mathrm{d}t_{\mathrm{CC,fin}}$",
        "dU/dtCC,init": r"$\mathrm{d}U/\mathrm{d}t_{\mathrm{CC,init}}$",
        "tCC,slow": r"$t_{\mathrm{CC,slow}}$",
        "UCC,slow,init": r"$U_{\mathrm{CC,slow,init}}$",
        "UCC,slow,fin": r"$U_{\mathrm{CC,slow,fin}}$",
        "tCC,slow,init": r"$t_{\mathrm{CC,slow,init}}$",
        "tCC,slow,fin": r"$t_{\mathrm{CC,slow,fin}}$",
        "tCV": r"$t_{\mathrm{CV}}$",
        "dI/dtCV,max": r"$\mathrm{d}I/\mathrm{d}t_{\mathrm{CV,max}}$",
        "dI/dtCV,min": r"$\mathrm{d}I/\mathrm{d}t_{\mathrm{CV,min}}$",
        "tCV,0.5V": r"$t_{\mathrm{CV,0.5V}}$",
        "tCV,2.5~0.5V": r"$t_{\mathrm{CV,2.5\sim0.5V}}$",
    }
    for yi, row in zip(y, corr.itertuples(index=False)):
        colour = colours[row.stage]
        ax_f.hlines(yi, 0, row.r, color=colour, lw=2.2)
        ax_f.scatter(row.r, yi, color=colour, s=21, zorder=3)
        ax_f.text(row.r + (0.018 if row.r >= 0 else -0.018), yi, f"{row.r:+.2f}", va="center", ha="left" if row.r >= 0 else "right", fontsize=6.4, color=TEXT)
    ax_f.axvline(0, color="#70757A", lw=0.8)
    ax_f.set_xlim(-1.05, 1.05)
    ax_f.set_yticks(y)
    ax_f.set_yticklabels([feature_labels.get(feature, feature) for feature in corr["feature"]], fontsize=6.7)
    ax_f.set_xlabel("correlation coefficient")
    ax_f.set_title("All-200 CDDI vs 17 descriptors")
    # Keep both axes and all descriptor labels neutral; only the correlation
    # curves carry the three group colours.
    for tick in ax_f.get_yticklabels():
        tick.set_color(TEXT)
    # Neutral separators make the three contiguous descriptor blocks explicit.
    ax_f.axhline(12.5, color=GRID, lw=0.7)
    ax_f.axhline(4.5, color=GRID, lw=0.7)
    ax_f.grid(axis="y", color=GRID, lw=0.5, alpha=0.7)
    panel_label(ax_f, "c")
    return fig


def main() -> None:
    program = next((candidate for candidate in PROGRAM_CANDIDATES if candidate.exists()), None)
    if program is None:
        searched = ", ".join(str(path) for path in PROGRAM_CANDIDATES)
        raise FileNotFoundError(f"Candidate MMD program not found; searched: {searched}")
    # Record the exact candidate source used for the frozen formal outputs.
    digest = hashlib.sha256(program.read_bytes()).hexdigest()
    (DATA / "calculate_mmd_v2.sha256").write_text(digest + "  calculate_mmd_v2.py\n", encoding="utf-8")
    # Export the exact plot-ready tables used below for auditability.
    trace = _search_trace()
    cddi = _frozen_cddi()
    corr = _correlations()
    trace.to_csv(DATA / "search_trace_plot.csv", index=False)
    cddi.to_csv(DATA / "frozen_cddi_seed0_plot.csv", index=False)
    corr.to_csv(DATA / "descriptor_correlations_plot.csv", index=False)
    figure = build_figure()
    save_figure(figure, HERE, "Figure3_CDDI_discovery")
    plt.close(figure)
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
