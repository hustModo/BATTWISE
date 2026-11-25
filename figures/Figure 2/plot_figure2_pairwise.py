from __future__ import annotations

"""Revised Figure 2 pairwise overview and standalone panel exporters.

The data loading and pair-selection definitions are reused from the adjacent
``plot_figure2_pairwise_nature.py`` file in the destination folder.  All panel
positions are explicit; panel a is narrowed to accommodate its external
capacity color scale, while panel f is a borderless, evenly spaced 5 x 5 grid.
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_figure2_pairwise_nature import load_data, _all_pairs, select_pairs  # noqa: E402


OUT_DIR = HERE / "Figure2_separate"
FIGURE_NAME = "Figure2_pairwise_revised"

# Fixed normalized figure coordinates. The a-e plotting frames all use the
# same width and height. The f frame is aligned to b/c/d/e exactly.
FRAME_W = 0.255
FRAME_H = 0.245
FRAME_A_LEFT = 0.055
FRAME_A_W = 0.219
FRAME_B_LEFT = 0.355
FRAME_C_LEFT = 0.655
FRAME_TOP_BOTTOM = 0.710
FRAME_D_BOTTOM = 0.400
FRAME_E_BOTTOM = 0.090
FRAME_F_LEFT = FRAME_B_LEFT
FRAME_F_RIGHT = FRAME_C_LEFT + FRAME_W
FRAME_F_BOTTOM = FRAME_E_BOTTOM
FRAME_F_TOP = FRAME_D_BOTTOM + FRAME_H

BLUE = "#1f77b4"
RED = "#d62728"
GRAY = "#5f6670"
ORANGE = "#d55e00"


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 12.0,
        "axes.labelsize": 12.0,
        "xtick.labelsize": 10.7,
        "ytick.labelsize": 10.7,
        "legend.fontsize": 9.7,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.unicode_minus": False,
    })


def closed_frame(ax, linewidth=0.9, color="black") -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(linewidth)
    ax.tick_params(direction="out", top=False, right=False, length=3.0, width=0.7, pad=2)
    ax.grid(alpha=0.18, linewidth=0.45)


def panel_label(ax, label: str) -> None:
    ax.text(0.0, 1.045, label, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=13, fontweight="bold",
            clip_on=False)


def _curve_axes(ax, curves: dict[int, np.ndarray], ids: np.ndarray, ylabel: str) -> None:
    cmap = get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=max(len(ids) - 1, 1))
    for k, bid in enumerate(ids):
        y = curves[int(bid)]
        if len(y) >= 2:
            ax.plot(np.linspace(0, 100, len(y)), y, color=cmap(norm(k)), lw=0.98, alpha=0.60)
    ax.set_xlabel("Charge progress (%)")
    ax.set_ylabel(ylabel)
    ax.margins(x=0)
    closed_frame(ax)


def draw_a(ax, stat: pd.DataFrame) -> None:
    capacity = stat.Capacity.to_numpy(float)
    span = max(capacity.max() - capacity.min(), np.finfo(float).eps)
    sc = ax.scatter(stat.OCV, stat.IR, c=capacity,
                    s=22 + 18 * (capacity - capacity.min()) / span,
                    cmap="plasma", norm=Normalize(vmin=capacity.min(), vmax=capacity.max()),
                    edgecolor="white", linewidth=0.35, alpha=0.9)
    ax.set_xlabel("OCV (V)")
    ax.set_ylabel("IR (mΩ)")
    closed_frame(ax)
    # Keep the colorbar outside the black plotting frame.  In particular, it
    # must not cover the right-hand spine or its tick labels.
    pos = ax.get_position()
    cax = ax.figure.add_axes([
        pos.x1 + 0.008,
        pos.y0 + 0.025,
        0.012,
        max(pos.height - 0.050, 0.04),
    ])
    cb = ax.figure.colorbar(sc, cax=cax, orientation="vertical")
    # The color scale is self-evident from the Capacity (Ah) tick values;
    # omitting a long vertical label keeps it clear of panel b's y-axis.
    cb.ax.tick_params(labelsize=8.2, length=2.5, pad=2)
    cb.outline.set_linewidth(0.6)


def draw_scatter(ax, pairs: pd.DataFrame, selected: pd.DataFrame, xcol: str, annotate=False) -> None:
    ax.scatter(pairs[xcol], pairs.capacity_difference, s=12, alpha=0.20,
               color=GRAY, rasterized=True, linewidths=0)
    metric = "OCV" if xcol == "ocv_distance" else "IR"
    marked = selected[selected.selection.str.startswith(metric)]
    color = ORANGE if metric == "OCV" else BLUE
    ax.scatter(marked[xcol], marked.capacity_difference, s=48, color=color,
               edgecolor="white", linewidth=0.45, zorder=3, label="Selected pairs")
    if annotate:
        for r in marked.itertuples(index=False):
            ax.annotate(f"{r.cell_a}–{r.cell_b}", (getattr(r, xcol), r.capacity_difference),
                        xytext=(3, 3), textcoords="offset points", fontsize=5.8)
    ax.set_xlabel("OCV distance (V)" if xcol == "ocv_distance" else "IR distance (mΩ)")
    ax.set_ylabel("Capacity difference (Ah)")
    closed_frame(ax)
    ax.legend(loc="upper left", fontsize=9.5, frameon=False)


def _add_f_inner_axes(fig, voltage, selected):
    left, right = FRAME_F_LEFT, FRAME_F_RIGHT
    bottom, top = FRAME_F_BOTTOM, FRAME_F_TOP
    # Panel f is a regular 5 x 5 grid without an enclosing outer frame.  The
    # bottom row starts at the same normalized y-position as panel e, so its
    # x-axis labels share the same baseline.
    inner_left, inner_right = left + 0.010, right - 0.010
    inner_bottom = bottom
    # Reserve a title band above every row.  The first row title is anchored
    # exactly at panel d's top edge; subsequent rows retain a visible gap from
    # the row above rather than colliding with its lower spine.
    title_offset = 0.020
    row_gap = 0.012
    first_title_top = top
    cell_h = ((top - bottom) - 5 * title_offset - 4 * row_gap) / 5
    cell_w = (inner_right - inner_left - 4 * 0.014) / 5
    gap_x = 0.014
    for k in range(25):
        row, col = divmod(k, 5)
        title_top = first_title_top - row * (cell_h + title_offset + row_gap)
        axis_top = title_top - title_offset
        y = axis_top - cell_h
        x = inner_left + col * (cell_w + gap_x)
        ax = fig.add_axes([x, y, cell_w, cell_h], zorder=2)
        r = selected.iloc[k]
        for bid, color in ((int(r.cell_a), BLUE), (int(r.cell_b), RED)):
            yv = voltage[bid]
            ax.plot(np.linspace(0, 100, len(yv)), yv, lw=0.78, color=color)
        kind = "OCV" if r.selection.startswith("OCV") else "IR"
        fig.text(x + cell_w / 2, title_top,
                 f"{int(r.cell_a)}–{int(r.cell_b)} | {kind}\nΔQ={r.capacity_difference:.2f} Ah",
                 ha="center", va="top", fontsize=7.2, linespacing=0.90)
        ax.grid(alpha=0.12, linewidth=0.4)
        closed_frame(ax, linewidth=0.45, color="#8b8b8b")
        ax.tick_params(labelsize=6.2, length=1.5, width=0.5, pad=1.2)
        if row < 4: ax.tick_params(labelbottom=False)
        if col > 0: ax.tick_params(labelleft=False)
        if row == 4: ax.set_xlabel("%", fontsize=6.5, labelpad=0.6)
        if col == 0: ax.set_ylabel("V", fontsize=6.5, labelpad=0.6)
    fig.text(left, top + 0.010, "f", ha="left", va="bottom",
             fontsize=13, fontweight="bold")
    fig.legend(handles=[Line2D([0], [0], color=BLUE, lw=1.4, label="Cell A"),
                        Line2D([0], [0], color=RED, lw=1.4, label="Cell B")],
               loc="lower center", bbox_to_anchor=((left + right) / 2,
                                                    bottom - 0.030),
               ncol=2, fontsize=9.0, frameon=False, handlelength=1.9,
               borderaxespad=0.0, columnspacing=1.4)
    return None


def make_group_figure(stat, voltage, current, selected, pairs):
    fig = plt.figure(figsize=(16, 12), facecolor="white")
    positions = {
        "a": (FRAME_A_LEFT, FRAME_TOP_BOTTOM, FRAME_A_W, FRAME_H),
        "b": (FRAME_B_LEFT, FRAME_TOP_BOTTOM, FRAME_W, FRAME_H),
        "c": (FRAME_C_LEFT, FRAME_TOP_BOTTOM, FRAME_W, FRAME_H),
        "d": (FRAME_A_LEFT, FRAME_D_BOTTOM, FRAME_W, FRAME_H),
        "e": (FRAME_A_LEFT, FRAME_E_BOTTOM, FRAME_W, FRAME_H),
    }
    axa = fig.add_axes(positions["a"]); draw_a(axa, stat); panel_label(axa, "a")
    ids = stat.Cell_num.to_numpy(int)
    axb = fig.add_axes(positions["b"]); _curve_axes(axb, current, ids, "Current (A)"); panel_label(axb, "b")
    axc = fig.add_axes(positions["c"]); _curve_axes(axc, voltage, ids, "Voltage (V)"); panel_label(axc, "c")
    axd = fig.add_axes(positions["d"]); draw_scatter(axd, pairs, selected, "ocv_distance"); panel_label(axd, "d")
    axe = fig.add_axes(positions["e"]); draw_scatter(axe, pairs, selected, "ir_distance"); panel_label(axe, "e")
    _add_f_inner_axes(fig, voltage, selected)
    return fig


def _save(fig, stem: str, folder: Path = OUT_DIR):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.png", dpi=600, bbox_inches=None, pad_inches=0)
    fig.savefig(folder / f"{stem}.pdf", bbox_inches=None, pad_inches=0)
    fig.savefig(folder / f"{stem}.svg", bbox_inches=None, pad_inches=0)
    plt.close(fig)


def load_context():
    stat, voltage, current = load_data()
    pairs = _all_pairs(stat)
    selected = select_pairs(stat)
    return stat, voltage, current, pairs, selected


def save_individual_panel(panel: str):
    apply_style()
    stat, voltage, current, pairs, selected = load_context()
    ids = stat.Cell_num.to_numpy(int)
    if panel == "a":
        fig, ax = plt.subplots(figsize=(6, 6)); draw_a(ax, stat); panel_label(ax, "a")
    elif panel == "b":
        fig, ax = plt.subplots(figsize=(6, 6)); _curve_axes(ax, current, ids, "Current (A)"); panel_label(ax, "b")
    elif panel == "c":
        fig, ax = plt.subplots(figsize=(6, 6)); _curve_axes(ax, voltage, ids, "Voltage (V)"); panel_label(ax, "c")
    elif panel == "d":
        fig, ax = plt.subplots(figsize=(6, 6)); draw_scatter(ax, pairs, selected, "ocv_distance"); panel_label(ax, "d")
    elif panel == "e":
        fig, ax = plt.subplots(figsize=(6, 6)); draw_scatter(ax, pairs, selected, "ir_distance"); panel_label(ax, "e")
    elif panel == "f":
        fig = plt.figure(figsize=(8.4, 7.8)); _add_f_inner_axes(fig, voltage, selected)
    else:
        raise ValueError("panel must be one of a, b, c, d, e, f")
    _save(fig, f"Fig2{panel}_standalone")


def main():
    apply_style()
    stat, voltage, current, pairs, selected = load_context()
    selected.to_csv(HERE / "selected_pairs_pairwise.csv", index=False, float_format="%.8g")
    _save(make_group_figure(stat, voltage, current, selected, pairs), FIGURE_NAME, HERE)
    print(f"Saved {FIGURE_NAME}.png/.pdf/.svg and selected_pairs_pairwise.csv to {HERE}")


if __name__ == "__main__":
    main()
