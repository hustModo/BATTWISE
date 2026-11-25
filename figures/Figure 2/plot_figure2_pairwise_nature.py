"""Create Nature-inspired, separately saved plots for Figure 2.

The scientific data-processing and pair-selection logic is preserved from the
original ``plot_figure2_pairwise.py``.  The main change is presentation:
individual data plots are exported to a dedicated folder instead of being
assembled into one large multi-panel figure.

Outputs (PNG + PDF):
    Figure2_separate/
        Fig2a_capacity_ir_ocv.*
        Fig2b_charging_current.*
        Fig2c_charging_voltage.*
        Fig2d_ocv_distance_capacity_difference.*
        Fig2e_ir_distance_capacity_difference.*
        Fig2f_selected_pair_voltage_curves.*
        selected_pairs_pairwise.csv

Notes on style:
- Arial is requested as the primary font, with metrically similar sans-serif
  fallbacks for systems where Arial is unavailable.
- Typography is enlarged relative to the earlier publication version to improve
  readability in standalone panels and manuscript layouts.
- Top and right borders are explicitly enabled for every axes.
- Background gridlines are removed.
- The rainbow colormap is replaced with perceptually uniform / colour-safe
  choices.
- Line widths, tick sizes and text sizes are restrained for publication use.
- PDF is saved alongside 600-dpi PNG so line art and text remain editable.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# This script is self-contained: data are loaded directly from the four Excel
# files placed in the same folder as this script.  No external ``plot_figure2``
# module is required.


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "Figure2_separate"

# -----------------------------------------------------------------------------
# User-facing switches
# -----------------------------------------------------------------------------
SAVE_PNG = True
SAVE_PDF = True
SHOW_TITLES = True
ANNOTATE_SELECTED_PAIRS = False
MARK_ALL_SELECTION_FAMILIES = False

# Nature-like sizing: ~89 mm single-column and ~183 mm double-column widths.
SINGLE_COLUMN_WIDTH_IN = 3.50
DOUBLE_COLUMN_WIDTH_IN = 7.20

# Colour-blind-friendly, restrained palette.
BLUE = "#0072B2"
ORANGE = "#D55E00"
DARK_GRAY = "#4D4D4D"
MID_GRAY = "#7A7A7A"
LIGHT_GRAY = "#B7BDC3"


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def _require_file(path: Path) -> None:
    """Raise a clear error if a required input file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required data file not found: {path.name}\n"
            f"Please place it in the same folder as this script: {HERE}"
        )


def _clean_curve(series: pd.Series) -> np.ndarray:
    """Convert one Excel column to a finite 1-D float array."""
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return values[np.isfinite(values)]


def load_data() -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Load statistics, charging voltage and charging current from Excel files.

    The four source workbooks are expected beside this script:

    - ``Statis data.xlsx``
    - ``CCCV (2~32).xlsx``
    - ``CCCV (63~84).xlsx``
    - ``CCCV (85~104).xlsx``

    The statistics workbook contains 71 rows.  Its original ``Cell_num`` column
    is a sequential row index (1--71), not the physical cell identifier used in
    the CCCV workbooks.  The physical IDs are therefore restored from the CCCV
    column layout: 3--31, 63--84 and 85--104.  This mapping reproduces the cell
    labels used in the original Figure 2 (for example 17--74 and 85--104).
    """
    stat_path = HERE / "Statis data.xlsx"
    cccv_2_32_path = HERE / "CCCV (2~32).xlsx"
    cccv_63_84_path = HERE / "CCCV (63~84).xlsx"
    cccv_85_104_path = HERE / "CCCV (85~104).xlsx"

    for path in (stat_path, cccv_2_32_path, cccv_63_84_path, cccv_85_104_path):
        _require_file(path)

    # Statistical descriptors used by panels a, d and e.
    stat = pd.read_excel(stat_path, sheet_name=0)
    required_cols = {"Cell_num", "OCV", "IR", "Capacity"}
    missing_cols = required_cols.difference(stat.columns)
    if missing_cols:
        raise ValueError(
            "Statis data.xlsx is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )
    stat = stat[["Cell_num", "OCV", "IR", "Capacity"]].copy()
    for col in ("OCV", "IR", "Capacity"):
        stat[col] = pd.to_numeric(stat[col], errors="coerce")
    stat = stat.dropna(subset=["OCV", "IR", "Capacity"]).reset_index(drop=True)

    # The 71 statistical rows correspond, in order, to the 71 CCCV columns.
    cell_ids = list(range(3, 32)) + list(range(63, 85)) + list(range(85, 105))
    if len(stat) != len(cell_ids):
        raise ValueError(
            f"Expected {len(cell_ids)} statistical rows, but found {len(stat)}. "
            "Please check Statis data.xlsx before plotting."
        )
    stat["Cell_num"] = cell_ids

    voltage: dict[int, np.ndarray] = {}
    current: dict[int, np.ndarray] = {}

    # CCCV (2~32): 29 columns, corresponding to physical cells 3--31.
    vol_1 = pd.read_excel(cccv_2_32_path, sheet_name=0, header=None)
    cur_1 = pd.read_excel(cccv_2_32_path, sheet_name=1, header=None)
    ids_1 = list(range(3, 32))
    if vol_1.shape[1] != len(ids_1) or cur_1.shape[1] != len(ids_1):
        raise ValueError(
            "Unexpected column count in CCCV (2~32).xlsx: "
            f"Vol={vol_1.shape[1]}, Cur={cur_1.shape[1]}, expected {len(ids_1)}."
        )
    for col_idx, cell_id in enumerate(ids_1):
        voltage[cell_id] = _clean_curve(vol_1.iloc[:, col_idx])
        current[cell_id] = _clean_curve(cur_1.iloc[:, col_idx])

    # CCCV (63~84): the first row of the Vol sheet contains cell IDs; the Cur
    # sheet starts directly with current data.
    vol_2_raw = pd.read_excel(cccv_63_84_path, sheet_name="Vol", header=None)
    cur_2 = pd.read_excel(cccv_63_84_path, sheet_name="Cur", header=None)
    ids_2 = [int(x) for x in pd.to_numeric(vol_2_raw.iloc[0], errors="coerce").dropna()]
    if ids_2 != list(range(63, 85)):
        raise ValueError(
            "Unexpected cell-ID header in CCCV (63~84).xlsx / Vol. "
            f"Read IDs: {ids_2}"
        )
    vol_2 = vol_2_raw.iloc[1:].reset_index(drop=True)
    if vol_2.shape[1] != len(ids_2) or cur_2.shape[1] != len(ids_2):
        raise ValueError(
            "Unexpected column count in CCCV (63~84).xlsx: "
            f"Vol={vol_2.shape[1]}, Cur={cur_2.shape[1]}, expected {len(ids_2)}."
        )
    for col_idx, cell_id in enumerate(ids_2):
        voltage[cell_id] = _clean_curve(vol_2.iloc[:, col_idx])
        current[cell_id] = _clean_curve(cur_2.iloc[:, col_idx])

    # CCCV (85~104): 20 data columns without a header row.
    vol_3 = pd.read_excel(cccv_85_104_path, sheet_name="Vol", header=None)
    cur_3 = pd.read_excel(cccv_85_104_path, sheet_name="Cur", header=None)
    ids_3 = list(range(85, 105))
    if vol_3.shape[1] != len(ids_3) or cur_3.shape[1] != len(ids_3):
        raise ValueError(
            "Unexpected column count in CCCV (85~104).xlsx: "
            f"Vol={vol_3.shape[1]}, Cur={cur_3.shape[1]}, expected {len(ids_3)}."
        )
    for col_idx, cell_id in enumerate(ids_3):
        voltage[cell_id] = _clean_curve(vol_3.iloc[:, col_idx])
        current[cell_id] = _clean_curve(cur_3.iloc[:, col_idx])

    # Fail early with a readable message rather than a later KeyError in plotting.
    expected = set(cell_ids)
    missing_voltage = sorted(expected.difference(voltage))
    missing_current = sorted(expected.difference(current))
    if missing_voltage or missing_current:
        raise ValueError(
            f"Missing CCCV curves. Voltage: {missing_voltage}; Current: {missing_current}"
        )

    return stat, voltage, current


# -----------------------------------------------------------------------------
# Global style helpers
# -----------------------------------------------------------------------------
def apply_publication_style() -> None:
    """Apply a clean Nature-inspired style using Arial sans-serif typography."""
    plt.rcParams.update({
        # Arial / sans-serif typography.  Arial is preferred on Windows;
        # Liberation Sans and DejaVu Sans provide robust fallbacks elsewhere.
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Liberation Sans",
            "DejaVu Sans",
        ],
        "mathtext.fontset": "dejavusans",

        # Enlarged publication typography for improved readability.
        "font.size": 9.0,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 8.0,

        # Fine but printable line work.
        "axes.linewidth": 0.65,
        "lines.linewidth": 0.75,
        "xtick.major.width": 0.60,
        "ytick.major.width": 0.60,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.width": 0.50,
        "ytick.minor.width": 0.50,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,

        # White, uncluttered background.
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,

        # Keep text editable in vector outputs where possible.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })


def style_axes(ax: plt.Axes) -> None:
    """Enforce full rectangular borders and restrained tick styling."""
    for side in ("left", "bottom", "top", "right"):
        spine = ax.spines[side]
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_color("black")

    # Keep the requested top/right border without adding redundant top/right
    # ticks, which keeps dense scientific panels visually quieter.
    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        top=False,
        right=False,
        pad=2.0,
        colors="black",
    )
    ax.grid(False)


def save_figure(fig: plt.Figure, stem: str, output_dir: Path = OUTPUT_DIR) -> None:
    """Save one figure as publication-quality PNG and editable PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_PNG:
        fig.savefig(
            output_dir / f"{stem}.png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    if SAVE_PDF:
        fig.savefig(
            output_dir / f"{stem}.pdf",
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(fig)


# -----------------------------------------------------------------------------
# Pairwise calculations: unchanged scientific logic
# -----------------------------------------------------------------------------
def _all_pairs(stat: pd.DataFrame) -> pd.DataFrame:
    """Return all pairwise OCV/IR distances and capacity differences."""
    rows = []
    for i, j in itertools.combinations(range(len(stat)), 2):
        a, b = stat.iloc[i], stat.iloc[j]
        rows.append({
            "i": i,
            "j": j,
            "ocv_distance": abs(float(a.OCV) - float(b.OCV)),
            "ir_distance": abs(float(a.IR) - float(b.IR)),
            "capacity_difference": abs(float(a.Capacity) - float(b.Capacity)),
        })
    return pd.DataFrame(rows)


def _select_extreme_pairs(
    pairs: pd.DataFrame,
    distance_col: str,
    n: int,
    quantile: float = 0.20,
    mode: str = "both",
) -> pd.DataFrame:
    """Select upper-left, lower-right, or both scatter extremes."""
    n_close = n // 2
    n_far = n - n_close
    close_cutoff = pairs[distance_col].quantile(quantile)
    far_cutoff = pairs[distance_col].quantile(1 - quantile)

    def greedy(cand: pd.DataFrame, count: int, unique: bool = True) -> list[dict]:
        if not unique:
            return [r._asdict() for r in cand.head(count).itertuples(index=False)]

        selected, used = [], set()
        for r in cand.itertuples(index=False):
            if r.i not in used and r.j not in used:
                selected.append(r._asdict())
                used.update([r.i, r.j])
                if len(selected) == count:
                    break

        if len(selected) < count:
            existing = {(r["i"], r["j"]) for r in selected}
            for r in cand.itertuples(index=False):
                key = (r.i, r.j)
                if key not in existing:
                    selected.append(r._asdict())
                    existing.add(key)
                    if len(selected) == count:
                        break
        return selected

    close = pairs[pairs[distance_col] <= close_cutoff].sort_values(
        ["capacity_difference", distance_col], ascending=[False, True]
    )
    far = pairs[pairs[distance_col] >= far_cutoff].sort_values(
        ["capacity_difference", distance_col], ascending=[True, False]
    )

    if mode == "far":
        return pd.DataFrame(greedy(far, n, unique=True))
    if mode == "close":
        return pd.DataFrame(greedy(close, n, unique=True))
    return pd.DataFrame(
        greedy(close, n_close, unique=True)
        + greedy(far, n_far, unique=False)
    )


def select_pairs(stat: pd.DataFrame) -> pd.DataFrame:
    """Select 12 OCV-based and 13 IR-based pairs (25 total)."""
    pairs = _all_pairs(stat)

    ir_extreme = _select_extreme_pairs(pairs, "ir_distance", 13)
    ir_close = ir_extreme.iloc[:6].assign(selection="IR-close / high-gap")
    ir_far = ir_extreme.iloc[6:].assign(selection="IR-far / low-gap")
    ir = pd.concat([ir_close, ir_far], ignore_index=True)

    used_by_ir = {(int(r.i), int(r.j)) for r in ir.itertuples(index=False)}
    ocv_low = pairs.ocv_distance.quantile(0.20)
    ocv_high = pairs.ocv_distance.quantile(0.80)

    far_cand = pairs[pairs.ocv_distance >= ocv_high].sort_values(
        ["capacity_difference", "ocv_distance"], ascending=[True, False]
    )
    close_cand = pairs[pairs.ocv_distance <= ocv_low].sort_values(
        ["capacity_difference", "ocv_distance"], ascending=[False, True]
    )

    ocv_far_rows, used_by_ocv = [], set(used_by_ir)
    for r in far_cand.itertuples(index=False):
        key = (int(r.i), int(r.j))
        if key not in used_by_ocv:
            ocv_far_rows.append(r._asdict())
            used_by_ocv.add(key)
            if len(ocv_far_rows) == 6:
                break

    ocv_rows = [(r, "OCV-far / low-gap") for r in ocv_far_rows]
    for r in close_cand.itertuples(index=False):
        if len(ocv_rows) == 12:
            break
        key = (int(r.i), int(r.j))
        if key not in used_by_ocv:
            ocv_rows.append((r._asdict(), "OCV-close / high-gap"))
            used_by_ocv.add(key)

    ocv = pd.DataFrame([dict(r, selection=cat) for r, cat in ocv_rows])
    selected = pd.concat([ocv, ir], ignore_index=True)

    out = []
    for k, r in enumerate(selected.itertuples(index=False), 1):
        a, b = stat.iloc[r.i], stat.iloc[r.j]
        out.append({
            "plot_order": k,
            "selection": r.selection,
            "cell_a": int(a.Cell_num),
            "cell_b": int(b.Cell_num),
            "ocv_distance": r.ocv_distance,
            "ir_distance": r.ir_distance,
            "capacity_difference": r.capacity_difference,
        })
    return pd.DataFrame(out)


# -----------------------------------------------------------------------------
# Plot builders
# -----------------------------------------------------------------------------
def plot_capacity_ir_ocv(stat: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH_IN, 3.05))

    capacity = stat.Capacity.to_numpy(float)
    norm = Normalize(vmin=capacity.min(), vmax=capacity.max())
    cmap = plt.get_cmap("viridis")

    # Redundant size + colour encoding improves readability in grayscale and
    # for readers with colour-vision deficiency.
    span = max(capacity.max() - capacity.min(), np.finfo(float).eps)
    sizes = 18 + 18 * (capacity - capacity.min()) / span

    sc = ax.scatter(
        stat.OCV,
        stat.IR,
        c=capacity,
        s=sizes,
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.92,
        zorder=2,
    )

    ax.set_xlabel("OCV (V)")
    ax.set_ylabel("IR (mΩ)")
    if SHOW_TITLES:
        ax.set_title("Capacity, IR and OCV distributions", pad=4, fontweight="normal")
    style_axes(ax)

    cb = fig.colorbar(sc, ax=ax, pad=0.025, fraction=0.050)
    cb.set_label("Capacity (Ah)", labelpad=3)
    cb.ax.tick_params(labelsize=8.0, width=0.55, length=2.5, direction="out")
    cb.outline.set_linewidth(0.60)

    fig.tight_layout(pad=0.5)
    return fig


def plot_all_curves(
    curves: dict[int, np.ndarray],
    ids: np.ndarray,
    ylabel: str,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH_IN, 2.85))

    # Use a perceptually uniform colormap rather than rainbow/jet.
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=max(len(ids) - 1, 1))

    for k, bid in enumerate(ids):
        y = curves[int(bid)]
        if len(y) < 2:
            continue
        ax.plot(
            np.linspace(0, 100, len(y)),
            y,
            color=cmap(norm(k)),
            lw=0.55,
            alpha=0.55,
            solid_capstyle="round",
        )

    ax.set_xlabel("Charge progress (%)")
    ax.set_ylabel(ylabel)
    if SHOW_TITLES:
        ax.set_title(title, pad=4, fontweight="normal")
    style_axes(ax)
    ax.margins(x=0)

    fig.tight_layout(pad=0.5)
    return fig


def plot_pair_scatter(
    pairs: pd.DataFrame,
    selected: pd.DataFrame,
    xcol: str,
    title: str,
    show_all: bool = False,
    annotate: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH_IN, 2.95))

    ax.scatter(
        pairs[xcol],
        pairs.capacity_difference,
        s=9,
        alpha=0.22,
        color=MID_GRAY,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    ocv_sel = selected[selected.selection.str.startswith("OCV")]
    ir_sel = selected[selected.selection.str.startswith("IR")]

    if show_all:
        ax.scatter(
            ocv_sel[xcol],
            ocv_sel.capacity_difference,
            s=30,
            color=ORANGE,
            marker="o",
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
            label="OCV-selected",
        )
        ax.scatter(
            ir_sel[xcol],
            ir_sel.capacity_difference,
            s=31,
            color=BLUE,
            marker="s",
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
            label="IR-selected",
        )
        marked = pd.concat([ocv_sel, ir_sel], ignore_index=True)
    else:
        metric = "OCV" if xcol == "ocv_distance" else "IR"
        marked = selected[selected.selection.str.startswith(metric)]
        highlight_color = ORANGE if metric == "OCV" else BLUE
        ax.scatter(
            marked[xcol],
            marked.capacity_difference,
            s=31,
            color=highlight_color,
            marker="o",
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
            label="Selected pairs",
        )

    if annotate:
        for r in marked.itertuples(index=False):
            ax.annotate(
                f"{r.cell_a}–{r.cell_b}",
                (getattr(r, xcol), r.capacity_difference),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7.0,
                color=DARK_GRAY,
            )

    ax.set_xlabel("OCV distance (V)" if xcol == "ocv_distance" else "IR distance (mΩ)")
    ax.set_ylabel("Capacity difference (Ah)")
    if SHOW_TITLES:
        ax.set_title(title, pad=4, fontweight="normal")
    style_axes(ax)

    leg = ax.legend(loc="upper left", frameon=False, handletextpad=0.4, borderaxespad=0.2)
    for text in leg.get_texts():
        text.set_color("black")

    fig.tight_layout(pad=0.5)
    return fig


def plot_selected_pair_grid(
    voltage: dict[int, np.ndarray],
    selected: pd.DataFrame,
) -> plt.Figure:
    """Create the 5x5 summary grid corresponding to the original panel f."""
    fig, axes = plt.subplots(
        5,
        5,
        figsize=(DOUBLE_COLUMN_WIDTH_IN, 5.65),
        sharex=True,
        sharey=False,
    )

    for k, ax in enumerate(axes.ravel()):
        r = selected.iloc[k]
        a_id = int(r.cell_a)
        b_id = int(r.cell_b)

        for bid, color in ((a_id, BLUE), (b_id, ORANGE)):
            y = voltage[bid]
            ax.plot(
                np.linspace(0, 100, len(y)),
                y,
                lw=0.70,
                color=color,
                solid_capstyle="round",
            )

        kind = "OCV" if r.selection.startswith("OCV") else "IR"
        ax.set_title(
            f"{a_id}–{b_id} | {kind}\nΔQ={r.capacity_difference:.2f} Ah",
            fontsize=6.4,
            pad=2.4,
            fontweight="normal",
        )
        style_axes(ax)
        ax.tick_params(labelsize=6.0, width=0.5, length=2.0, pad=1.5)

        row, col = divmod(k, 5)
        if row == 4:
            ax.set_xlabel("Charge progress (%)", fontsize=6.5, labelpad=1.7)
        else:
            ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel("Voltage (V)", fontsize=6.5, labelpad=1.7)
        else:
            ax.tick_params(labelleft=False)

    legend_handles = [
        Line2D([0], [0], color=BLUE, lw=1.2, label="Cell A"),
        Line2D([0], [0], color=ORANGE, lw=1.2, label="Cell B"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.998),
        ncol=2,
        frameon=False,
        fontsize=8.0,
        handlelength=2.4,
        columnspacing=1.8,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.995,
        bottom=0.075,
        top=0.93,
        wspace=0.28,
        hspace=0.55,
    )
    return fig



# -----------------------------------------------------------------------------
# Main export workflow
# -----------------------------------------------------------------------------
def main() -> None:
    apply_publication_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stat, voltage, current = load_data()
    pairs = _all_pairs(stat)
    selected = select_pairs(stat)

    # Keep the exact pair list used by panels d/e/f for traceability.
    selected.to_csv(
        OUTPUT_DIR / "selected_pairs_pairwise.csv",
        index=False,
        float_format="%.8g",
    )

    ids = stat.Cell_num.to_numpy(int)

    save_figure(
        plot_capacity_ir_ocv(stat),
        "Fig2a_capacity_ir_ocv",
    )
    save_figure(
        plot_all_curves(current, ids, "Current (A)", "Charging current"),
        "Fig2b_charging_current",
    )
    save_figure(
        plot_all_curves(voltage, ids, "Voltage (V)", "Charging voltage"),
        "Fig2c_charging_voltage",
    )
    save_figure(
        plot_pair_scatter(
            pairs,
            selected,
            "ocv_distance",
            "OCV distance vs. capacity difference",
            show_all=MARK_ALL_SELECTION_FAMILIES,
            annotate=ANNOTATE_SELECTED_PAIRS,
        ),
        "Fig2d_ocv_distance_capacity_difference",
    )
    save_figure(
        plot_pair_scatter(
            pairs,
            selected,
            "ir_distance",
            "IR distance vs. capacity difference",
            show_all=MARK_ALL_SELECTION_FAMILIES,
            annotate=ANNOTATE_SELECTED_PAIRS,
        ),
        "Fig2e_ir_distance_capacity_difference",
    )
    save_figure(
        plot_selected_pair_grid(voltage, selected),
        "Fig2f_selected_pair_voltage_curves",
    )

    print(f"Saved separate Figure 2 plots to: {OUTPUT_DIR}")
    print("Saved formats: " + ", ".join(
        fmt for fmt, enabled in (("PNG (600 dpi)", SAVE_PNG), ("PDF", SAVE_PDF)) if enabled
    ))


if __name__ == "__main__":
    main()
