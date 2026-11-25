# BATTWISE Figure Code and Data Guide

This directory organizes the plotting scripts and associated data included in the current submission by figure number. The scripts read the saved data and generate figures; unless otherwise noted, they do not retrain BATTWISE or rerun clustering.

## Directory Contents

### Figure 2: Heterogeneity in Retired-Battery Charging Data

- `plot_figure2_pairwise.py`: Generates the Figure 2 multipanel figure and includes functions for plotting panels a–f individually.
- `Figure2_separate/plot_a.py` through `plot_f.py`: Generate the individual panels. See `README.md` in that directory for instructions on producing individual-panel outputs.
- `Statis data.xlsx`: Statistical data, including battery capacity, internal resistance, and OCV.
- `CCCV (1~29).xlsx`, `CCCV (30~51).xlsx`, and `CCCV (52~71).xlsx`: Charging-curve data.

### Figure 3: CDDI Discovery and Method Comparison

- `plot_figure3_comparison.py`: Generates the comparison figure for BATTWISE, KPCA, FC-VAE, DECM, and OCV–IR using results from 30 paired random seeds. This script depends on the shared `style.py`, which is not included in the current submission directory.
- `Figure 3 new/data/`: Per-seed method-comparison metrics, paired statistics, Friedman test results, clustering-stability results, and electrochemical-feature summaries.
- `plot_figure3_cddi_discovery.py`: Generates the candidate-search trajectory, the final CDDI distribution across 71 cells, and correlations between CDDI and 17 charging descriptors. This script depends on the project-level shared style module `05_FIGURES/code/paper_figures`, which is not included in the current submission directory.
- `Figure S3/data/`: Search records, frozen CDDI values, correlations, charging descriptors, candidate programs, and cell-curve data used for the CDDI-discovery figure.
- `plot_individual_panels.py`: Generates the individual panels for the Figure 3 comparison and Figure S3 discovery figures, as well as a 3 × 3 preview.
- `methods/`: Implementations, configurations, and descriptions of the one-dimensional representation methods used in the fair comparison.

The comparison figure uses the saved results from 30 paired runs; it does not retrain representation models, rerun clustering, or recalculate statistical tests. `plot_individual_panels.py` uses local plotting styles and reads data from the adjacent `Figure 3 new/data/` and `Figure S3/data/` directories. It is the entry point in the current submission directory for reproducing the nine individual panels and the 3 × 3 preview.

### Figure 4: Stability with Incomplete Data

- `plot_from_revised_csv.py`: Reads `plot_data/revised_45_models_all_panels.csv` and generates six result panels for three charging windows: First 100 s, First 200 s, and First 500 s.

### Figure 5: Battery-Pack Statistics and Cycling Data

- `data/table0_variance.csv`: Metric-dispersion data for different grouping schemes.
- `data/table1_capacity_summary.csv`: Capacity statistics for the reassembled battery packs.
- `data/table2_electrochemical_summary.csv`: Electrochemical-feature statistics for the reassembled battery packs.
- `data/table3_cycle_capacity.csv`: Per-cycle capacity data for the battery packs.
