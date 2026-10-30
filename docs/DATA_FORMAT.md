# Input data contract

Measured data are intentionally omitted from this code-only stage. This section
documents inputs; it does not assert that a complete research data deposit has
already been published.

## All-200 experiment loader

Expected directory: `battery_compute/AII_200_new/` relative to the repository root.
The loader accepts `.xls` or `.xlsx` for each required workbook stem.

| File | Required content |
|---|---|
| `Model_200.xlsx` or `.xls` | `Index`, `Current`, `Voltage`; 200 records |
| `Cell<ID>_200.xlsx` or `.xls` | Same three columns and reference-aligned index |
| `Statis data.xlsx` | Unique `Cell_num`, `OCV`, `IR`, `Capacity` |
| `Charging_curve_characteristics.xlsx` | `Cell_num` and 17 discovery descriptors |
| `Electrochemical_indicators.xlsx` | `Cell_num` and 33 external validation descriptors |

The configured measured-data experiment expects 71 cells. IDs are sorted
numerically, aligned across files, and checked for duplicates, missing IDs,
record counts, finite curves, and index agreement. Do not invent units or a
sampling interval: those fields were not encoded in the saved loader contract.

The common representation matrix contains 200 current residual coordinates
followed by 200 voltage residual coordinates, standardized across the dataset.
The frozen CDDI reducer instead receives each raw `(200, 2)` cell curve and the
raw reference curve. This distinction is explicit in `REPRODUCIBILITY.md`.

## First-200s application inputs

Expected built-in directory: `battery_compute/first_200s/`, with cell curve files,
`Model_first_200s.xls`, and `Charging_curve_characteristics.xlsx`.
OCV/IR statistics are resolved using the application settings/data resolver.
Inspect those modules when preparing a separately released measured-data bundle.

## Standalone search data path

The integrated historical evaluator also recognizes
`battwise_cddi_search/problems/battery_data/<case>/`.
On Linux, a local link can avoid duplicating the input directory:

```bash
ln -s ../../battery_compute battwise_cddi_search/problems/battery_data
```

Create it only if the target path does not already exist. The link and measured
files are ignored by Git.

## No validation leakage

CDDI discovery may use the 17 discovery descriptors. Capacity and the 33 external
descriptors are independent validation variables. They must not be used to
reselect the frozen program or enter the OCV/IR/CDDI clustering feature matrix.
