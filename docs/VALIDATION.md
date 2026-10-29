# Validation record

Date: 2026-10-29. These are code-validation checks, not manuscript experiments.

| Check | Outcome |
|---|---|
| Release source/private-path/credential/artifact scan | Passed |
| Frozen CDDI normalized source hash | Matches the documented frozen program |
| Offline synthetic CDDI demo | Passed; three finite scalars, no saved outputs |
| Complete benchmark unit-test directory | **78 passed** |
| Application CDDI compatibility regression | Passed |
| Legacy router-tool regression | Passed |
| Plan parameter-editing regression | Passed |
| Reflective-search source contract | Passed |
| Statistical-analysis CLI `--help` | Passed |
| Search CLI `--help` | Passed; no search started |
| NumPy/tensor/NumPy conversion | Passed in the observed optional runtime |

The demo's measured computation took approximately 0.06 s for three synthetic
200-point/two-channel curves; the complete interpreter invocation took about
0.58 s. The benchmark unit tests completed in 9.54 s. These values are machine-
and cache-dependent and are not installation-time or scientific timing claims.

Validation used an isolated temporary source directory and the observed Linux
Python 3.10.12 environment. Package versions are in `environment.observed.json`.
The current application frontend source also passed TypeScript `--noEmit` and
Vite production build before packaging; no build products are shipped.

No full measured-data benchmark, LLM request, reflective discovery run, or formal
statistical recomputation was performed. No result table or plot was generated
for inclusion in this repository. Unit tests may train tiny synthetic fixtures
inside temporary directories; those fixtures are not research data.

A fresh network installation, cross-platform execution, full provider integration,
and a complete FunSearch comparison were not validated by these checks.
