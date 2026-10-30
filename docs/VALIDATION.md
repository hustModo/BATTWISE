# Validation record

Date: 2026-10-30, version `review-code-20261030`. These are code-validation
checks, not manuscript experiments.

| Check | Outcome |
|---|---|
| Release source/private-path/credential/artifact scan | Passed |
| Frozen CDDI normalized source hash | Matches the documented frozen program |
| Offline synthetic CDDI demo | Passed; three finite scalars, no saved outputs |
| Default `python scripts/test_offline.py` | **19 pytest cases passed**, all 8 command stages passed |
| Complete `python scripts/test_offline.py --full` | **90 pytest cases passed**, all 8 command stages passed |
| Complete benchmark unit-test directory | 78 cases included in the full suite |
| Renamed package, application paths and seven Hydra configurations | Passed |
| First-200s and All-200 evaluator subprocesses | Passed using temporary synthetic workbooks and a known correlation target |
| New and historical saved-candidate filename readers | Both passed |
| Application CDDI compatibility regression | Passed |
| Legacy router-tool regression | Passed within the CDDI compatibility check |
| Plan parameter-editing regression | Passed |
| Reflective-search source contract | Passed |
| Statistical-analysis CLI `--help` | Passed |
| Search CLI `--help` | Passed; no search started |
| NumPy/tensor/NumPy conversion | Passed in the observed optional runtime |

The default command completed in 14.25 s (pytest: 5.17 s); the full command
completed in 21.22 s (pytest: 11.51 s). The demo took approximately 0.05 s to
compute three synthetic 200-point/two-channel curves. These values are machine-
and cache-dependent and are not installation-time or scientific timing claims.

The directory migration also removed stale package-initialization imports of
absent historical helpers. Synthetic evaluator fixtures seed the temporary
harness because the frozen program performs random sampling. The frozen
program's bytes and normalized SHA-256 were not changed.

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
