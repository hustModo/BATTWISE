# Reproducibility and version boundaries

## What this revision represents

This distribution combines the current application/search source with preserved
experiment/statistics source and the frozen selected program. The current
application includes the CDDI naming/compatibility changes and the repaired
periodic reflection callback. The original main project had no Git metadata.
A new release commit identifies this distribution; it must not be described as
the historical commit used to generate the manuscript's saved results.

`source_provenance.json` records file-level source hashes and packaging changes.
`MANIFEST.sha256` records shipped file bytes. Neither contains a server address,
private account path, credentials, or historical Git history.

## Preserve the actual input semantics

- Baseline representations consume the channel-standardized reference-residual
  matrix with 400 coordinates per cell.
- `BATTWISECDDReducer` passes raw cell and reference Current/Voltage arrays to
  the frozen function. The function compares the two responses; it does not
  explicitly receive a preconstructed 400-dimensional residual vector.
- Do not claim identical residual input for every method until the implementation
  and formal-result provenance are reconciled. Packaging does not silently
  alter the reducer to make a manuscript statement appear correct.

## Environment evidence

The preserved formal record reports Python 3.10.12, NumPy 2.2.6, pandas 2.3.3,
and scikit-learn 1.5.2. Older backend requirements list different NumPy/pandas
pins. `environment.observed.json` records the currently inspected server package
metadata; it does not establish that every listed package was used in the formal
run or that every dependency combination has been validated from a clean install.

The inspected server includes PyTorch 2.3.0+cpu. A NumPy-to-tensor-to-NumPy
conversion check passed with the inspected NumPy 2.2.6 environment. This basic
check does not by itself reconstruct the historical deep-model training
environment. The optional requirements add PyTorch to the core profile; see
`VALIDATION.md` for executed synthetic experiment tests.

## Evidence that is intentionally absent

- Measured inputs and saved experimental/statistical outputs.
- A verified complete FunSearch runner and experiment protocol.
- Native candidate IDs, parent lineage, candidate-linked reflections, and every
  individual discovery coefficient for the preserved historical search.
- Evidence of non-empty accumulated long-term reflection in the formal run.
- Saved GA-KMeans centroids; do not synthesize centroids and label them historical.
- A formal refinement-session registry. Historical application demonstrations
  must not be relabeled as formal releases.

## Release validation

Read `VALIDATION.md` for executed checks. Synthetic demos/test fixtures validate
code contracts and installation behavior; they do not validate manuscript claims
or replace the measured dataset. Full LLM discovery and measured-data experiments
were not rerun to prepare this distribution.
