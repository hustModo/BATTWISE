# One-dimensional All-200 benchmark

See the repository-level [experimental protocol](../../docs/EXPERIMENTS.md),
[data contract](../../docs/DATA_FORMAT.md), and
[reproducibility boundaries](../../docs/REPRODUCIBILITY.md).

The common baseline matrix is a globally standardized reference-residual matrix.
The frozen BATTWISE-CDDI reducer receives raw cell/reference curves instead.
All learned one-dimensional outputs are paired with OCV/IR and clustered using
the shared GA-KMeans implementation. Capacity and the 33 electrochemical
descriptors are evaluation-only.

The saved formal configuration uses K=6, population 200, 80 iterations, and
seeds 0–29. Do not report smoke or synthetic checks as formal experimental results.
This code-only package includes no measured inputs or saved outputs.
