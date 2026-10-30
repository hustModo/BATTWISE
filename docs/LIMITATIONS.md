# Known limits

- CDDI's current reducer uses raw cell/reference curves; the common benchmark
  matrix is standardized residual data. The difference is retained and disclosed.
- The periodic reflection callback's invalid method call has been repaired.
  That repair does not demonstrate that adaptive weights were applied to the
  historical EoH sampler, or that the formal run accumulated long-term memory.
- The selected generation-20 program is unchanged. A later candidate was excluded
  historically for non-finite output; no selection is repeated in this package.
- LLM responses, provider versions, and stochastic candidate searches can vary.
  No exact search-trajectory replay is promised.
- Measured-data experiments cannot be reproduced from a code-only package alone.
- FunSearch candidate functions do not constitute a complete FunSearch runner.
- The historical `battery_cddi.py` problem-class wrapper expects evaluation
  helpers that are absent from the replaceable candidate program. It is retained
  for provenance and is not a supported entry point. Use the documented search
  CLI and `problems/battery_cddi/eval.py`; both are covered by offline checks.
- Observed optional deep-model dependencies do not establish the complete
  historical training environment; see `REPRODUCIBILITY.md`.
- The battery-pack work is proof-of-concept pack-level validation. No new pack
  experiment, individual-cell trace, or unrecorded measurement is supplied here.
