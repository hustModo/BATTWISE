# Working on the code-only review distribution

- Public names are BATTWISE, CDDI and BATTWISE-CDDI. Preserve explicit
  compatibility readers for historical names and the upstream EoH attribution.
- Keep changes focused. State assumptions, reproduce defects, and report actual
  validation instead of claiming an unexecuted test or full scientific replay.
- Never commit credentials, runtime sessions, measured inputs, saved results,
  figures, caches, checkpoints, or original private operational history.
- Do not modify the frozen CDDI source or reselect it against capacity or the
  33 external validation descriptors. Its normalized SHA-256 is documented.
- Capacity and external descriptors are validation-only; clustering inputs are
  OCV + IR + CDDI. Cohort-selection thresholds are not discovery features.
- Application tests are standalone scripts; benchmark tests use pytest.
  Synthetic test fixtures do not become manuscript evidence.
- Do not launch a real LLM search, measured-data training, or formal experiment
  merely to validate documentation or packaging.
- `docs/RELEASE_SCOPE.md` and `docs/LIMITATIONS.md` define current boundaries.
