# Offline synthetic demonstration

Run from the repository root:

```bash
python examples/demo_cddi.py
```

The script verifies the frozen program's normalized SHA-256, creates three
deterministic synthetic two-channel curves with 200 points, and calls the frozen
CDDI function. It prints one finite scalar per curve and a measured elapsed time.

Expected behavior is successful hash verification, three finite values, and a
final `PASS` line. Values are interface-demo outputs, not measured battery
results. No files, plots, candidate populations, or result tables are produced.
No API credential, network connection, model fitting, or metric search is needed.

No scientific claim is inferred from the ranking of these synthetic values.
The recorded runtime for the actual validation machine is in `VALIDATION.md`.
