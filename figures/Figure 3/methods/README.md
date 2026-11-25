# All_200 fair one-dimensional benchmark

This experiment compares one-dimensional representations of the same 71
retired-cell Current/Voltage curves. Every learned method receives the common
`All200Dataset.curves` matrix with shape `(71, 400)` and returns one finite
value per `Cell_num`.

## Fairness contract

- Curve input is the channel-standardised residual relative to
  `Model_200.xls`: 200 current residuals followed by 200 voltage residuals.
- The 17 charging-curve characteristics are the only optional training
  targets.
- Capacity and the 33 electrochemical indicators are evaluation-only.
- Every one-dimensional representation is combined with the same OCV and IR.
- Final clustering always uses the shared GA-KMeans implementation with
  `K=6`; method modules must not provide their own final clustering.
- Cell order must exactly match the public dataset contract.

## Method ownership

The shared foundation owns data loading, preprocessing, clustering, metrics,
reporting, and the command-line runner. Reproduction tasks add only:

- `methods/kpca_1d.py` plus its tests;
- `methods/fc_vae_1d.py` plus its tests;
- `methods/decm_1d.py` plus its tests.

Each method is registered by adding its dotted class path to `config.yaml`.

## Commands

Run a one-seed foundation smoke test:

```bash
<REMOTE_HOME>/miniconda3/bin/python3.10 -m experiments.fair_1d_benchmark.run \
  --config experiments/fair_1d_benchmark/config.smoke.yaml
```

The smoke configuration uses `pop_size=20` and `max_iter=5` only to validate
the shared interfaces. Results from it must not be used in the paper. The
formal `config.yaml` retains `pop_size=200`, `max_iter=80`, and seeds 0–29.

Run every registered method with the 30 configured seeds:

```bash
<REMOTE_HOME>/miniconda3/bin/python3.10 -m experiments.fair_1d_benchmark.run \
  --config experiments/fair_1d_benchmark/config.yaml
```

Run foundation tests:

```bash
<REMOTE_HOME>/miniconda3/bin/python3.10 -m pytest \
  experiments/fair_1d_benchmark/tests -q
```
