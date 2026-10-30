# Experimental code

## Shared All-200 benchmark

The `experiments.fair_1d_benchmark` package provides data loading, representation
contracts, shared GA-KMeans, metrics, result serialization, and a CLI.

Included representations are the frozen BATTWISE-CDDI program, KPCA-1D,
FC-VAE-1D, DECM-1D, and an OCV/IR-only control. These are the implementations
present in the preserved benchmark. Their presence does not redefine the planned
main manuscript comparison or establish missing FunSearch evidence.

With the separately provided measured input directory:

```bash
python -m experiments.fair_1d_benchmark.run \
  --config experiments/fair_1d_benchmark/config.yaml
```

The formal configuration retains fixed `K=6`, GA population 200, 80 iterations,
and seeds 0–29. Every one-dimensional representation is paired with the same OCV
and IR. Capacity and external electrochemical descriptors are evaluation-only.
Outputs go to ignored `outputs/benchmark/` in this distribution.

The smoke configuration shortens the GA and seed list; it is an interface check,
not manuscript evidence. Its FC-VAE settings still include a potentially long
training schedule. The offline demo is the preferred quick reviewer check.

## Frozen CDDI

`programs/cddi/program.json` contains executable source only; the `.py` file
contains the same frozen generation-20 program. The normalized source SHA-256 is:

```text
f18815885200043f9f14371d169b6bc600757d8b19e70f262fd175d2d6bbfda7
```

It is included as the method's executable definition, not as a saved result
table. No new candidate selection was performed while preparing this release.
The benchmark applies a post-processing factor of `1e5`, as recorded in the
configuration. Do not conflate that scaling with the discovered function itself.

## Reflective discovery

The integrated search is in `reflective_metric_search.py`. The default reflection
templates are embedded in its source and materialized when needed. Battery
problem descriptions, function contracts, the seed function, and evaluation code
are included. Search requires input data and a configured LLM endpoint.

Example, after supplying compatible data and environment credentials:

```bash
cd battwise_cddi_search
python main.py algorithm=reflective_metric_search case_num=[4] \
  model=your-provider-model max_fe=3 pop_size=4 batch_size=4
```

This is a new search, not a replay of the formal run. Historical populations,
session-specific programs, logs, and reflection records are omitted. Without
historical seed directories, the application seed builder falls back to the
supplied baseline. A fresh stochastic search is not guaranteed to recover the
frozen function or its historical trajectory.

## Statistical analysis code

```bash
python experiments/statistics/analyze_fair_benchmark.py \
  --input outputs/benchmark --output outputs/statistics
```

The script reads already-generated benchmark CSVs. It provides descriptive
statistics, Friedman tests, paired Wilcoxon tests, Holm corrections, win/tie/loss,
rank-biserial effects, clustering stability, and cross-method ARI. Its validation
expects the complete saved experiment shape (five methods, 30 seeds, 71 cells,
and 33 external indicators). It is not a generic analysis of partial smoke runs.
No result CSVs or statistical outputs are committed here.

## FunSearch

See `comparisons/funsearch/README.md`. Only candidate function text was located;
a complete FunSearch runner, configuration, and verified comparison provenance
must be supplied before claiming end-to-end FunSearch reproducibility.
