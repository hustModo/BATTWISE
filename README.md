# BATTWISE

**BATTWISE** is a hierarchical multi-agent workflow for retired-battery triage,
curve-discrepancy indicator discovery, clustering, and interactive refinement.
The discovered scalar indicator is called **CDDI**; the combined method is
**BATTWISE-CDDI**.

This is the **review-code-20261029** code-only distribution. It contains application
and experiment implementations, not the manuscript's saved result tables or
figures. See [release scope](docs/RELEASE_SCOPE.md) and
[known limitations](docs/LIMITATIONS.md) before interpreting reproducibility claims.

## Start here

| Task | Documentation |
|---|---|
| Install in a separate environment | [Installation](docs/INSTALL.md) |
| Run a small offline example | [Demo](docs/DEMO.md) |
| Run the application | [Application](docs/APPLICATION.md) |
| Inspect or rerun experiments | [Experiments](docs/EXPERIMENTS.md) |
| Prepare input tables | [Data contract](docs/DATA_FORMAT.md) |
| Understand evidence and version boundaries | [Reproducibility](docs/REPRODUCIBILITY.md) |
| Prepare an anonymous reviewer link | [Anonymous review](docs/ANONYMOUS_REVIEW.md) |
| Draft the manuscript statement | [Code availability](docs/CODE_AVAILABILITY.md) |

## What the code implements

1. Router and Planner interpret a user's request and maintain an interactive plan.
2. Designer discovers or revises a scalar curve-discrepancy program using a
   lightweight design pass or performance-guided reflective search.
3. Compute runs the program in a separate process with interface, finite-output,
   progress, and timeout checks.
4. Cluster groups cells using **OCV + IR + CDDI**.
5. Refine designs and evaluates an All-200 program for selected parent groups,
   then merges subgroup labels back into the global grouping.
6. Chat, contextual guidance, and conversation memory support human intervention
   throughout the workflow.

First-200s supports rapid screening. All-200 supports complete indicator discovery
and selected-group refinement. Capacity and the 33 external validation indicators
are evaluation-only. A 2 Ah threshold may select a refinement cohort; it is not an
input to CDDI discovery or clustering.

## Repository layout

```text
app/backend/                         FastAPI, agents, orchestration, services, tests
app/frontend/                        React/TypeScript interface
Evo_Textgrad_battery_eoh_deepseek/    Integrated reflective search; legacy path retained
experiments/fair_1d_benchmark/       Data loading, representations, GA-KMeans, metrics
experiments/statistics/              Friedman, paired Wilcoxon/Holm, stability, ARI
programs/cddi/                       Frozen selected executable CDDI program
comparisons/funsearch/               Preserved candidate code and scope explanation
dianchi03/                          Compute reference code, without measured data
examples/                           Deterministic synthetic offline demo
scripts/                            Release checks
docs/                               Installation, protocols, scope, limitations
deploy/                             Localhost launch scripts
```

The historical search directory name is retained to keep imports and file
contracts usable. It is not the public method name. Upstream EoH components are
credited separately in [third-party notices](THIRD_PARTY_NOTICES.md).

## Quick offline demo

After installing the core dependencies:

```bash
python examples/demo_cddi.py
```

The demo generates synthetic two-channel curves in memory, checks the frozen
program's source hash, and prints finite scalar outputs. It does not call an LLM,
train a model, run a search, or reproduce a manuscript experiment.

## Tests

```bash
python -m pytest experiments/fair_1d_benchmark/tests/test_data.py \
  experiments/fair_1d_benchmark/tests/test_metrics.py \
  experiments/fair_1d_benchmark/tests/test_battwise_cddi.py -q
python app/backend/tests/test_cddi_compatibility.py
python scripts/check_release.py
```

Application tests are standalone scripts. Benchmark tests use pytest. Some
integration tests require measured input files or optional PyTorch dependencies;
they are not implied by the short offline commands above.

## Licensing and publication

BATTWISE-authored code is released under the [MIT License](LICENSE). Bundled
third-party code retains its upstream notices and license. No claim is made that
this new repository commit is the historical commit used for formal experiments.
For publication, freeze the relevant revision and archive that version in a
DOI-minting repository. This review package does not yet have a DOI.

The current code-only scope intentionally omits measured datasets, saved
experimental outputs, figures, plotting pipelines, runtime sessions, logs,
credentials, build products, and historical Git metadata.
