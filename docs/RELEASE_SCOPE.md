# Code-only release scope

| Include now | Purpose |
|---|---|
| Agent, planner, router, workflow, frontend source | Inspect the BATTWISE system |
| Designer/Compute isolation and validators | Inspect executable-program handling |
| Integrated reflective search and battery prompts/evaluator | Inspect metric discovery |
| Frozen executable CDDI and source hash | Define the deployed indicator |
| Data loader and preprocessing | Inspect ordering and feature construction |
| GA-KMeans, refinement and label merge | Inspect grouping behavior |
| Benchmark representations, configurations and tests | Inspect experimental methods |
| Statistical-analysis source | Inspect the planned analysis of saved outputs |
| Dependencies, licenses, docs, synthetic example | Install and test code contracts |

| Exclude now | Reason |
|---|---|
| Measured battery files | Separate data-release work, not this code-only stage |
| Result tables, assignments, embeddings, model checkpoints | Experimental outputs are out of scope |
| Statistical output tables | Results are out of scope |
| Paper figures, plotting scripts, figure source data | Figure work is deferred |
| Search logs, historical populations, workflow sessions | Historical/runtime evidence is not the release source |
| Saved LLM configuration, encryption keys, `.env` | Credentials and machine state |
| `node_modules`, caches, `dist`, virtual environments | Rebuildable local artifacts |
| Original Git history, backup directories, handoff reports | Private or identifying operational material |

The frozen CDDI is code needed to evaluate the method. Candidate text under
`comparisons/funsearch` is clearly labeled partial code evidence. Neither is
presented as a complete experimental-results bundle.

## Before claiming a complete submission artifact

1. Supply a separately documented measured-data access/deposit plan.
2. Complete or explicitly delimit the FunSearch reproduction materials.
3. Reconcile the CDDI raw-response/residual description in the manuscript.
4. Validate optional deep-model dependencies in a clean environment.
5. Freeze the revision supplied to reviewers; verify the anonymous view.
6. For publication, archive the relevant revision with a permanent identifier.

This distribution does not assert that these deferred items are already complete.
