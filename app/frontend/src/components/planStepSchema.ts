/**
 * Per-agent param schema for the plan-step inline editor.
 *
 * This is a *mirror* of ``app/backend/services/planner.PARAM_SCHEMA``.
 * The backend is the source of truth and re-validates / clamps every
 * field on the way in — this front-end copy only drives the UI
 * (label / kind / range / dropdown options). Keep them in sync.
 */

export type ParamKind = "int" | "float" | "enum" | "int_list";

export interface ParamSpec {
  key: string;
  kind: ParamKind;
  label: string;
  description?: string;
  // for "int"
  min?: number;
  max?: number;
  // for "enum"
  options?: string[];
}

export type AgentId = "designer" | "compute" | "cluster" | "refine";

export const PARAM_SCHEMA: Record<AgentId, ParamSpec[]> = {
  designer: [
    {
      key: "mode",
      kind: "enum",
      label: "Mode",
      description:
        "Lightweight = one-shot LLM design; Reflective Search = performance-guided program search.",
      options: ["lightweight", "reflective_search"],
    },
    {
      key: "max_fe",
      kind: "int",
      label: "Max generations",
      description: "Search generations (only used in Reflective Search).",
      min: 1,
      max: 30,
    },
    {
      key: "pop_size",
      kind: "int",
      label: "Population",
      description: "Individuals per generation.",
      min: 2,
      max: 32,
    },
    {
      key: "batch_size",
      kind: "int",
      label: "Batch size",
      description: "Per-generation evaluation batch.",
      min: 1,
      max: 32,
    },
  ],
  compute: [],
  cluster: [
    {
      key: "target_k",
      kind: "int",
      label: "Target k",
      description: "Fixed cluster count; leave blank to let GA pick.",
      min: 2,
      max: 12,
    },
    {
      key: "min_k",
      kind: "int",
      label: "Min k",
      description: "Lower bound for the GA k search.",
      min: 2,
      max: 12,
    },
    {
      key: "max_k",
      kind: "int",
      label: "Max k",
      description: "Upper bound for the GA k search.",
      min: 2,
      max: 12,
    },
    {
      key: "pop_size",
      kind: "int",
      label: "GA population",
      description: "Cluster-GA population size.",
      min: 20,
      max: 500,
    },
    {
      key: "max_iter",
      kind: "int",
      label: "GA iterations",
      description: "Cluster-GA max generations.",
      min: 10,
      max: 500,
    },
    {
      key: "feedback_metric",
      kind: "enum",
      label: "Feedback metric",
      description: "GA fitness criterion.",
      options: ["combined", "dbi", "variance"],
    },
    {
      key: "ocv_weight",
      kind: "float",
      label: "OCV weight",
      description: "Relative OCV weight after normalization; default 1.",
      min: 0.05,
      max: 20,
    },
    {
      key: "ir_weight",
      kind: "float",
      label: "IR weight",
      description: "Relative internal-resistance weight after normalization; default 1.",
      min: 0.05,
      max: 20,
    },
    {
      key: "cddi_weight",
      kind: "float",
      label: "CDDI weight",
      description: "Relative CDDI weight after normalization; default 1.",
      min: 0.05,
      max: 20,
    },
  ],
  refine: [
    {
      key: "cluster_ids",
      kind: "int_list",
      label: "Cluster IDs",
      description: "Which clusters to drill into (comma-separated, e.g. 1,3,5).",
      min: 0,
      max: 50,
    },
  ],
};

export function schemaFor(agent: string): ParamSpec[] {
  return PARAM_SCHEMA[agent as AgentId] || [];
}
