import { useState } from "react";
import clsx from "clsx";
import useSession from "@/store/sessionStore";
import type { PlanStep, PlanStepStatus } from "@/types";
import PlanStepEditor from "./PlanStepEditor";

/**
 * Workflow plan panel — visible when a multi-step plan is active.
 *
 * Backend (services/planner + orchestrator/pipeline) drives the plan and
 * we mirror it here. The user can:
 *   * Click "Run Next Step" to advance.
 *   * Click "Skip" to mark the current step skipped + advance.
 *   * Click "Cancel" to abort the whole plan.
 *   * Click the pencil icon on a pending step to inline-edit its params
 *     (Designer mode, Cluster k / GA knobs, Refine cluster_ids, …).
 *
 * Buttons round-trip through ``plan_action`` (control) and
 * ``plan_update_step`` (param edit) REST actions. The same controls
 * also work conversationally ("继续" / "跳过" / "取消计划").
 */

const AGENT_LABEL: Record<PlanStep["agent"], string> = {
  designer: "Designer",
  compute: "Compute",
  cluster: "Cluster",
  refine: "Refine",
};

const STATUS_GLYPH: Record<PlanStepStatus, string> = {
  pending: "○",
  running: "▶",
  done: "✓",
  skipped: "↷",
  cancelled: "✗",
  error: "!",
};

const STATUS_COLOR: Record<PlanStepStatus, string> = {
  pending: "text-slate-400",
  running: "text-battery-accent animate-pulse",
  done: "text-emerald-600",
  skipped: "text-slate-400",
  cancelled: "text-slate-400",
  error: "text-battery-danger",
};

const PLAN_STATUS_LABEL: Record<string, string> = {
  awaiting_confirmation: "Awaiting confirmation",
  in_progress: "Running",
  done: "Completed",
  cancelled: "Cancelled",
};

function renderParamsSummary(params: Record<string, unknown> | undefined): string {
  if (!params) return "";
  const parts: string[] = [];
  for (const [key, v] of Object.entries(params)) {
    const k = key === "ddia_weight" ? "cddi_weight" : key;
    if (v === undefined || v === null || v === "") continue;
    if (Array.isArray(v)) {
      parts.push(`${k}=[${v.join(",")}]`);
    } else if (typeof v === "object") {
      parts.push(`${k}=${JSON.stringify(v)}`);
    } else {
      parts.push(`${k}=${v}`);
    }
  }
  return parts.join(", ");
}

export default function PlanPanel() {
  const plan = useSession((s) => s.plan);
  const planAction = useSession((s) => s.planAction);
  const [editingStepId, setEditingStepId] = useState<number | null>(null);

  if (!plan) return null;

  const terminal = plan.status === "done" || plan.status === "cancelled";
  const current = plan.steps[plan.current_idx];

  return (
    <div className="card overflow-hidden shrink-0 border-l-2 border-l-emerald-500/70">
      <div className="px-3 py-2 flex items-center justify-between border-b border-slate-100">
        <div className="flex items-center gap-2 min-w-0">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-500 shrink-0" />
          <div className="min-w-0">
            <div className="text-[10px] uppercase tracking-widest text-slate-400 font-medium">
              Workflow plan · {PLAN_STATUS_LABEL[plan.status] || plan.status}
            </div>
            <div className="text-[12px] font-medium text-slate-700 truncate" title={plan.goal}>
              {plan.goal}
            </div>
          </div>
        </div>
        {!terminal && (
          <button
            type="button"
            className="text-[11px] px-2 py-0.5 rounded text-slate-500 hover:bg-slate-100"
            onClick={() => planAction("cancel")}
            title="Cancel the plan"
          >
            Cancel
          </button>
        )}
      </div>

      <ul className="px-3 py-2 space-y-1">
        {plan.steps.map((step, idx) => {
          const isCurrent = idx === plan.current_idx && plan.status === "in_progress";
          const isEditing = editingStepId === step.id;
          const editable = step.status === "pending" && !terminal;
          const summary = renderParamsSummary(step.params as Record<string, unknown> | undefined);
          return (
            <li key={step.id} className="text-[12px]">
              <div
                className={clsx(
                  "flex items-start gap-2 py-0.5",
                  isCurrent && "font-medium",
                )}
              >
                <span className={clsx("font-mono w-4 text-center", STATUS_COLOR[step.status])}>
                  {STATUS_GLYPH[step.status]}
                </span>
                <div className="flex-1 min-w-0">
                  <div className="text-slate-700 truncate">
                    <span className="text-slate-400 mr-1">{idx + 1}.</span>
                    <span className="font-medium">{AGENT_LABEL[step.agent]}</span>
                    <span className="text-slate-500"> — {step.description}</span>
                  </div>
                  {summary && !isEditing && (
                    <div className="text-[10px] text-slate-400 font-mono truncate" title={summary}>
                      {summary}
                    </div>
                  )}
                </div>
                {editable && (
                  <button
                    type="button"
                    aria-label={isEditing ? "Close param editor" : "Edit parameters"}
                    className={clsx(
                      "shrink-0 px-1 py-0.5 rounded text-[11px]",
                      isEditing
                        ? "text-emerald-700 bg-emerald-50"
                        : "text-slate-400 hover:text-slate-700 hover:bg-slate-100",
                    )}
                    onClick={() =>
                      setEditingStepId(isEditing ? null : step.id)
                    }
                    title={isEditing ? "Close editor" : "Edit step params"}
                  >
                    {isEditing ? "▾" : "✎"}
                  </button>
                )}
              </div>
              {isEditing && editable && (
                <PlanStepEditor
                  step={step}
                  onClose={() => setEditingStepId(null)}
                />
              )}
            </li>
          );
        })}
      </ul>

      {!terminal && current && (
        <div className="px-3 pb-3 pt-1 border-t border-slate-100 flex items-center gap-2">
          <button
            type="button"
            className={clsx(
              "px-3 py-1 rounded text-[12px] font-medium",
              "bg-emerald-600 text-white hover:bg-emerald-700",
            )}
            onClick={() => planAction("confirm")}
            title="Run the next pending step"
          >
            {plan.status === "awaiting_confirmation" ? "Start" : "Run Next Step"}
          </button>
          <button
            type="button"
            className="px-3 py-1 rounded text-[12px] text-slate-600 hover:bg-slate-100"
            onClick={() => planAction("skip")}
            title="Mark this step skipped and advance"
          >
            Skip
          </button>
          <span className="text-[11px] text-slate-400 ml-auto">
            ✎ to edit params · chat: <code className="font-mono">继续 / 跳过 / 取消计划</code>
          </span>
        </div>
      )}
    </div>
  );
}
