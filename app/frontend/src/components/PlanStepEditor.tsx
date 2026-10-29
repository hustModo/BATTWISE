import { useEffect, useMemo, useState } from "react";
import clsx from "clsx";
import { schemaFor, type ParamSpec } from "./planStepSchema";
import useSession from "@/store/sessionStore";
import type { PlanStep } from "@/types";

/**
 * Inline parameter editor for a single pending plan step.
 *
 * Rendered by PlanPanel when the user clicks the pencil icon on a
 * pending step. Builds a small form from the agent's schema, lets the
 * user edit values, and rounds the result through the backend
 * ``plan_update_step`` action. The backend re-clamps everything, so we
 * stay relaxed about validation here (just type-of-input hints).
 */

type Draft = Record<string, string>;

interface Props {
  step: PlanStep;
  onClose: () => void;
}

function paramToDraftString(spec: ParamSpec, v: unknown): string {
  if (v === undefined || v === null) return "";
  if (spec.kind === "int_list") {
    if (Array.isArray(v)) return v.join(",");
    return String(v);
  }
  return String(v);
}

function draftToParams(specs: ParamSpec[], draft: Draft): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const spec of specs) {
    const raw = (draft[spec.key] ?? "").trim();
    if (raw === "") {
      // Backend clears the key when it sees "" or null.
      out[spec.key] = "";
      continue;
    }
    if (spec.kind === "int") {
      const n = Number(raw);
      out[spec.key] = Number.isFinite(n) ? Math.trunc(n) : raw;
    } else if (spec.kind === "float") {
      const n = Number(raw);
      out[spec.key] = Number.isFinite(n) ? n : raw;
    } else if (spec.kind === "int_list") {
      const parts = raw
        .split(/[,\s]+/)
        .map((x) => x.trim())
        .filter(Boolean)
        .map((x) => Number(x))
        .filter((n) => Number.isFinite(n))
        .map((n) => Math.trunc(n));
      out[spec.key] = parts;
    } else {
      out[spec.key] = raw;
    }
  }
  return out;
}

export default function PlanStepEditor({ step, onClose }: Props) {
  const planUpdateStep = useSession((s) => s.planUpdateStep);
  const specs = useMemo(() => schemaFor(step.agent), [step.agent]);

  const initial: Draft = useMemo(() => {
    const d: Draft = {};
    for (const spec of specs) {
      const value = step.params?.[spec.key]
        ?? (spec.key === "cddi_weight" ? step.params?.ddia_weight : undefined);
      d[spec.key] = paramToDraftString(spec, value);
    }
    return d;
  }, [specs, step.params]);

  const [draft, setDraft] = useState<Draft>(initial);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDraft(initial);
    setError(null);
  }, [initial]);

  if (specs.length === 0) {
    return (
      <div className="mt-1 ml-6 mr-1 px-2 py-1.5 rounded bg-slate-50 border border-slate-200 text-[11px] text-slate-500 flex items-center justify-between">
        <span>This agent has no editable parameters.</span>
        <button
          type="button"
          className="text-slate-500 hover:text-slate-700 px-1"
          onClick={onClose}
        >
          Close
        </button>
      </div>
    );
  }

  const dirty = Object.keys(draft).some((k) => (draft[k] ?? "") !== (initial[k] ?? ""));

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const params = draftToParams(specs, draft);
      const res = await planUpdateStep(step.id, params);
      if (!res.ok) {
        setError(res.error ?? "Failed to save");
        setSaving(false);
        return;
      }
      onClose();
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="mt-1 ml-6 mr-1 px-2.5 py-2 rounded bg-slate-50 border border-slate-200 text-[11px]">
      <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
        {specs.map((spec) => (
          <label key={spec.key} className="flex flex-col gap-0.5 min-w-0">
            <span className="text-slate-500 truncate" title={spec.description}>
              {spec.label}
              {(spec.kind === "int" || spec.kind === "float") && (
                <span className="text-slate-400 font-normal ml-1">
                  [{spec.min}–{spec.max}]
                </span>
              )}
            </span>
            {spec.kind === "enum" ? (
              <select
                value={draft[spec.key] ?? ""}
                onChange={(e) =>
                  setDraft((d) => ({ ...d, [spec.key]: e.target.value }))
                }
                className="px-1.5 py-1 rounded border border-slate-300 bg-white text-[11px] focus:outline-none focus:border-emerald-500"
              >
                <option value="">(default)</option>
                {(spec.options || []).map((opt) => (
                  <option key={opt} value={opt}>
                    {opt}
                  </option>
                ))}
              </select>
            ) : spec.kind === "int" || spec.kind === "float" ? (
              <input
                type="number"
                inputMode={spec.kind === "float" ? "decimal" : "numeric"}
                step={spec.kind === "float" ? "0.05" : "1"}
                min={spec.min}
                max={spec.max}
                value={draft[spec.key] ?? ""}
                onChange={(e) =>
                  setDraft((d) => ({ ...d, [spec.key]: e.target.value }))
                }
                placeholder="(default)"
                className="px-1.5 py-1 rounded border border-slate-300 bg-white text-[11px] focus:outline-none focus:border-emerald-500"
              />
            ) : (
              <input
                type="text"
                value={draft[spec.key] ?? ""}
                onChange={(e) =>
                  setDraft((d) => ({ ...d, [spec.key]: e.target.value }))
                }
                placeholder={spec.description || "1,2,3"}
                className="px-1.5 py-1 rounded border border-slate-300 bg-white text-[11px] focus:outline-none focus:border-emerald-500"
              />
            )}
          </label>
        ))}
      </div>

      {error && (
        <div className="mt-1.5 text-[11px] text-battery-danger">{error}</div>
      )}

      <div className="mt-2 flex items-center gap-2 justify-end">
        <span className="text-[10px] text-slate-400 mr-auto">
          Values are clamped to the listed range on save.
        </span>
        <button
          type="button"
          className="px-2 py-0.5 rounded text-[11px] text-slate-600 hover:bg-slate-200"
          onClick={onClose}
          disabled={saving}
        >
          Cancel
        </button>
        <button
          type="button"
          className={clsx(
            "px-2.5 py-0.5 rounded text-[11px] font-medium text-white",
            dirty && !saving
              ? "bg-emerald-600 hover:bg-emerald-700"
              : "bg-slate-300 cursor-not-allowed",
          )}
          onClick={handleSave}
          disabled={!dirty || saving}
        >
          {saving ? "Saving…" : "Save"}
        </button>
      </div>
    </div>
  );
}
