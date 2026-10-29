import clsx from "clsx";
import useSession from "@/store/sessionStore";

const COLOR: Record<string, string> = {
  direct_reuse: "from-battery-charge/20 to-battery-charge/5 border-battery-charge/30",
  reuse: "from-battery-charge/20 to-battery-charge/5 border-battery-charge/30",
  second_life: "from-battery-electrolyte/20 to-battery-electrolyte/5 border-battery-electrolyte/30",
  material_recycle: "from-battery-amber/20 to-battery-amber/5 border-battery-amber/30",
  needs_review: "from-slate-200 to-slate-50 border-slate-200",
  unknown: "from-slate-200 to-slate-50 border-slate-200",
};

const LABEL: Record<string, string> = {
  direct_reuse: "Direct Reuse",
  reuse: "Direct Reuse",
  second_life: "Second-life (Storage)",
  material_recycle: "Material Recycle",
  needs_review: "Needs Review",
  unknown: "Unknown",
};

interface Props { scope?: "main" | "sub" }

export default function RecyclingDecisionCards({ scope = "main" }: Props) {
  const summary = useSession((s) =>
    scope === "sub" ? s.subCharts.cluster_summary : s.charts.cluster_summary,
  );
  const selected = useSession((s) => s.session?.selected_cluster ?? null);
  // Working multi-selection (ticked cards). Backend keeps this in sync but
  // the user controls the actual checkmarks here. Read the raw list so
  // zustand's shallow-equality avoids re-rendering on unrelated state
  // changes (creating a fresh Set in the selector would always be !==).
  const selectedClustersList = useSession(
    (s) => s.session?.selected_clusters ?? [],
  );
  const checkedSet = new Set<number>(selectedClustersList);
  const select = useSession((s) => s.selectCluster);
  const refine = useSession((s) => s.refineCluster);
  const refineSubmitting = useSession((s) => s.refineSubmitting);
  const toggleCluster = useSession((s) => s.toggleCluster);
  const refineClusters = useSession((s) => s.refineClusters);
  const clearSelectedClusters = useSession((s) => s.clearSelectedClusters);

  // backend returns either `per_cluster` (ClusterAgent) or `per_cluster` (RefinementAgent).
  // Earlier code referenced `clusters`, keep both for safety.
  const clusters: any[] = summary?.per_cluster ?? summary?.clusters ?? [];

  const validIds = new Set<number>(clusters.map((c) => Number(c.cluster)));
  // Only count ticks that still match an existing cluster (defensive
  // against stale selections from a previous run).
  const checked = Array.from(checkedSet).filter((c) => validIds.has(c));
  const checkedCount = checked.length;
  const allChecked = checkedCount > 0 && checkedCount === clusters.length;
  const totalCellsInSelection = clusters
    .filter((c) => checkedSet.has(Number(c.cluster)))
    .reduce((sum, c) => sum + (c.size ?? c.count ?? 0), 0);

  const handleToggleAll = () => {
    if (allChecked) {
      clearSelectedClusters();
    } else {
      // Tick every cluster — sequence the toggles by replacing the set
      // via the existing toggleCluster action; one round-trip per click
      // because we want backend-side persistence.
      for (const c of clusters) {
        if (!checkedSet.has(Number(c.cluster))) toggleCluster(Number(c.cluster));
      }
    }
  };

  const handleRefineSelected = () => {
    if (checkedCount === 0 || refineSubmitting) return;
    refineClusters(checked.slice().sort((a, b) => a - b));
  };

  return (
    <div className="card p-4 flex flex-col gap-3 h-full min-h-0 overflow-hidden">
      <div className="flex items-center justify-between gap-2 shrink-0">
        <div className="text-xs uppercase tracking-widest text-slate-500 font-medium">
          {scope === "sub" ? "Sub-cluster Recycling Decisions" : "Recycling Decisions"}
        </div>
        <div className="text-[11px] text-slate-400">{clusters.length} cluster(s)</div>
      </div>

      {/* Sticky multi-select action bar (main scope only) */}
      {scope === "main" && clusters.length > 0 && (
        <div
          className={clsx(
            "rounded-card border px-3 py-2 flex items-center justify-between gap-2 transition-colors shrink-0",
            checkedCount > 0
              ? "bg-battery-electrolyte/10 border-battery-electrolyte/30"
              : "bg-slate-50 border-slate-200",
          )}
        >
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-1.5 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={allChecked}
                ref={(el) => {
                  if (el) el.indeterminate = checkedCount > 0 && !allChecked;
                }}
                onChange={handleToggleAll}
                className="h-3.5 w-3.5 accent-battery-electrolyte cursor-pointer"
              />
              <span className="text-[11px] text-slate-600 font-medium">
                {checkedCount === 0
                  ? "Select clusters to refine together"
                  : `${checkedCount} cluster${checkedCount === 1 ? "" : "s"} ticked · ${totalCellsInSelection} cells`}
              </span>
            </label>
          </div>
          <div className="flex items-center gap-1.5">
            {checkedCount > 0 && (
              <button
                onClick={() => clearSelectedClusters()}
                className="text-[11px] px-2 py-0.5 rounded border border-slate-300 text-slate-600 hover:bg-white transition-colors"
              >
                Clear
              </button>
            )}
            <button
              onClick={handleRefineSelected}
              disabled={checkedCount === 0 || refineSubmitting}
              className={clsx(
                "text-[11px] px-2.5 py-1 rounded font-medium transition-colors border",
                checkedCount === 0 || refineSubmitting
                  ? "bg-slate-100 border-slate-200 text-slate-400 cursor-not-allowed"
                  : "bg-battery-electrolyte text-white border-battery-electrolyte hover:bg-battery-electrolyte/90",
              )}
            >
              {refineSubmitting
                ? "Refining..."
                : checkedCount > 1
                ? `Refine ${checkedCount} clusters →`
                : "Refine Selected →"}
            </button>
          </div>
        </div>
      )}

      <div className="flex-1 min-h-0 overflow-y-auto scrollbar-thin grid grid-cols-1 lg:grid-cols-2 gap-3 content-start">
        {clusters.length === 0 && (
          <div className="col-span-full h-full grid place-items-center text-slate-400 text-sm">
            {scope === "sub"
              ? "Sub-cluster decisions appear after refinement completes."
              : "Per-cluster decisions will appear after clustering completes."}
          </div>
        )}
        {clusters.map((c) => {
          // Support both ClusterAgent format and RefinementAgent format
          const decision = c.decision?.choice ?? c.recycle?.decision ?? "unknown";
          const reasons: string[] = c.decision?.reason
            ? [c.decision.reason]
            : (c.recycle?.reasons ?? []);
          const count = c.size ?? c.count;
          const capMean = c.capacity?.mean ?? c.capacity_mean;
          const mmdMean = c.mmd_loss?.mean;
          const clusterId = Number(c.cluster);
          const isChecked = checkedSet.has(clusterId);
          const isSelected = clusterId === selected;
          return (
            <div
              key={c.cluster}
              className={clsx(
                "rounded-card border p-3 bg-gradient-to-br transition-all relative",
                COLOR[decision] ?? COLOR.unknown,
                isChecked
                  ? "ring-2 ring-battery-electrolyte shadow-cardHover"
                  : isSelected
                    ? "ring-2 ring-battery-electrolyte/40"
                    : "hover:shadow-card",
              )}
            >
              {/* Top row: tickbox + cluster id + decision label */}
              {scope === "main" ? (
                <div className="flex items-start gap-2">
                  <input
                    type="checkbox"
                    checked={isChecked}
                    onChange={() => toggleCluster(clusterId)}
                    onClick={(e) => e.stopPropagation()}
                    className="h-4 w-4 mt-0.5 accent-battery-electrolyte cursor-pointer shrink-0"
                    aria-label={`Select cluster ${clusterId} for batch refinement`}
                  />
                  <button
                    onClick={() => select(clusterId)}
                    className="flex-1 min-w-0 text-left"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="font-semibold truncate">Cluster {clusterId}</div>
                      <span className="text-[10px] uppercase tracking-wide font-semibold text-slate-700 shrink-0">
                        {LABEL[decision] ?? LABEL.unknown}
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-500 mt-1">
                      {count} cells · capacity {capMean?.toFixed?.(3) ?? "—"}
                      {mmdMean !== undefined && ` · CDDI ${mmdMean?.toFixed?.(3)}`}
                    </div>
                    {reasons.length > 0 && (
                      <ul className="mt-2 space-y-0.5">
                        {reasons.map((r, i) => (
                          <li key={i} className="text-[11px] text-slate-700 break-words">
                            · {r}
                          </li>
                        ))}
                      </ul>
                    )}
                  </button>
                </div>
              ) : (
                // Sub-scope (RefinementAgent output) — no tickbox / refine
                // button; sub-decisions are terminal.
                <button onClick={() => select(clusterId)} className="w-full text-left">
                  <div className="flex items-center justify-between gap-2">
                    <div className="font-semibold truncate">Sub-Cluster {clusterId}</div>
                    <span className="text-[10px] uppercase tracking-wide font-semibold text-slate-700">
                      {LABEL[decision] ?? LABEL.unknown}
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-500 mt-1">
                    {count} cells · capacity {capMean?.toFixed?.(3) ?? "—"}
                    {mmdMean !== undefined && ` · CDDI ${mmdMean?.toFixed?.(3)}`}
                  </div>
                  {reasons.length > 0 && (
                    <ul className="mt-2 space-y-0.5">
                      {reasons.map((r, i) => (
                        <li key={i} className="text-[11px] text-slate-700 break-words">
                          · {r}
                        </li>
                      ))}
                    </ul>
                  )}
                </button>
              )}

              {/* Per-card Refine button kept for the legacy "one cluster at a
                  time" shortcut. The new multi-select bar above is the
                  preferred flow for combining several. */}
              {scope === "main" && (
                <div className="mt-2 pt-2 border-t border-white/40 flex justify-end">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      refine(clusterId);
                    }}
                    disabled={refineSubmitting}
                    className="text-[11px] px-2 py-0.5 rounded border border-battery-electrolyte/30 text-battery-electrolyte hover:bg-battery-electrolyte/10 transition-colors font-medium"
                    title="Refine just this cluster"
                  >
                    {refineSubmitting ? "Refining..." : "Refine only →"}
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
