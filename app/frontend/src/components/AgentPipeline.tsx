import clsx from "clsx";
import useSession from "@/store/sessionStore";
import type { NodeId } from "@/types";

const NODES: { id: NodeId; label: string; color: string }[] = [
  { id: "router", label: "Router", color: "agent-router" },
  { id: "designer", label: "Designer", color: "agent-designer" },
  { id: "compute", label: "Compute", color: "agent-compute" },
  { id: "cluster", label: "Cluster", color: "agent-cluster" },
  { id: "refine", label: "Refine", color: "agent-refine" },
];

export default function AgentPipeline() {
  const pipeline = useSession((s) => s.pipeline);
  const active = useSession((s) => s.activeAgent);

  return (
    <div className="card p-4">
      <div className="text-xs uppercase tracking-widest text-slate-500 font-medium mb-3">
        Agent Pipeline
      </div>
      <div className="flex items-center gap-2">
        {NODES.map((n, i) => {
          const state = pipeline[n.id];
          const isActive = active === n.id || state === "running";
          return (
            <div key={n.id} className="flex items-center flex-1 min-w-0">
              <div className="flex flex-col items-center min-w-0">
                <div
                  className={clsx(
                    "w-12 h-12 rounded-full grid place-items-center font-semibold text-white text-sm transition-all",
                    state === "idle" && "bg-slate-200 text-slate-500",
                    state === "running" && "bg-battery-electrolyte animate-pulse_ring",
                    state === "done" && "bg-battery-charge",
                    state === "error" && "bg-battery-danger",
                    isActive && "scale-110",
                  )}
                  title={`${n.label}: ${state}`}
                >
                  {n.label[0]}
                </div>
                <div
                  className={clsx(
                    "mt-1.5 text-[11px] font-medium truncate",
                    state === "idle" && "text-slate-500",
                    state === "running" && "text-battery-electrolyte",
                    state === "done" && "text-battery-charge",
                    state === "error" && "text-battery-danger",
                  )}
                >
                  {n.label}
                </div>
              </div>
              {i < NODES.length - 1 && (
                <div className="flex-1 h-1 rounded-full mx-1 bg-slate-200 overflow-hidden">
                  {(pipeline[n.id] === "done" || pipeline[n.id] === "running") && (
                    <div
                      className={clsx(
                        "h-full",
                        pipeline[n.id] === "running" ? "pipeline-wire" : "bg-battery-charge",
                      )}
                      style={{ width: "100%" }}
                    />
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
