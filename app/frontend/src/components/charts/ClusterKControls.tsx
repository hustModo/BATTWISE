import { useState } from "react";
import clsx from "clsx";
import useSession from "@/store/sessionStore";

/**
 * Inline control strip above the cluster charts. Lets the user lock the
 * number of clusters (K) for the next ClusterAgent run without typing.
 *
 * - "Auto" → router sends "Run clustering." (no target_k, GA-KMeans picks K).
 * - "K = N" → router sends "Run clustering with k=N." → ClusterAgent receives
 *   target_k=N and pins min_k = max_k = N in clustering_runner.cluster_dataset.
 *
 * The choice is mirrored in the chat panel so users can see exactly what
 * command was dispatched (full audit trail through Router/Pipeline).
 */
export default function ClusterKControls() {
  const sendMessage = useSession((s) => s.sendMessage);
  const lastK = useSession((s) => s.charts.cluster_scatter?.k);
  const running = useSession((s) => s.pipeline?.cluster === "running");

  // Local UI state: which K the user has selected. We default to the last
  // run's K so the control reflects current chart state.
  const [k, setK] = useState<number | "auto">(lastK ?? "auto");

  const runWithK = async (value: number | "auto") => {
    setK(value);
    if (running) return;
    const cmd =
      value === "auto"
        ? "Run clustering in auto mode."
        : `Run clustering with k=${value}.`;
    await sendMessage(cmd);
  };

  const options: Array<number | "auto"> = ["auto", 2, 3, 4, 5, 6, 7, 8];

  return (
    <div className="card px-4 py-2 flex items-center gap-3 flex-wrap">
      <span className="text-xs uppercase tracking-widest text-slate-500 font-medium">
        Cluster K
      </span>
      <div className="flex gap-1 flex-wrap">
        {options.map((opt) => (
          <button
            key={String(opt)}
            disabled={running}
            onClick={() => runWithK(opt)}
            className={clsx(
              "px-2.5 py-1 rounded-md text-xs font-medium border transition-colors",
              k === opt
                ? "bg-battery-electrolyte text-white border-battery-electrolyte"
                : "bg-white text-slate-700 border-slate-200 hover:bg-slate-50",
              running && "opacity-50 cursor-not-allowed",
            )}
            title={
              opt === "auto"
                ? "GA-KMeans picks K adaptively"
                : `Force exactly ${opt} clusters`
            }
          >
            {opt === "auto" ? "Auto" : `K=${opt}`}
          </button>
        ))}
      </div>
      {lastK !== undefined && (
        <span className="text-[11px] text-slate-500 ml-auto">
          current: K={lastK}
        </span>
      )}
      {running && (
        <span className="text-[11px] text-battery-electrolyte animate-pulse">
          re-clustering…
        </span>
      )}
    </div>
  );
}
